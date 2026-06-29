import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy
overwatch = initialize_overwatch(__name__)

class FSDPStrategy(TrainingStrategy):

    def __init__(self, vlm: PrismaticVLM, device_id: int, stage: str, epochs: int, max_steps: Optional[int], global_batch_size: int, per_device_batch_size: int, learning_rate: float, weight_decay: float, max_grad_norm: float, lr_scheduler_type: str, warmup_ratio: float, enable_gradient_checkpointing: bool=True, enable_mixed_precision_training: bool=True, reduce_in_full_precision: bool=False, mixed_precision_dtype: torch.dtype=torch.bfloat16, worker_init_fn: Optional[Callable[[int], None]]=None, sharding_strategy: str='shard-grad-op', state_dict_type: StateDictType=StateDictType.FULL_STATE_DICT) -> None:
        super().__init__(vlm=vlm, device_id=device_id, stage=stage, epochs=epochs, max_steps=max_steps, global_batch_size=global_batch_size, per_device_batch_size=per_device_batch_size, learning_rate=learning_rate, weight_decay=weight_decay, max_grad_norm=max_grad_norm, lr_scheduler_type=lr_scheduler_type, warmup_ratio=warmup_ratio, enable_gradient_checkpointing=enable_gradient_checkpointing, enable_mixed_precision_training=enable_mixed_precision_training, reduce_in_full_precision=reduce_in_full_precision, mixed_precision_dtype=mixed_precision_dtype, worker_init_fn=worker_init_fn)
        if sharding_strategy == 'shard-grad-op':
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == 'full-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f'FSDP Sharding Strategy {sharding_strategy} is not supported!')
        assert state_dict_type == StateDictType.FULL_STATE_DICT, 'Sharded state saving is not yet implemented!'
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(self, run_dir: Path, global_step: int, epoch: int, train_loss: Optional[float]=None, only_trainable: bool=True) -> None:
        assert isinstance(self.vlm, FSDP), 'FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!'
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            full_vlm_state_dict = self.vlm.state_dict()
            model_state_dicts = {mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)}
            for (key, param) in full_vlm_state_dict.items():
                for mkey in model_state_dicts:
                    if key.startswith((mprefix := f'{mkey}.')):
                        model_state_dicts[mkey][key.removeprefix(mprefix)] = param
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / 'checkpoints'
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f'step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt'
                else:
                    checkpoint_path = checkpoint_dir / f'step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt'
                torch.save({'model': model_state_dicts}, checkpoint_path)

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype)
            if self.stage not in {'full-finetune', 'vla-full-train', 'vla-sandwich-train'}:
                overwatch.info('Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`')
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)
        else:
            fsdp_precision_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        self.vlm = FSDP(self.vlm, auto_wrap_policy=vlm_fsdp_wrapping_policy, mixed_precision=fsdp_precision_policy, sharding_strategy=self.fsdp_sharding_strategy, device_id=torch.cuda.current_device(), limit_all_gathers=True, use_orig_params=True)
        if self.enable_gradient_checkpointing:
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
        dist.barrier()
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = n_train_examples * self.epochs // self.global_batch_size
        else:
            num_training_steps = self.max_steps
        if self.lr_scheduler_type == 'linear-warmup+cosine-decay':
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            (decay, no_decay) = ([], [])
            for (name, param) in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [{'params': decay, 'weight_decay': self.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0
        elif self.lr_scheduler_type == 'constant':
            num_warmup_steps = 0
            (decay, no_decay) = ([], [])
            for (name, param) in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [{'params': decay, 'weight_decay': self.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f'Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!')
        overwatch.info(f'FSDP Full-Shard Strategy =>> Finalized Training Setup:\n         |-> Global (Effective) Batch Size = {self.global_batch_size}\n         |-> Per-Device Batch Size = {self.per_device_batch_size}\n         |-> Distributed World Size = {overwatch.world_size()}\n         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n         |-> Default AdamW LR = {self.learning_rate}\n         |-> AdamW Weight Decay = {self.weight_decay}\n         |-> LR Scheduler Type = {self.lr_scheduler_type}\n         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n         |-> Dataset Size = {n_train_examples} Examples\n         |-> Max Steps = {num_training_steps}\n')

    def clip_grad_norm(self) -> None:
        self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)
