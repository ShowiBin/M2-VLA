import shutil
from pathlib import Path
from typing import Optional
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup
from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy
overwatch = initialize_overwatch(__name__)

class DDPStrategy(TrainingStrategy):

    @overwatch.rank_zero_only
    def save_checkpoint(self, run_dir: Path, global_step: int, epoch: int, train_loss: Optional[float]=None, only_trainable: bool=True) -> None:
        assert isinstance(self.vlm, DDP), 'save_checkpoint assumes VLM is already wrapped in DDP!'
        model_state_dicts = {mkey: getattr(self.vlm.module, mkey).state_dict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)}
        optimizer_state_dict = self.optimizer.state_dict()
        checkpoint_dir = run_dir / 'checkpoints'
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f'step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt'
        else:
            checkpoint_path = checkpoint_dir / f'step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt'
        torch.save({'model': model_state_dicts, 'optimizer': optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / 'latest-checkpoint.pt')

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        if self.enable_gradient_checkpointing:
            overwatch.info('Enabling Gradient Checkpointing on LLM Backbone', ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()
        overwatch.info('Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU', ctx_level=1)
        self.vlm.to(self.device_id)
        overwatch.info('Wrapping VLM with Distributed Data Parallel', ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            num_training_steps = n_train_examples * self.epochs // self.global_batch_size
        else:
            num_training_steps = self.max_steps
        if self.lr_scheduler_type == 'linear-warmup+cosine-decay':
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            assert self.weight_decay == 0, 'DDP training does not currently support `weight_decay` > 0!'
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0
        elif self.lr_scheduler_type == 'constant':
            num_warmup_steps = 0
            assert self.weight_decay == 0, 'DDP training does not currently support `weight_decay` > 0!'
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f'Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!')
        overwatch.info(f'DDP Strategy =>> Finalized Training Setup:\n         |-> Global (Effective) Batch Size = {self.global_batch_size}\n         |-> Per-Device Batch Size = {self.per_device_batch_size}\n         |-> Distributed World Size = {overwatch.world_size()}\n         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n         |-> Default AdamW LR = {self.learning_rate}\n         |-> AdamW Weight Decay = {self.weight_decay}\n         |-> LR Scheduler Type = {self.lr_scheduler_type}\n         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n         |-> Dataset Size = {n_train_examples} Examples\n         |-> Max Steps = {num_training_steps}\n')

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)
