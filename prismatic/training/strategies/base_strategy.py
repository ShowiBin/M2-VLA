from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.training.train_utils import compute_actions_l1_loss, compute_token_accuracy, get_current_action_mask, get_next_actions_mask
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, NUM_ACTIONS_CHUNK, IGNORE_INDEX
NEWLINE_INDEX = 13
STOP_INDEX = 2
overwatch = initialize_overwatch(__name__)

class TrainingStrategy(ABC):

    def __init__(self, vlm: PrismaticVLM, device_id: int, stage: str, epochs: int, max_steps: Optional[int], global_batch_size: int, per_device_batch_size: int, learning_rate: float, weight_decay: float, max_grad_norm: float, lr_scheduler_type: str, warmup_ratio: float, enable_gradient_checkpointing: bool=True, enable_mixed_precision_training: bool=True, reduce_in_full_precision: bool=False, mixed_precision_dtype: torch.dtype=torch.bfloat16, worker_init_fn: Optional[Callable[[int], None]]=None, **_: str) -> None:
        (self.vlm, self.device_id, self.stage) = (vlm, device_id, stage)
        (self.all_module_keys, self.trainable_module_keys) = (self.vlm.all_module_keys, self.vlm.trainable_module_keys)
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls
        (self.epochs, self.max_steps) = (epochs, max_steps)
        (self.global_batch_size, self.per_device_batch_size) = (global_batch_size, per_device_batch_size)
        (self.learning_rate, self.weight_decay, self.max_grad_norm) = (learning_rate, weight_decay, max_grad_norm)
        (self.lr_scheduler_type, self.warmup_ratio) = (lr_scheduler_type, warmup_ratio)
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.worker_init_fn = worker_init_fn
        (self.optimizer, self.lr_scheduler) = (None, None)
        assert self.global_batch_size % self.per_device_batch_size == 0, 'Per-device batch size must evenly divide global batch size!'
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, 'Only BF16 mixed precision training is supported!'
            assert check_bloat16_supported(), 'BFloat16 is not supported on this hardware; unset `mixed_precision`'

    @abstractmethod
    def save_checkpoint(self, run_dir: Path, global_step: int, epoch: int, train_loss: Optional[float]=None, only_trainable: bool=True) -> None:
        ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        ...

    @abstractmethod
    def clip_grad_norm(self) -> None:
        ...

    def run_training(self, dataset: Dataset, collator: PaddedCollatorForLanguageModeling, metrics: Metrics, stage: str='finetune', batch_construction_strategy: str='split-modality', seed: int=7) -> None:
        if 'finetune' in stage and batch_construction_strategy == 'split-modality':
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(dataset, modality_lengths, global_batch_size=self.global_batch_size, num_replicas=overwatch.world_size(), rank=overwatch.rank(), seed=seed, drop_last=False)
        else:
            sampler = DistributedSampler(dataset, num_replicas=overwatch.world_size(), rank=overwatch.rank(), shuffle=True, seed=seed, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=self.per_device_batch_size, sampler=sampler, collate_fn=collator, num_workers=2, worker_init_fn=self.worker_init_fn)
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            self.epochs = 100
        status = metrics.get_status()
        with tqdm(total=self.epochs * (len(dataloader) // self.grad_accumulation_steps) if self.max_steps is None else self.max_steps, desc=status, leave=False, disable=not overwatch.is_rank_zero()) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)
                self.optimizer.zero_grad()
                for (train_idx, batch) in enumerate(dataloader):
                    with torch.autocast('cuda', dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                        output: CausalLMOutputWithPast = self.vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'], labels=batch['labels'], multimodal_indices=batch['multimodal_indices'])
                        loss = output.loss
                    metrics.commit(loss=loss)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)
                        self.clip_grad_norm()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()
                            return
                        progress.update()
                        progress.set_description(status)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    def run_vla_training(self, vla_dataset: IterableDataset, collator: PaddedCollatorForActionPrediction, action_tokenizer: ActionTokenizer, metrics: VLAMetrics, save_interval: int=2500, save_full_model: bool=True) -> None:
        assert isinstance(vla_dataset, IterableDataset), 'VLA training expects an IterableDataset!'
        assert self.grad_accumulation_steps == 1, 'VLA training does not support gradient accumulation!'
        dataloader = DataLoader(vla_dataset, batch_size=self.per_device_batch_size, sampler=None, collate_fn=collator, num_workers=0, worker_init_fn=self.worker_init_fn)
        status = metrics.get_status()
        with tqdm(total=self.epochs * len(dataloader) if self.max_steps is None else self.max_steps, desc=status, leave=False, disable=not overwatch.is_rank_zero()) as progress:
            self.vlm.train()
            self.optimizer.zero_grad()
            for batch in dataloader:
                with torch.autocast('cuda', dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training):
                    output: CausalLMOutputWithPast = self.vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'], labels=batch['labels'])
                    loss = output.loss
                metrics.commit(loss=loss)
                loss.backward()
                predicted_token_ids = output.logits[:, self.vlm.vision_backbone.num_patches:-1].argmax(dim=2)
                ground_truth_token_ids = batch['labels'][:, 1:].to(predicted_token_ids.device)
                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
                action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
                next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, next_actions_accuracy=next_actions_accuracy, next_actions_l1_loss=next_actions_l1_loss, update_step_time=True)
                if overwatch.is_rank_zero():
                    datasets = set(batch['dataset_names'])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch['dataset_names']])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            pred_continuous_actions_ds = torch.tensor(action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[ds_mask][mask[ds_mask]].cpu().numpy()))
                            continuous_actions_gt_ds = torch.tensor(action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[ds_mask][mask[ds_mask]].cpu().numpy()))
                            action_l1_loss_ds = torch.nn.functional.l1_loss(pred_continuous_actions_ds, continuous_actions_gt_ds)
                            metrics.commit_for_dataset(dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds, next_actions_accuracy=next_actions_accuracy, next_actions_l1_loss=next_actions_l1_loss)
                self.clip_grad_norm()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or metrics.global_step % save_interval == 0:
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model)
                    dist.barrier()
                    if terminate:
                        return
                progress.update()
                progress.set_description(status)
