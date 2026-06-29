import math
from typing import Iterator, List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

class SplitModalitySampler(Sampler):

    def __init__(self, dataset: Dataset, modality_lengths: List[Tuple[bool, int]], global_batch_size: int, num_replicas: Optional[int]=None, rank: Optional[int]=None, seed: int=0, drop_last: bool=False) -> None:
        super().__init__()
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        (self.seed, self.epoch) = (seed, 0)
        (self.dataset, self.modality_lengths, self.drop_last) = (dataset, modality_lengths, drop_last)
        self.global_batch_size = global_batch_size
        assert not self.drop_last, 'SplitModalitySampler must set `drop_last = False`!'
        self.total_size = math.ceil(len(self.dataset) / self.global_batch_size) * self.global_batch_size
        self.num_samples = self.total_size // self.num_replicas

    @staticmethod
    def reindex_batch(batch_idxs: List[int], idx2lengths: List[int], n_buckets: int) -> List[List[int]]:
        assert len(batch_idxs) % n_buckets == 0, 'Batch length is not divisible by `num_replicas`!'
        n_examples_per_bucket = len(batch_idxs) // n_buckets
        bucket_indices = [[] for _ in range(n_buckets)]
        bucket_lengths = [0 for _ in range(n_buckets)]
        for idx in batch_idxs:
            shortest_bucket_idx = bucket_lengths.index(min(bucket_lengths))
            bucket_indices[shortest_bucket_idx].append(idx)
            bucket_lengths[shortest_bucket_idx] += idx2lengths[idx]
            if len(bucket_indices[shortest_bucket_idx]) == n_examples_per_bucket:
                bucket_lengths[shortest_bucket_idx] = float('inf')
        return bucket_indices

    def get_modality_and_length_grouped_indices(self, generator: torch.Generator) -> List[int]:
        (multimodal_indices, multimodal_lengths) = zip(*[(idx, length) for (idx, (is_multimodal, length)) in enumerate(self.modality_lengths) if is_multimodal])
        unimodal_split = [(idx, length) for (idx, (is_multimodal, length)) in enumerate(self.modality_lengths) if not is_multimodal]
        if len(unimodal_split) == 0:
            (unimodal_indices, unimodal_lengths) = ([], [])
        else:
            (unimodal_indices, unimodal_lengths) = zip(*unimodal_split)
        mm_shuffled_idxs = torch.randperm(len(multimodal_indices), generator=generator)
        uni_shuffled_idxs = torch.randperm(len(unimodal_indices), generator=generator)
        g_bsz = self.global_batch_size
        mm_batch_idxs = [mm_shuffled_idxs[i:i + g_bsz].tolist() for i in range(0, len(mm_shuffled_idxs), g_bsz)]
        uni_batch_idxs = [uni_shuffled_idxs[i:i + g_bsz].tolist() for i in range(0, len(uni_shuffled_idxs), g_bsz)]
        if len(mm_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(mm_batch_idxs[-1])
            mm_batch_idxs[-1].extend(mm_batch_idxs[0][:n_missing])
        if len(uni_batch_idxs) > 0 and len(uni_batch_idxs[-1]) < g_bsz:
            n_missing = g_bsz - len(uni_batch_idxs[-1])
            uni_batch_idxs[-1].extend(uni_batch_idxs[0][:n_missing])
        mm_sorted_batch_idxs = [sorted(b, key=lambda i: multimodal_lengths[i], reverse=True) for b in mm_batch_idxs]
        uni_sorted_batch_idxs = [sorted(b, key=lambda i: unimodal_lengths[i], reverse=True) for b in uni_batch_idxs]
        mm_length_bucketed_idxs = [self.reindex_batch(batch, multimodal_lengths, self.num_replicas) for batch in mm_sorted_batch_idxs]
        uni_length_bucketed_idxs = [self.reindex_batch(batch, unimodal_lengths, self.num_replicas) for batch in uni_sorted_batch_idxs]
        mm_output_idxs = [idx for batch in mm_length_bucketed_idxs for bucket in batch for idx in bucket]
        mm_reindexed = [multimodal_indices[idx] for idx in mm_output_idxs]
        mm_batches = [mm_reindexed[i:i + g_bsz] for i in range(0, len(mm_reindexed), g_bsz)]
        uni_output_idxs = [idx for batch in uni_length_bucketed_idxs for bucket in batch for idx in bucket]
        uni_reindexed = [unimodal_indices[idx] for idx in uni_output_idxs]
        uni_batches = [uni_reindexed[i:i + g_bsz] for i in range(0, len(uni_reindexed), g_bsz)]
        merged_batches = mm_batches + uni_batches
        merge_idxs = torch.randperm(len(merged_batches), generator=generator)
        all_batches = [merged_batches[idx] for idx in merge_idxs]
        all_lengths = [length + ((_n_patches := (24 * 24)) if is_mm else 0) for (is_mm, length) in self.modality_lengths]
        all_batches_max_lengths = []
        for batch in all_batches:
            all_batches_max_lengths.append(max([all_lengths[idx] for idx in batch]))
        longest_batch_idx = np.argmax(all_batches_max_lengths)
        (all_batches[0], all_batches[longest_batch_idx]) = (all_batches[longest_batch_idx], all_batches[0])
        indices = [idx for batch in all_batches for idx in batch]
        return indices

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.get_modality_and_length_grouped_indices(g)
        assert len(set(indices)) == len(self.modality_lengths) == len(self.dataset), 'Oops!'
        assert len(indices) % self.global_batch_size == 0 and len(indices) % self.num_replicas == 0, 'Oops'
        per_replica_batch_size = self.global_batch_size // self.num_replicas
        indices_t = torch.as_tensor(indices)
        per_replica_batch_indices_t = indices_t.reshape(-1, per_replica_batch_size)
        replica_indices_t = per_replica_batch_indices_t[self.rank::self.num_replicas]
        replica_indices = replica_indices_t.flatten().tolist()
        return iter(replica_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
