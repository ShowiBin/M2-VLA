import os
import random
from typing import Callable, Optional
import numpy as np
import torch

def set_global_seed(seed: int, get_worker_init_fn: bool=False) -> Optional[Callable[[int], None]]:
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, 'Seed outside the np.uint32 bounds!'
    os.environ['EXPERIMENT_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return worker_init_function if get_worker_init_fn else None

def worker_init_function(worker_id: int) -> None:
    (global_rank, process_seed) = (int(os.environ['LOCAL_RANK']), torch.initial_seed())
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_seq.generate_state(4))
    (torch_seed_seq, random_seed_seq) = seed_seq.spawn(2)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist
        return torch.version.cuda is not None and torch.cuda.is_bf16_supported() and (packaging.version.parse(torch.version.cuda).release >= (11, 0)) and dist.is_nccl_available() and (nccl.version() >= (2, 10))
    except Exception:
        return False

def sequence_combine_call_split(sequence: torch.Tensor, fn: Callable):
    (B, T) = sequence.shape[:2]
    flat_sequence = sequence.reshape([-1, *sequence.shape[2:]])
    flat_outputs = fn(flat_sequence)
    return flat_outputs.reshape([B, T, *flat_outputs.shape[1:]])

def merge_two_dims(tensor: torch.Tensor, start_dim: int=0):
    if start_dim < 0:
        start_dim = len(tensor.shape) + start_dim
        assert start_dim >= 0
    assert len(tensor.shape) > start_dim + 1, 'Start dimension for merge is too big!'
    return tensor.reshape([*tensor.shape[:start_dim], -1, *tensor.shape[start_dim + 2:]])
