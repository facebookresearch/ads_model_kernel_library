# pyre-ignore-all-errors

import torch


def should_use_i64_idx(*tensors: torch.Tensor) -> bool:
    return any(isinstance(t, torch.Tensor) and t.numel() >= 2**31 for t in tensors)
