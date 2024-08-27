from importlib import import_module
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["patch"]


def _get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    seqlens = []
    for k in range(1, torch.max(attention_mask) + 1):
        seqlens.append(torch.sum(attention_mask == k, dim=1))
    seqlens = torch.stack(seqlens, axis=1).flatten()
    return seqlens[seqlens != 0].to(dtype=torch.int32)


def _get_unpad_data(attention_mask: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = _get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def patch(model: nn.Module) -> None:
    m = import_module(model.__module__)
    if not hasattr(m, "_get_unpad_data"):
        raise ValueError(f"Module {m} does not have function `_get_unpad_data` for packing")
    m._get_unpad_data = _get_unpad_data
