import torch
from argparse import Namespace
import math


def full_MHA(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]
    """

    kq_dim = queries.shape[-1]
    # [B, H, N, N]
    full_attention_weights = torch.einsum("bhqd,bhkd->bhqk", queries, keys) / math.sqrt(
        kq_dim
    )

    # [B, H, N, val_dim]
    out = torch.einsum("bhqk,bhkd->bhqd", full_attention_weights, values)

    return out
