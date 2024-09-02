import torch
import torch.nn.functional as F
from argparse import Namespace
import math
import time


def batched_full_MHA(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    biggest_allocation_memory: int,
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]
        biggest_allocation_memory: int, the biggest possible allocation in bytes

    Returns:
        [B, H, N, val_dim]
    """

    N = keys.shape[-2]
    batch_size = math.ceil(
        biggest_allocation_memory / (4 * N)
    )  # choose batch size such that the number of bytes in the LazyTensors is below biggest_allocation_memory
    num_batches = math.ceil(N / batch_size)

    result = torch.empty_like(values)
    for i in range(num_batches):
        queries_batch = queries[:, :, i * batch_size : (i + 1) * batch_size]

        result[:, :, i * batch_size : (i + 1) * batch_size] = full_MHA_(
            queries_batch, keys, values
        )

    return result


def full_MHA_(
    queries_batch: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer.

    Args:
        queries_batch: [B, H, N_batch, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]
    """

    kq_dim = queries_batch.shape[-1]

    # begin = time.time()
    # [B, H, N_batch, N]
    full_attention_weights = torch.einsum(
        "bhqd,bhkd->bhqk", queries_batch, keys
    ) / math.sqrt(kq_dim)
    # torch.cuda.synchronize()
    # print("Einsum 1 time:", time.time() - begin)

    # begin = time.time()
    full_attention_weights = F.softmax(full_attention_weights, dim=-1)
    # torch.cuda.synchronize()
    # print("Softmax time:", time.time() - begin)

    # [B, H, N_batch, val_dim]
    # begin = time.time()
    out = torch.einsum("bhqk,bhkd->bhqd", full_attention_weights, values)
    # torch.cuda.synchronize()
    # print("Einsum 2 time:", time.time() - begin)

    return out
