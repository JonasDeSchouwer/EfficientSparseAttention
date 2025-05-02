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
    detailed_profiling=False,
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]
        biggest_allocation_memory: int, the biggest possible allocation in bytes
        detailed_profiling: bool, whether to print detailed profiling information

    Returns:
        [B, H, N, val_dim]
    """

    B = queries.shape[-4]
    H = queries.shape[-3]
    N = keys.shape[-2]
    val_dim = values.shape[-1]
    available_memory = int(
        min([torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())])
    )

    # the amount of GPU memory necessary is ct_mem + #queries * marg_mem_per_query
    ct_mem = torch.cuda.memory_allocated(values)
    marg_mem_per_query = B * H *2 * queries.element_size() * N + val_dim
    batch_size = math.ceil(
        (available_memory - ct_mem) / (marg_mem_per_query)
    )  # choose batch size such that the number of bytes in the tensors is below available_memory
    num_batches = math.ceil(N / batch_size)

    result = torch.empty_like(values)
    for i in range(num_batches):
        queries_batch = queries[:, :, i * batch_size : (i + 1) * batch_size]

        result[:, :, i * batch_size : (i + 1) * batch_size] = full_MHA_(
            queries_batch, keys, values, detailed_profiling
        )

    return result


def full_MHA_(
    queries_batch: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    detailed_profiling: bool,
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer.

    Args:
        queries_batch: [B, H, N_batch, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]
        detailed_profiling: bool, whether to print detailed profiling information
    """

    kq_dim = queries_batch.shape[-1]

    begin = time.time()
    # [B, H, N_batch, N]
    full_attention_weights = torch.einsum(
        "bhqd,bhkd->bhqk", queries_batch, keys
    ) / math.sqrt(kq_dim)
    if detailed_profiling:
        torch.cuda.synchronize()
        print("Einsum 1 time:", time.time() - begin)

    begin = time.time()
    full_attention_weights = F.softmax(full_attention_weights, dim=-1)
    if detailed_profiling:
        torch.cuda.synchronize()
        print("Softmax time:", time.time() - begin)

    # [B, H, N_batch, val_dim]
    begin = time.time()
    out = torch.einsum("bhqk,bhkd->bhqd", full_attention_weights, values)
    if detailed_profiling:
        torch.cuda.synchronize()
        print("Einsum 2 time:", time.time() - begin)

    return out
