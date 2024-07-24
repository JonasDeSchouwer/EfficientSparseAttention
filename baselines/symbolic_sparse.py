import torch
from pykeops.torch import LazyTensor
import math


def batched_symbolic_sparse_nearest_k_keys(
    queries: torch.Tensor, keys: torch.Tensor, k, biggest_allocation_memory
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [B, H, N, k] tensor.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        k: int
        biggest_allocation_memory: int, the biggest possible allocation in bytes

    Returns:
        [B, H, N, k]
    """
    B = queries.shape[0]
    H = queries.shape[1]
    N = queries.shape[2]

    batch_size = math.ceil(
        biggest_allocation_memory / (4 * N)
    )  # choose batch size such that the number of bytes in the LazyTensors is below biggest_allocation_memory
    num_batches = math.ceil(N / batch_size)

    result = torch.empty((B, H, N, k), device=queries.device)
    for i in range(num_batches):
        queries_batch = queries[:, :, i * batch_size : (i + 1) * batch_size]
        result[:, :, i * batch_size : (i + 1) * batch_size] = (
            symbolic_sparse_nearest_k_keys_(queries_batch, keys, k)
        )

    return result


def symbolic_sparse_nearest_k_keys_(
    queries_batch: torch.Tensor, keys: torch.Tensor, k
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N_batch, k] tensor.

    Args:
        queries: [**, num_heads, N_batch, kq_dim]
        keys: [**, num_heads, N, kq_dim]

    Returns:
        [**, num_heads, N_batch, k]
    """
    # --- Compute the nearest k keys: [**, num_heads, N, k] ---

    # [**, num_heads, N, 1, kq_dim]
    queries_extended: LazyTensor = LazyTensor(queries_batch[..., :, :, None, :])
    #  [**, num_heads, 1, N, kq_dim]
    keys_extended: LazyTensor = LazyTensor(keys[..., :, None, :, :])
    # [**, num_heads, N, N]
    full_attention_weights: LazyTensor = (queries_extended * keys_extended).sum(
        -1
    )  # / math.sqrt(kq_dim) does not matter for ordering

    # [**, num_heads, N, k]
    ndims = len(full_attention_weights.shape)
    assert ndims in [3, 4]
    nearest_key_indices = (-full_attention_weights).argKmin(k, dim=ndims - 1)

    return nearest_key_indices
