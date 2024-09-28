import torch
import math
from tqdm import tqdm


@torch.no_grad()
def batched_naive_sparse_nearest_k_keys(
    queries: torch.Tensor, keys: torch.Tensor, k,
    biggest_allocation_memory: int,
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N, k] tensor.

    Args:
        queries: [**, num_heads, N, kq_dim]
        keys: [**, num_heads, N, kq_dim]

    Returns:
        [**, num_heads, N, k]
    """

    N = keys.shape[-2]
    batch_size = math.ceil(
        biggest_allocation_memory / (4 * N)
    )  # choose batch size such that the number of bytes in the tensors is below biggest_allocation_memory
    num_batches = math.ceil(N / batch_size)
    print(f"batch_size: {batch_size}, num_batches: {num_batches}")

    result = torch.empty_like(queries[..., :k])
    for i in tqdm(list(range(num_batches))):
        queries_batch = queries[..., i * batch_size : (i + 1) * batch_size, :]

        result[..., i * batch_size : (i + 1) * batch_size, :] = naive_sparse_nearest_k_keys_(
            queries_batch, keys, k
        )

    return result



@torch.no_grad()
def naive_sparse_nearest_k_keys_(
    queries: torch.Tensor, keys: torch.Tensor, k
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N, k] tensor.
    """

    # --- Compute the nearest k keys: [**, num_heads, N, k] ---
        
    full_attention_weights = torch.einsum("...ik,...jk->...ij", queries, keys)

    # [**, num_heads, N, k]
    return full_attention_weights.topk(k, dim=-1)[1]
