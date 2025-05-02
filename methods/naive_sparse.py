import torch
import math
from tqdm import tqdm


@torch.no_grad()
def batched_naive_sparse_nearest_k_keys(
    queries: torch.Tensor, keys: torch.Tensor, k,
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N, k] tensor.

    Args:
        queries: [**, num_heads, N, kq_dim]
        keys: [**, num_heads, N, kq_dim]

    Returns:
        [**, num_heads, N, k]
    """

    B = queries.shape[-4]
    H = queries.shape[-3]
    N = keys.shape[-2]
    available_memory = int(
        min([torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())])
    )

    # the amount of GPU memory necessary is ct_mem + #queries * marg_mem_per_query
    ct_mem = torch.cuda.memory_allocated(queries[..., :k])
    safety_ct = 9*ct_mem
    safety_factor = 1.1
    marg_mem_per_query = B * H * queries.element_size() * N
    batch_size = math.ceil(
        (available_memory - ct_mem - safety_ct) / (marg_mem_per_query * safety_factor)
    )  # choose batch size such that the number of bytes in the tensors is below available_memory
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
