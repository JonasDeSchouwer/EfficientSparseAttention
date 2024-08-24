from torchpq.index import IVFPQIndex
import torch


def torchpq_search(
    queries: torch.Tensor,
    keys: torch.Tensor,
    k: int,
    biggest_allocation_memory: int,
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [B, H, N, k] tensor.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        k: int, number of nearest keys to return
        biggest_allocation_memory: int, the biggest possible allocation in bytes
        device: str, a specific device (e.g. 'cuda:0') to put the index on

    Returns:
        [B, H, N, k]
    """

    B = queries.shape[0]
    num_heads = queries.shape[1]
    N = queries.shape[2]
    kq_dim = queries.shape[3]

    for b in range(B):
        for h in range(num_heads):
            keys_b_h = keys[b, h]

            # Create the index
            index = IVFPQIndex(
                d_vector=kq_dim,
                n_subvectors=12,  # number of subquantizers, essentially the byte size of each quantized vector.
                # divisibility: 4 | n_subvectors | d_vector
                n_cells=1024,  # number of coarse quantizer clusters
                initial_size=N,  # number of vectors in the dataset
                distance="cosine",  # distance metric to use
                device=keys.device,
            )

            # Add the keys to the index
            ids = torch.arange(N, device=keys.device)
            index.add(keys_b_h, ids=ids)

            # Search the index
            topk_keys, topk_ids = index.search(queries[b, h], k=k)
