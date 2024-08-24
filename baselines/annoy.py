from annoy import AnnoyIndex
import torch

import numpy as np


NUM_TREES = 10


def annoy_search(
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
            keys_np = keys.cpu().numpy()

            # Create the index
            index: AnnoyIndex = AnnoyIndex(
                kq_dim,
                "dot",
            )

            for n in range(N):
                index.add_item(n, keys_np[b, h, n])

            index.build(NUM_TREES)

            # Search the index
            topk_keys = index.get_nns_by_item
