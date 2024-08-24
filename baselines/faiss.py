import torch
import faiss
import numpy as np
import time


def faiss_search(
    queries: torch.Tensor,
    keys: torch.Tensor,
    k: int,
    biggest_allocation_memory: int,
    device: str,
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
    H = queries.shape[1]
    N = queries.shape[2]
    kq_dim = queries.shape[3]

    res = faiss.StandardGpuResources()

    for b in range(B):
        for h in range(H):
            begin = time.time()
            keys_b_h = keys[b, h].cpu().numpy()
            queries_b_h = queries[b, h].cpu().numpy()
            torch.cuda.synchronize()
            end = time.time()

            print("putting keys on cpu:", end - begin)

            # Create the index
            nlist = 100  # not sure what this does. number of clusters?
            quantizer = faiss.IndexFlatL2(kq_dim)
            index_ivf = faiss.IndexIVFFlat(
                quantizer, kq_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )  # build a flat CPU index
            if device == "cuda":
                device_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
            elif device == "cpu":
                device_index_ivf = index_ivf
            else:
                raise ValueError(f"device {device} not supported")

            assert not device_index_ivf.is_trained
            # Add the keys to the index
            begin = time.time()
            device_index_ivf.train(keys_b_h)
            end = time.time()
            print("training:", end - begin)
            assert device_index_ivf.is_trained

            begin = time.time()
            device_index_ivf.add(keys_b_h)
            end = time.time()
            print("adding:", end - begin)

            # Search the index
            begin = time.time()
            topk_keys, topk_ids = device_index_ivf.search(queries_b_h, k=k)
            end = time.time()
            print("searching:", end - begin)
