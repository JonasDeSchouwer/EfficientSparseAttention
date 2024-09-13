import torch
import faiss
import numpy as np
import time

@torch.no_grad()
def faiss_search(
    queries: torch.Tensor,
    keys: torch.Tensor,
    k: int,
    biggest_allocation_memory: int,
    device: str,
    nlist=None,
    nprobe=None,
    detailed_profiling=False,
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

    # --- faiss hyperparameters ---

    # the number of inverted lists in the index structure
    # 4 * sqrt(n) is usually reasonable, or some other O(sqrt(n)) -> https://github.com/facebookresearch/faiss/issues/112
    # It's a tradeoff between speed (high nlist) against accuracy (low nlist)
    if nlist is None:
        nlist = 4 * int(np.sqrt(N))

    # the number of inverted lists to visit during search
    if nprobe is None:
        nprobe = 30

    num_sub_quantizers = kq_dim  # should be a divisor of the vector dimension
    nbits_per_subquantizer = 8


    # --- Create the index ---
    begin = time.time()
    quantizer = faiss.IndexFlatIP(kq_dim)
    # quantizer.reserve(N)    # reserve memory for N vectors
    index_ivf = faiss.IndexIVFFlat(
        quantizer, kq_dim, nlist, faiss.METRIC_INNER_PRODUCT
    )
    # index_ivf = faiss.IndexIVFPQ(
    #     quantizer, kq_dim, nlist, num_sub_quantizers, nbits_per_subquantizer, faiss.METRIC_INNER_PRODUCT
    # )
    index_ivf.nprobe = nprobe
    index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    if device == "cuda":
        device_index_ivf = faiss.index_cpu_to_gpu(
            provider=res,
            device=0,
            index=index_ivf
    )
    elif device == "cpu":
        device_index_ivf = index_ivf
    else:
        raise ValueError(f"device {device} not supported")
    if detailed_profiling:
        torch.cuda.synchronize()
        end = time.time()
        print("creating index:", end - begin)
    device_index_ivf.verbose = False    # silence the warning about too few training points

    topk_ids = np.empty((B, H, N, k), dtype=np.int64)


    for b in range(B):
        for h in range(H):
            begin = time.time()
            device_index_ivf.reset()    # clears the contents but keeps any learned parameters from training
            if detailed_profiling:
                torch.cuda.synchronize()
                end = time.time()
                print("resetting index:", end - begin)

            begin = time.time()
            keys_b_h = keys[b, h].cpu().numpy()
            queries_b_h = queries[b, h].cpu().numpy()
            if detailed_profiling:
                torch.cuda.synchronize()
                end = time.time()
                print("putting keys on cpu:", end - begin)

             # Train the index on the keys of the first head of the first graph
            device_index_ivf.cp.min_points_per_centroid = 1 # to silence the warning about too few training points
            begin = time.time()
            device_index_ivf.train(keys[b,h].cpu().numpy())
            end = time.time()
            if detailed_profiling:
                torch.cuda.synchronize()
                print("training:", end - begin)

            begin = time.time()
            device_index_ivf.add(keys_b_h)
            if detailed_profiling:
                torch.cuda.synchronize()
                end = time.time()
                print("adding:", end - begin)

            # Search the index
            begin = time.time()
            # topk_keys: [N,k]
            # topk_ids: [N,k]
            device_index_ivf.search(queries_b_h, k=k, I=topk_ids[b,h])
            if detailed_profiling:
                torch.cuda.synchronize()
                end = time.time()
                print("searching:", end - begin)

    begin = time.time()
    topk_ids = torch.tensor(topk_ids).to(queries.device)
    if detailed_profiling:
        torch.cuda.synchronize()
        end = time.time()
        print("putting topk_ids on device:", end - begin)

    return topk_ids


def create_index(index_type: str, parameters: dict):
    if index_type == "faiss":
        return FaissIndex(parameters)
    else:
        raise ValueError(f"index_type {index_type} not supported")
