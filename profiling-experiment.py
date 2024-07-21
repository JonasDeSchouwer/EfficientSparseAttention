import torch
import sparse_attention
from pykeops.torch import LazyTensor
import math
import time
import os
import shutil


# Create empty folder to save the keys and queries
SAVE_DIR = "keys-queries"
if os.path.exists(SAVE_DIR):
    response = input(
        f"Folder {SAVE_DIR} already exists. Do you want to delete it? (y/n) "
    )
    if response == "y":
        shutil.rmtree(SAVE_DIR)
        os.mkdir(SAVE_DIR)
    else:
        print("Please delete the folder and run the script again.")
        exit()
else:
    os.mkdir(SAVE_DIR)

B = 2
H = 3
N = 10
kq_dim = 5
k = 3
maxLeafSize = 10


def symbolic_sparse_nearest_k_keys(
    queries: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """
    Compute the nearest k keys for each query. Return their indices in a [**, num_heads, N, k] tensor.

    Args:
        queries: [**, num_heads, N, kq_dim]
        keys: [**, num_heads, N, kq_dim]

    Returns:
        [**, num_heads, N, k]
    """
    # --- Compute the nearest k keys: [**, num_heads, N, k] ---

    # [**, num_heads, N, 1, kq_dim]
    queries_extended: LazyTensor = LazyTensor(queries[..., :, :, None, :])
    #  [**, num_heads, 1, N, kq_dim]
    keys_extended: LazyTensor = LazyTensor(keys[..., :, None, :, :])
    # [**, num_heads, N, N]
    full_attention_weights: LazyTensor = (queries_extended * keys_extended).sum(
        -1
    ) / math.sqrt(kq_dim)

    # [**, num_heads, N, k]
    ndims = len(full_attention_weights.shape)
    assert ndims in [3, 4]
    nearest_key_indices = (-full_attention_weights).argKmin(k, dim=ndims - 1)

    return nearest_key_indices


Ns = [1e2, 1e3, 1e4]
Ns = list(map(int, Ns))
sparse_cpp_times = []
sparse_symbolic_times = []

for N in Ns:
    # generate queries and keys and save them to SAVE_DIR
    print(f"generating queries & keys for N = {N}")
    queries = torch.randn((B, H, N, kq_dim))
    keys = torch.randn((B, H, N, kq_dim))
    torch.save(queries, f"{SAVE_DIR}/queries-{N}.pt")
    torch.save(keys, f"{SAVE_DIR}/keys-{N}.pt")

# do sparse C++ profiling
print("\n--- profiling sparse C++ ---")
for N in Ns:
    # load queries and keys
    queries = torch.load(f"{SAVE_DIR}/queries-{N}.pt")
    keys = torch.load(f"{SAVE_DIR}/keys-{N}.pt")

    print(f"N = {N}: ", end=" ")
    begin = time.time()
    sparse_cpp_results = sparse_attention.nearestKKeys(queries, keys, k, maxLeafSize)
    end = time.time()
    sparse_cpp_times.append(end - begin)
    print(end - begin)
print("sparse C++ times:", sparse_cpp_times)

# do sparse symbolic profiling
print("\n--- profiling sparse symbolic ---")
for N in Ns:
    # load queries and keys
    queries = torch.load(f"{SAVE_DIR}/queries-{N}.pt")
    keys = torch.load(f"{SAVE_DIR}/keys-{N}.pt")

    print(f"N = {N}: ", end=" ")
    begin = time.time()
    sparse_symbolic_results = symbolic_sparse_nearest_k_keys(queries, keys)
    end = time.time()
    sparse_symbolic_times.append(end - begin)
    print(end - begin)
print("sparse symbolic times:", sparse_symbolic_times)

# remove the folder and its contents
shutil.rmtree(SAVE_DIR)
