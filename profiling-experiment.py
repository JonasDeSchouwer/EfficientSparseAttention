import torch
import torch.nn.functional as F
import sparse_attention
from pykeops.torch import LazyTensor
import math
import time
import os
import shutil

from baselines.symbolic_sparse import symbolic_sparse_nearest_k_keys


# Create empty folder to save the keys and queries
SAVE_DIR = "kqv"
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
kq_dim = 50
val_dim = 50
k = 10
maxLeafSize = 10

sqrt10 = math.sqrt(10)
Ns = [
    1e2,
    sqrt10 * 1e2,
    1e3,
    sqrt10 * 1e3,
    1e4,
    sqrt10 * 1e4,
    1e5,
    sqrt10 * 1e5,
    1e6,
]
Ns = list(map(int, Ns))
sparse_cpp_times = []
sparse_symbolic_times = []
gather_times = []
other_times = []

for N in Ns:
    # generate queries and keys and save them to SAVE_DIR
    print(f"generating queries & keys for N = {N}")
    queries = torch.randn((B, H, N, kq_dim))
    keys = torch.randn((B, H, N, kq_dim))
    values = torch.randn((B, H, N, kq_dim))
    torch.save(queries, f"{SAVE_DIR}/queries-{N}.pt")
    torch.save(keys, f"{SAVE_DIR}/keys-{N}.pt")
    torch.save(values, f"{SAVE_DIR}/values-{N}.pt")


# do sparse symbolic profiling
print("\n--- profiling sparse symbolic ---")
for N in Ns:
    # load queries and keys
    queries = torch.load(f"{SAVE_DIR}/queries-{N}.pt")
    keys = torch.load(f"{SAVE_DIR}/keys-{N}.pt")
    values = torch.load(f"{SAVE_DIR}/values-{N}.pt")

    print(f"N = {N}: ", end=" ")
    begin = time.time()
    nearest_key_indices = symbolic_sparse_nearest_k_keys(queries, keys, k)
    end = time.time()
    sparse_symbolic_times.append(end - begin)
    print(end - begin)

    begin = time.time()
    nearest_keys = torch.gather(
        input=keys.unsqueeze(-2).expand(
            *keys.shape[:-1], k, kq_dim
        ),  # [**, num_heads, N, k, kq_dim]
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(
            *nearest_key_indices.shape, kq_dim
        ),  # [**, num_heads, N, k, kq_dim]
        # sparse_grad=True,
    )
    nearest_values = torch.gather(
        input=values.unsqueeze(-2).expand(
            *keys.shape[:-1], k, val_dim
        ),  # [**, num_heads, N, k, kq_dim]
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(
            *nearest_key_indices.shape, val_dim
        ),  # [**, num_heads, N, k, kq_dim]
        # sparse_grad=True,
    )
    end = time.time()
    gather_times.append(end - begin)
    print("gather time:", end - begin)

    begin = time.time()
    queries_extended = queries.unsqueeze(-2).expand(*queries.shape[:-1], k, kq_dim)
    largest_attention_weights = (queries_extended * nearest_keys).sum(-1) / math.sqrt(
        kq_dim
    )
    largest_attention_weights = F.softmax(largest_attention_weights, dim=-1)
    out = (largest_attention_weights.unsqueeze(-1) * nearest_values).sum(
        dim=-2
    )  # sum over k nearest keys for each query
    end = time.time()
    other_times.append(end - begin)
    print("other time:", end - begin)

print("sparse symbolic times:", sparse_symbolic_times)
print("gather times:", gather_times)
print("other times:", other_times)


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


# remove the folder and its contents
shutil.rmtree(SAVE_DIR)
