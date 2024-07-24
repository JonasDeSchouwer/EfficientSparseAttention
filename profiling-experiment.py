import torch
import sparse_attention
import math
import time
import argparse
import time
import numpy as np

from baselines.symbolic_sparse import batched_symbolic_sparse_nearest_k_keys
from baselines.post_processing import post_processing
from baselines.full import full_MHA


parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--H", type=int, default=1)
parser.add_argument("--kq_dim", type=int, default=50)
parser.add_argument("--val_dim", type=int, default=50)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--maxLeafSize", type=int, default=10)
parser.add_argument(
    "--method", type=str, choices=["sparse_symbolic", "sparse_cpp", "full"]
)
parser.add_argument("--maxN", type=int, default=1e9)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# fix all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

B = args.B
H = args.H
kq_dim = args.kq_dim
val_dim = args.val_dim
k = args.k
maxLeafSize = args.maxLeafSize
num_runs = args.num_runs
method = args.method

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
Ns = list(filter(lambda x: x < args.maxN, map(int, Ns)))


print("method:", method)
print("B:", B)
print("H:", H)
print("kq_dim:", kq_dim)
print("val_dim:", val_dim)
print("k:", k)
print("maxLeafSize:", maxLeafSize)
print("num_runs:", num_runs)
print("Ns:", Ns)

print(f"--- Profiling method {method} ---")

for N in Ns:
    print("\n- N =", N, "-")

    run_attention_times = []
    run_key_search_times = []
    run_post_processing_times = []

    # --- main processing and profiling ---
    for run in range(num_runs + 1):
        # generate queries and keys and save them to SAVE_DIR
        queries = torch.randn((B, H, N, kq_dim))
        keys = torch.randn((B, H, N, kq_dim))
        values = torch.randn((B, H, N, val_dim))

        if args.device == "cuda":
            queries = queries.cuda()
            keys = keys.cuda()
            values = values.cuda()
        else:
            assert (
                args.device == "cpu"
            ), f"device {args.device} not supported: only 'cpu' and 'cuda' are supported"

        if method == "full":
            begin = time.time()
            out = full_MHA(queries, keys, values)
            end = time.time()
            run_attention_times.append(end - begin)

        elif method in ("sparse_symbolic", "sparse_cpp"):
            begin = time.time()
            if method == "sparse_symbolic":
                nearest_key_indices = batched_symbolic_sparse_nearest_k_keys(
                    queries, keys, k
                ).to(torch.int64)
            elif method == "sparse_cpp":
                nearest_key_indices = sparse_attention.nearestKKeys(
                    queries, keys, k, maxLeafSize
                ).to(torch.int64)
            end = time.time()
            run_key_search_times.append(end - begin)

            begin = time.time()
            out = post_processing(nearest_key_indices, queries, keys, values, args)
            end = time.time()
            run_post_processing_times.append(end - begin)

            run_attention_times.append(
                run_key_search_times[-1] + run_post_processing_times[-1]
            )

        else:
            raise Exception(f"method {method} unknown")

    # Throw away the first ('cold') run
    run_attention_times = run_attention_times[1:]
    if method in ("sparse_symbolic", "sparse_cpp"):
        run_key_search_times = run_key_search_times[1:]
        run_post_processing_times = run_post_processing_times[1:]

    print("- Results for N =", N, "-")
    print(
        "Total attention time:",
        np.mean(run_attention_times),
        "±",
        np.std(run_attention_times),
    )
    if method in ("sparse_symbolic", "sparse_cpp"):
        print(
            "Key search time:",
            np.mean(run_key_search_times),
            "±",
            np.std(run_key_search_times),
        )
        print(
            "Post processing time:",
            np.mean(run_post_processing_times),
            "±",
            np.std(run_post_processing_times),
        )
