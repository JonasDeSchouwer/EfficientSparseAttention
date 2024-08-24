import torch
import sparse_attention
import math
import time
import argparse
import time
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--H", type=int, default=1)
parser.add_argument("--kq_dim", type=int, default=50)
parser.add_argument("--val_dim", type=int, default=50)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--maxLeafSize", type=int, default=10)
parser.add_argument(
    "--method",
    type=str,
    choices=["sparse_symbolic", "sparse_cpp", "full", "faiss"],
)
parser.add_argument("--maxN", type=int, default=1e9)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# imports
from baselines.symbolic_sparse import symbolic_sparse_nearest_k_keys
from baselines.post_processing import batched_post_processing
from baselines.full import batched_full_MHA
# faiss is a conditional import, because it gives issues sometimes
if args.method == "faiss":
    from baselines.faiss import faiss_search


# fix all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.device == "cuda":
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
Ns = list(filter(lambda x: x <= args.maxN, map(int, Ns)))

if args.device == "cuda":
    num_gpus = torch.cuda.device_count()
    gpu_memories = [
        torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)
    ]  # in bytes
    biggest_allocation_memory = int(
        min(gpu_memories) / 2
    )  # in bytes, this determines the batch size. The /2 is to be on the safe side
else:
    biggest_allocation_memory = 1e9


print("method:", method)
print("B:", B)
print("H:", H)
print("kq_dim:", kq_dim)
print("val_dim:", val_dim)
print("k:", k)
print("maxLeafSize:", maxLeafSize)
print("num_runs:", num_runs)
print("Ns:", Ns)
print("device:", args.device)


attention_times = []
attention_stds = []
key_search_times = []
key_search_stds = []
post_processing_times = []
post_processing_stds = []
num_keys_searched = []
num_keys_searched_stds = []
num_nodes_searched = []
num_nodes_searched_stds = []


print(f"--- Profiling method {method} ---")

for N in Ns:
    print("\n- N =", N, "-")

    if method == "full":
        print("Batch size:", math.ceil(biggest_allocation_memory / (4 * N)))
    elif method == "sparse_symbolic":
        print(
            "Batch size for key search:",
            "no batching! LazyTensors are never materialized",
        )
        print(
            "Batch size for post processing:",
            math.ceil(
                biggest_allocation_memory / (4 * B * H * k * max(kq_dim, val_dim) * 3)
            ),
        )

    run_attention_times = []
    run_key_search_times = []
    run_post_processing_times = []
    run_num_keys_searched = []
    run_num_nodes_searched = []

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
            out = batched_full_MHA(queries, keys, values, biggest_allocation_memory)
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_attention_times.append(end - begin)

        elif method in ("sparse_symbolic", "sparse_cpp"):
            begin = time.time()
            if method == "sparse_symbolic":
                nearest_key_indices = symbolic_sparse_nearest_k_keys(
                    queries, keys, k
                ).to(torch.int64)
            elif method == "sparse_cpp":
                nearest_key_indices, num_k_s, num_n_s = sparse_attention.nearestKKeys(
                    queries, keys, k, maxLeafSize
                )
                nearest_key_indices = nearest_key_indices.to(torch.int64)
                run_num_keys_searched.append(num_k_s)
                run_num_nodes_searched.append(num_n_s)
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_key_search_times.append(end - begin)

            begin = time.time()
            out = batched_post_processing(
                nearest_key_indices,
                queries,
                keys,
                values,
                args,
                biggest_allocation_memory,
            )
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_post_processing_times.append(end - begin)

            run_attention_times.append(
                run_key_search_times[-1] + run_post_processing_times[-1]
            )

        elif method == "faiss":
            begin = time.time()
            faiss_search(queries, keys, k, biggest_allocation_memory, device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_attention_times.append(end - begin)

        else:
            raise Exception(f"method {method} unknown")

    # Throw away the first ('cold') run
    run_attention_times = run_attention_times[1:]
    if method in ("sparse_symbolic", "sparse_cpp"):
        run_key_search_times = run_key_search_times[1:]
        run_post_processing_times = run_post_processing_times[1:]
    if method in ("sparse_cpp"):
        run_num_keys_searched = run_num_keys_searched[1:]
        run_num_nodes_searched = run_num_nodes_searched[1:]

    print("- Results for N =", N, "-")
    attention_times.append(np.mean(run_attention_times))
    attention_stds.append(np.std(run_attention_times))
    print(
        "Total attention time:",
        np.mean(run_attention_times),
        "±",
        np.std(run_attention_times),
    )
    if method in ("sparse_symbolic", "sparse_cpp"):
        key_search_times.append(np.mean(run_key_search_times))
        key_search_stds.append(np.std(run_key_search_times))
        post_processing_times.append(np.mean(run_post_processing_times))
        post_processing_stds.append(np.std(run_post_processing_times))
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
    if method in ("sparse_cpp"):
        num_keys_searched.append(np.mean(run_num_keys_searched))
        num_nodes_searched.append(np.mean(run_num_nodes_searched))
        num_keys_searched_stds.append(np.std(run_num_keys_searched))
        num_nodes_searched_stds.append(np.std(run_num_nodes_searched))

    print("\nAttention time means:", attention_times)
    print("Attention time stds:", attention_stds)
    if method in ("sparse_symbolic", "sparse_cpp"):
        print("Key search time means:", key_search_times)
        print("Key search time stds:", key_search_stds)
        print("Post processing time means:", post_processing_times)
        print("Post processing time stds:", post_processing_stds)
    if method in ("sparse_cpp"):
        print("Num keys searched means:", num_keys_searched)
        print("Num keys searched stds:", num_keys_searched_stds)
        print("Num nodes searched means:", num_nodes_searched)
        print("Num nodes searched stds:", num_nodes_searched_stds)
