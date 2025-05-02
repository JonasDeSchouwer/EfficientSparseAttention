import gc
import torch
import math
import time
import argparse
import time
import numpy as np
import datetime
import os


# Set environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--H", type=int, default=1)
parser.add_argument("--kq_dim", type=int, default=10)
parser.add_argument("--val_dim", type=int, default=50)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--maxLeafSize", type=int, default=10)
parser.add_argument(
    "--method",
    type=str,
    choices=["sym", "naive", "sparse_cpp", "full", "faiss", "full_builtin", "flash_attn"],
)
parser.add_argument("--minN", type=int, default=1e2)
parser.add_argument("--maxN", type=int, default=1e6)
parser.add_argument("--specific-ns", type=int, nargs='+', default=None, help="List of specific N values to test")
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--name", type=str, default="", help="Name of the experiment")
parser.add_argument("--folder", type=str, default="random")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-track_approx", action="store_false", dest="track_approx")
parser.add_argument("--detailed_profiling", action="store_true")
parser.add_argument("--nlist", type=int, default=None)
parser.add_argument("--nprobe", type=int, default=None)
parser.add_argument("--do_backward", action="store_true")
parser.add_argument("--require_grad", action="store_true")
parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Data type for tensors")
args = parser.parse_args()


# imports
from utils import rowwise_recall
from methods.naive_sparse import batched_naive_sparse_nearest_k_keys
from methods.symbolic_sparse import symbolic_sparse_nearest_k_keys
from methods.post_processing import batched_post_processing
from methods.full import batched_full_MHA# faiss and sparse_attention are conditional imports, because they give issues sometimes
if args.method == "sparse_cpp":
    import sparse_attention
if args.method == "faiss":
    from methods.faiss import faiss_search
if args.method == "flash_attn":
    from methods.flash_attn import flash_attn_mech


# fix all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.device == "cuda":
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


B = args.B
H = args.H
kq_dim = args.kq_dim
val_dim = args.val_dim
k = args.k
maxLeafSize = args.maxLeafSize
num_runs = args.num_runs
method = args.method
track_approx = args.track_approx
detailed_profiling = args.detailed_profiling

sqrt10 = math.sqrt(10)
if args.specific_ns is not None:
    Ns = args.specific_ns
else:
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
        sqrt10 * 1e6,
        1e7,
        sqrt10 * 1e7,
        1e8,
        sqrt10 * 1e8,
        1e9,
    ]
    Ns = list(filter(lambda x: args.minN <= x <= args.maxN, map(int, Ns)))

if args.device == "cuda":
    num_gpus = torch.cuda.device_count()
    gpu_memories = [
        torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)
    ]  # in bytes
    biggest_allocation_memory = int(
        min([torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in range(num_gpus)]) / 2
    )  # in bytes, this determines the batch size. The /2 is to be on the safe side
else:
    biggest_allocation_memory = 1e9

timestamp = datetime.datetime.now().strftime("%m.%d-%H:%M")

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
print("name:", args.name)
print("folder:", args.folder)
print("timestamp:", timestamp)
print("track_approx:", track_approx)
print("detailed_profiling:", detailed_profiling)
print("nlist:", args.nlist)
print("nprobe:", args.nprobe)
print("do_backward:", args.do_backward)
print("dtype:", args.dtype)


attention_times = []
attention_stds = []
backward_times = []
backward_stds = []
total_times = []
total_stds = []
bf_attention_times = []
bf_attention_stds = []
key_search_times = []
key_search_stds = []
post_processing_times = []
post_processing_stds = []
num_keys_searched = []
num_keys_searched_stds = []
num_nodes_searched = []
num_nodes_searched_stds = []
approximation_qualities = []
gpu_memory_usage = []  # Memory before backward
gpu_memory_usage_stds = []
peak_memory_usage = []  # New list for peak memory
peak_memory_usage_stds = []  # New list for peak memory standard deviation


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    if dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_peak_memory():
    """Get peak memory usage in MB."""
    if args.device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0  # For CPU, return 0 as we don't track CPU memory


print(f"--- Profiling method {method} ---")

for N in Ns:
    print("\n- N =", N, "-")

    if method == "full":
        print("Batch size:", math.ceil(biggest_allocation_memory / (4 * N)))
    elif method == "sym":
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
    run_backward_times = []
    run_total_times = []
    run_num_keys_searched = []
    run_num_nodes_searched = []
    run_approximation_qualities = []
    run_bf_attention_times = []
    run_gpu_memory_usage = []
    run_peak_memory = []  # New list for peak memory per run

    # --- main processing and profiling ---
    for run in range(num_runs + 1):
        # generate queries and keys and save them to SAVE_DIR
        queries = torch.randn((B, H, N, kq_dim), requires_grad=args.do_backward or args.require_grad, dtype=get_torch_dtype(args.dtype))
        keys = torch.randn((B, H, N, kq_dim), requires_grad=args.do_backward or args.require_grad, dtype=get_torch_dtype(args.dtype))
        values = torch.randn((B, H, N, val_dim), requires_grad=args.do_backward or args.require_grad, dtype=get_torch_dtype(args.dtype))

        if args.device == "cuda":
            torch.cuda.synchronize()
            queries = queries.cuda()
            keys = keys.cuda()
            values = values.cuda()
        else:
            assert (
                args.device == "cpu"
            ), f"device {args.device} not supported: only 'cpu' and 'cuda' are supported"

        if method == "full":
            begin = time.time()
            out = batched_full_MHA(queries, keys, values, biggest_allocation_memory, detailed_profiling)
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_attention_times.append(end - begin)
            print("attention time:", end - begin)

        elif method == "full_builtin":
            # is an end-to-end method, so we need to put the input in a different format
            # IMPORTANT NOTE: only fairly comparable to the other methods when embed_dim = kq_dim = val_dim
            dim = kq_dim
            x = torch.randn(B,N, dim, device=args.device)

            # when embed_dim = kq_dim = val_dim, fastpath inference is used. However, this only occurs when autograd is disabled, which is usually not the case.
            multihead_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=H, batch_first=True, kdim=dim, vdim=dim).to(args.device)
            begin = time.time()
            out = multihead_attn(x,x,x)
            if args.device == "cuda":
                torch.cuda.synchronize() # for accurate time measurement
            end = time.time()
            run_attention_times.append(end - begin)
            print("attention time:", end - begin)

        elif method in ("sym", "sparse_cpp", "naive"):
            begin = time.time()
            if method == "sym":
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
            elif method == "naive":
                nearest_key_indices = batched_naive_sparse_nearest_k_keys(
                    queries, keys, k, biggest_allocation_memory
                ).to(torch.int64)
            if args.device == "cuda":
                print("waiting for synchronize ...", end=" ")
                torch.cuda.synchronize()  # for accurate time measurement
                print("done")
            end = time.time()
            run_key_search_times.append(end - begin)

            gc.collect()
            
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
            print("attention time:", run_attention_times[-1])

        elif method == "faiss":
            begin = time.time()
            topk_faiss = faiss_search(
                queries, keys, k,
                biggest_allocation_memory,
                device=args.device,
                detailed_profiling=detailed_profiling,
                nlist=args.nlist,
                nprobe=args.nprobe,)
            if args.device == "cuda":
                torch.cuda.synchronize()  # for accurate time measurement
            end = time.time()
            run_attention_times.append(end - begin)
            print("attention time:", end - begin)

            if track_approx:
                begin = time.time()
                topk_sym = symbolic_sparse_nearest_k_keys(
                    queries, keys, k
                ).to(torch.int64)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                print("brute force search time:", end - begin)
                run_bf_attention_times.append(end - begin)
                run_approximation_qualities.append(
                    rowwise_recall(topk_faiss, topk_sym).mean().item()
                )
                print(
                    "approximation quality:",
                    run_approximation_qualities[-1],
                )

        elif method == "flash_attn":
            begin = time.time()
            out = flash_attn_mech(queries, keys, values)
            if args.device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            run_attention_times.append(end - begin)
            print("attention time:", end - begin)
        else:
            raise Exception(f"method {method} unknown")
        
        if args.do_backward:
            # Measure GPU memory usage before backward pass
            if args.device == "cuda":
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
                run_gpu_memory_usage.append(current_memory)
                print(f"GPU memory usage before backward: {current_memory:.2f} MB")
            
            begin = time.time()
            out.sum().backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            run_backward_times.append(end - begin)
            print("backward time:", end - begin)

            run_total_times.append(run_attention_times[-1] + run_backward_times[-1])
            print("total time:", run_total_times[-1])

        if args.device == "cuda":
            torch.cuda.synchronize()
            peak_mem = get_peak_memory()
            run_peak_memory.append(peak_mem)
            print(f"Peak memory usage: {peak_mem:.2f} MB")
        
        del queries, keys, values, out

    # Throw away the first ('cold') run
    run_attention_times = run_attention_times[1:]
    if args.do_backward:
        run_backward_times = run_backward_times[1:]
        run_total_times = run_total_times[1:]
        run_gpu_memory_usage = run_gpu_memory_usage[1:]
    if method in ("sym", "sparse_cpp", "naive"):
        run_key_search_times = run_key_search_times[1:]
        run_post_processing_times = run_post_processing_times[1:]
    if method in ("sparse_cpp"):
        run_num_keys_searched = run_num_keys_searched[1:]
        run_num_nodes_searched = run_num_nodes_searched[1:]

    print("- Results for N =", N, "-")
    attention_times.append(np.mean(run_attention_times))
    attention_stds.append(np.std(run_attention_times))
    print(
        "Forward time:",
        np.mean(run_attention_times),
        "±",
        np.std(run_attention_times),
    )
    if args.do_backward:
        backward_times.append(np.mean(run_backward_times))
        backward_stds.append(np.std(run_backward_times))
        total_times.append(np.mean(run_total_times))
        total_stds.append(np.std(run_total_times))
        gpu_memory_usage.append(np.mean(run_gpu_memory_usage))
        gpu_memory_usage_stds.append(np.std(run_gpu_memory_usage))
        print(
            "Backward time:",
            np.mean(run_backward_times),
            "±",
            np.std(run_backward_times),
        )
        print(
            "GPU memory usage before backward:",
            np.mean(run_gpu_memory_usage),
            "±",
            np.std(run_gpu_memory_usage),
            "MB",
        )
        print(
            "Total time:",
            np.mean(run_total_times),
            "±",
            np.std(run_total_times),
        )
    if args.device == "cuda":
        run_peak_memory = run_peak_memory[1:]  # Remove first run
        peak_memory_usage.append(np.mean(run_peak_memory))
        peak_memory_usage_stds.append(np.std(run_peak_memory))
        print(
            "Peak memory usage:",
            np.mean(run_peak_memory),
            "±",
            np.std(run_peak_memory),
            "MB",
        )

    if method in ("sym", "sparse_cpp", "naive"):
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
    if method == "faiss" and track_approx:
        approximation_qualities.append(np.mean(run_approximation_qualities))
        bf_attention_times.append(np.mean(run_bf_attention_times))
        bf_attention_stds.append(np.std(run_bf_attention_times))

    print("\nforward_means:", attention_times)
    print("forward_stds:", attention_stds)
    if args.do_backward:
        print("backward_means:", backward_times)
        print("backward_stds:", backward_stds)
        print("total_means:", total_times)
        print("total_stds:", total_stds)
        print("memory_means:", gpu_memory_usage)
        print("memory_stds:", gpu_memory_usage_stds)
    if args.device == "cuda":
        print("peak_means:", peak_memory_usage)
        print("peak_stds:", peak_memory_usage_stds)
    if method in ("sym", "sparse_cpp"):
        print("Key search time means:", key_search_times)
        print("Key search time stds:", key_search_stds)
        print("Post processing time means:", post_processing_times)
        print("Post processing time stds:", post_processing_stds)
    if method in ("sparse_cpp"):
        print("Num keys searched means:", num_keys_searched)
        print("Num keys searched stds:", num_keys_searched_stds)
        print("Num nodes searched means:", num_nodes_searched)
        print("Num nodes searched stds:", num_nodes_searched_stds)
    if method == "faiss" and track_approx:
        print("Brute force attention times:", bf_attention_times)
        print("Brute force attention stds:", bf_attention_stds)
        print("Approximation qualities:", approximation_qualities)

    filename = f"profiling-output/{args.folder}/{method}-{args.name}-{timestamp}.out"
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as file:
        file.write("Forward time means: " + str(attention_times) + "\n")
        file.write("Forward time stds: " + str(attention_stds) + "\n")
        if args.do_backward:
            file.write("Backward time means: " + str(backward_times) + "\n")
            file.write("Backward time stds: " + str(backward_stds) + "\n")
            file.write("Total time means: " + str(total_times) + "\n")
            file.write("Total time stds: " + str(total_stds) + "\n")
            file.write("GPU memory usage means: " + str(gpu_memory_usage) + "\n")
            file.write("GPU memory usage stds: " + str(gpu_memory_usage_stds) + "\n\n")
        if args.device == "cuda":
            file.write("Peak memory usage means: " + str(peak_memory_usage) + "\n")
            file.write("Peak memory usage stds: " + str(peak_memory_usage_stds) + "\n\n")
        if method in ("sym", "sparse_cpp", "naive"):
            file.write("Key search time means: " + str(key_search_times) + "\n")
            file.write("Key search time stds: " + str(key_search_stds) + "\n")
            file.write("Post processing time means: " + str(post_processing_times) + "\n")
            file.write("Post processing time stds: " + str(post_processing_stds) + "\n")
        if method in ("sparse_cpp"):
            file.write("Num keys searched means: " + str(num_keys_searched) + "\n")
            file.write("Num keys searched stds: " + str(num_keys_searched_stds) + "\n")
            file.write("Num nodes searched means: " + str(num_nodes_searched) + "\n")
            file.write("Num nodes searched stds: " + str(num_nodes_searched_stds) + "\n")
        if method == "faiss" and track_approx:
            file.write("Brute force attention times: " + str(bf_attention_times) + "\n")
            file.write("Brute force attention stds: " + str(bf_attention_stds) + "\n")
            file.write("Approximation qualities: " + str(approximation_qualities) + "\n")

        file.write("\n" + "-"*20 + "\n\n")

        file.write(f"method: {method}\n")
        file.write(f"B: {B}\n")
        file.write(f"H: {H}\n")
        file.write(f"kq_dim: {kq_dim}\n")
        file.write(f"val_dim: {val_dim}\n")
        file.write(f"k: {k}\n")
        file.write(f"maxLeafSize: {maxLeafSize}\n")
        file.write(f"num_runs: {num_runs}\n")
        file.write(f"Ns: {Ns}\n")
        file.write(f"device: {args.device}\n")
        file.write(f"name: {args.name}\n")
        file.write(f"folder: {args.folder}\n")
        file.write(f"timestamp: {timestamp}\n")
        file.write(f"track_approx: {track_approx}\n")
        file.write(f"detailed_profiling: {detailed_profiling}\n")
        file.write(f"nlist: {args.nlist}\n")
        file.write(f"nprobe: {args.nprobe}\n")
        file.write(f"do_backward: {args.do_backward}\n")
        file.write(f"dtype: {args.dtype}\n")