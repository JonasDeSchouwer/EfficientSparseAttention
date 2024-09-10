from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import math
import time
import argparse
import time
import numpy as np
import datetime
import os
import os.path as osp
from tqdm import tqdm


parser = argparse.ArgumentParser()

# method arguments
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--maxLeafSize", type=int, default=10)
parser.add_argument(
    "--method",
    type=str,
    choices=["sparse_symbolic", "sparse_cpp", "full", "faiss", "full_builtin"],
)

# experiment arguments
parser.add_argument("--maxN", type=int, default=1e9)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--name", type=str, default="", help="Name of the experiment")
parser.add_argument("--folder", type=str, default="random")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--detailed_profiling", action="store_true")

# where to get the q,k,v from
parser.add_argument("--qkv", type=str, default="random", choices=["cifar", "malnet"])
parser.add_argument("--qkv_seed", type=int, default=0)
parser.add_argument("--trained_model_type", type=str, default="GPS+SparseAttention", choices=["GPS+SparseAttention", "GPS+Transformer"])
parser.add_argument("--experiment_type", type=str, default="approximation", choices=["approximation", "softmax_weights"])

args = parser.parse_args()

# post processing arguments
if args.qkv == 'cifar':
    args.qkv = 'Cifar10'
if args.qkv == 'malnet':
    args.qkv = 'MalNet-Tiny'


# imports
from utils import rowwise_recall, all_equal, get_rounded_geometric_progression
from baselines.symbolic_sparse import symbolic_sparse_nearest_k_keys
from baselines.post_processing import batched_post_processing
from baselines.full import batched_full_MHA
# faiss and sparse_attention are conditional imports, because they give issues sometimes
if args.method == "sparse_cpp":
    import sparse_attention
if args.method == "faiss":
    from baselines.faiss import faiss_search


# fix all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.device == "cuda":
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

timestamp = datetime.datetime.now().strftime("%m.%d-%H:%M")

print("method:", args.method)
print("k:", args.k)
print("maxLeafSize:", args.maxLeafSize)
print("num_runs:", args.num_runs)
print("Ns:", Ns)
print("device:", args.device)
print("name:", args.name)
print("folder:", args.folder)
print("timestamp:", timestamp)
print("detailed_profiling:", args.detailed_profiling)
print("qkv:", args.qkv)
print("qkv_seed:", args.qkv_seed)
print("trained_model_type:", args.trained_model_type)
print("experiment_type:", args.experiment_type)


@torch.no_grad()
def get_full_cum_weights(queries, keys):
    """
    q,k: (1,H,N,kq_dim)

    returns: (N,): entry i is the (average) cumulative softmax weight of all keys up to the i-th key
    """
    kq_dim = queries.shape[-1]

    # [B, H, N, N]
    full_attention_weights = torch.einsum(
        "bhqd,bhkd->bhqk", queries, keys
    ) / math.sqrt(kq_dim)

    # [B, H, N, N]
    full_attention_weights = F.softmax(full_attention_weights, dim=-1)

    # sort each row of full_attention_weights in descending order
    full_attention_weights, _ = full_attention_weights.sort(dim=-1, descending=True)

    full_attention_weights = full_attention_weights.cumsum(dim=-1)

    avg_attention_weight = full_attention_weights.mean(dim=(0,1,2))

    return avg_attention_weight


@torch.no_grad()
def get_full_approx_qualities(queries, keys, values, ks):
    """
    queries, keys: (1,H,N,kq_dim)
    values: (1,H,N,val_dim)
    ks: (#ks,)

    returns: (#ks,): entry i is the (average) L2 distance when k-MIP attention is performed with k=i
    """
    N = keys.shape[-2]
    result = torch.zeros(len(ks), dtype=torch.float32)

    full_out = batched_full_MHA(queries, keys, values, biggest_allocation_memory)

    full_attention_weights = torch.einsum("...ik,...jk->...ij", queries, keys)

    # [**, num_heads, N, k]
    _, nearest_key_indices = full_attention_weights.sort(dim=-1)
    
    # [**, H, N, N, kq_dim]
    nearest_keys = torch.gather(
        input=keys.unsqueeze(-2).expand(
            *keys.shape[:-1], N, args.kq_dim
        ),  # [**, H, N, N, kq_dim]
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(
            *nearest_key_indices.shape, args.kq_dim
        ),  # [**, H, N, N, kq_dim]
        # sparse_grad=True,
    )

    # [**, H, N, N, val_dim]
    nearest_values = torch.gather(
        input=values.unsqueeze(-2).expand(
            *keys.shape[:-1], N, args.val_dim
        ),  # [**, H, N, N, val_dim]
        dim=-3,
        index=nearest_key_indices.unsqueeze(-1).expand(
            *nearest_key_indices.shape, args.val_dim
        ),  # [**, H, N, N, val_dim]
        # sparse_grad=True,
    )

    # [**, H, N, N, kq_dim]
    queries_extended = queries.unsqueeze(-2).expand(
        *queries.shape[:-1], N, args.kq_dim
    )
    # [**, H, N, N]
    # attention weights before softmax
    weights = (queries_extended * nearest_keys).sum(-1) / math.sqrt(
        args.kq_dim
    )
    # [**, H, N, N]
    exp_weights = torch.exp(weights)
    # [**, H, N, N]
    cum_sum_exp_weights = exp_weights.cumsum(dim=-1)

    for k_id, k in enumerate(ks):
        if k > N:
            continue
        # [**, H, N, k]
        approx_attention_weights = exp_weights[..., :k] / cum_sum_exp_weights[..., k].unsqueeze(-1)
        # [**, H, N, val_dim]
        approx_out = torch.einsum(
            "...ik,...ikj->...ij",
            approx_attention_weights,
            nearest_values[..., :k, :]
        )

        result[k_id] = (full_out - approx_out).norm(dim=-1).mean().item()

    return result

@torch.no_grad()
def compare_to_full(queries, keys, values):
    """
    q,k: (1,H,N,kq_dim)
    v: (1,H,N,val_dim)
    """
    assert queries.ndim == keys.ndim == values.ndim == 4
    assert queries.shape[0] == keys.shape[0] == values.shape[0] == 1
    assert queries.shape[1] == keys.shape[1] == values.shape[1] == H
    assert queries.shape[2] == keys.shape[2] == values.shape[2]
    assert queries.shape[3] == keys.shape[3] == kq_dim
    assert values.shape[3] == val_dim

    # a dict that will contain all outputs we need, including times and approximation quality
    output = {}

    full_out = batched_full_MHA(queries, keys, values, biggest_allocation_memory)

    if args.method in ("sparse_symbolic", "sparse_cpp"):
        begin = time.time()
        if args.method == "sparse_symbolic":
            nearest_key_indices = symbolic_sparse_nearest_k_keys(
                queries, keys, args.k
            ).to(torch.int64)
        elif args.method == "sparse_cpp":
            nearest_key_indices, num_k_s, num_n_s = sparse_attention.nearestKKeys(
                queries, keys, args.k, args.maxLeafSize
            )
            nearest_key_indices = nearest_key_indices.to(torch.int64)
            output['num_keys_searched'] = num_k_s
            output['num_nodes_searched'] = num_n_s
        if args.device == "cuda":
            torch.cuda.synchronize()  # for accurate time measurement
        end = time.time()
        output['key_search_time'] = end - begin

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
        output['post_processing_time'] = end - begin

        output['attention_time'] = output['key_search_time'] + output['post_processing_time']


    elif args.method == "faiss":
        begin = time.time()
        topk_faiss = faiss_search(
            queries, keys, args.k,
            biggest_allocation_memory,
            device=args.device,
            detailed_profiling=args.detailed_profiling)
        if args.device == "cuda":
            torch.cuda.synchronize()  # for accurate time measurement
        end = time.time()
        output['attention_time'] = end - begin

        # if track_approx:
        #     topk_sym = symbolic_sparse_nearest_k_keys(
        #         queries, keys, k
        #     ).to(torch.int64)
        #     run_approximation_qualities.append(
        #         rowwise_recall(topk_faiss, topk_sym).mean().item()
        #     )

    print(full_out - out)

    # get the average (over tokens & token feature elements) of the L2 norm of the difference between the two outputs
    output['l2_diff'] = (full_out - out).norm(dim=-1).mean().item()

    return output


if args.qkv in ('Cifar10', 'MalNet-Tiny'):
    Ks = (1,2,3,4,5,10,15,20,30,50)

    # load the tokens and parameters
    token_dir = f"../GraphGPS/saved_tokens/{args.qkv}/{args.trained_model_type}/{args.qkv_seed}"
    layers = os.listdir(token_dir)
    print("layers:", layers)
    num_graphs_per_layer = [len(os.listdir(osp.join(token_dir, layer))) for layer in layers]
    assert all_equal(num_graphs_per_layer), "Number of graphs per layer should be the same"
    num_graphs = num_graphs_per_layer[0]

    sample_token = torch.load(osp.join(token_dir, layers[0], '0.pt'))
    queries = sample_token['q']
    keys = sample_token['k']
    values = sample_token['v']

    _, H, _, kq_dim = queries.shape
    _, _, _, val_dim = values.shape
    print("")
    print("detected H:", H)
    print("detected kq_dim:", kq_dim)
    print("detected val_dim:", val_dim)
    args.H = H
    args.kq_dim = kq_dim
    args.val_dim = val_dim

    
    if args.experiment_type == 'approximation':
        if args.qkv == 'Cifar10':
            max_N = 200
            min_N = 80
        elif args.qkv == 'MalNet-Tiny':
            max_N = 5000
            min_N = 1000
        else:
            raise ValueError(f"Unknown qkv type {args.qkv}")
        
        ks = get_rounded_geometric_progression(alpha=1.2, max=max_N) + [max_N]
        print("ks:", ks)
        # will accumulate the l2 distances
        l2_distances = torch.zeros(len(layers), num_graphs, len(ks), dtype=torch.float32)
        l2_distances_normal = torch.zeros(len(layers), num_graphs, len(ks), dtype=torch.float32)

        for layer_id, layer in enumerate(layers):
            num_graphs_in_N_range = 0
            for token_file in tqdm(os.listdir(osp.join(token_dir, layer))[:num_graphs]):
                token_dict = torch.load(osp.join(token_dir, layer, token_file))
                queries, keys, values = token_dict['q'], token_dict['k'], token_dict['v']

                N = queries.shape[-2]
                if N < min_N or N > max_N:
                    continue

                l2_distances[layer_id, num_graphs_in_N_range] = get_full_approx_qualities(queries, keys, values, ks)

                # if we assume that queries and keys are normally distributed, and the graph also has N nodes
                queries = torch.randn_like(queries)
                keys = torch.randn_like(keys)
                values = torch.randn_like(values)

                l2_distances_normal[layer_id, num_graphs_in_N_range] = get_full_approx_qualities(queries, keys, values, ks)

                num_graphs_in_N_range += 1

            print(num_graphs_in_N_range)
        

        l2_distances = l2_distances[:, :num_graphs_in_N_range]
        l2_distances_normal = l2_distances_normal[:, :num_graphs_in_N_range]

        # averaging over all graphs
        # [num_layers, max_N]
        avg_l2_distances = l2_distances.mean(dim=1)
        avg_l2_distances_normal = l2_distances_normal.mean(dim=1)
        std_l2_distances = l2_distances.std(dim=1)
        std_l2_distances_normal = l2_distances_normal.std(dim=1)

        # save attention weights
        torch.save(l2_distances, osp.join('approx-output', f'full_l2_distances_{timestamp}.pt'))
        torch.save(l2_distances_normal, osp.join('approx-output', f'full_l2_distances_normal_{timestamp}.pt'))
        torch.save(avg_l2_distances, osp.join('approx-output', f'avg_l2_distances_{timestamp}.pt'))
        torch.save(avg_l2_distances_normal, osp.join('approx-output', f'avg_l2_distances_normal_{timestamp}.pt'))
        torch.save(std_l2_distances, osp.join('approx-output', f'std_l2_distances_{timestamp}.pt'))
        torch.save(std_l2_distances_normal, osp.join('approx-output', f'std_l2_distances_normal_{timestamp}.pt'))
        torch.save(ks, osp.join('approx-output', f'ks_{timestamp}.pt'))
    
    elif args.experiment_type == 'softmax_weights':
        if args.qkv == 'Cifar10':
            max_N = 200
            min_N = 80
        elif args.qkv == 'MalNet-Tiny':
            max_N = 5000
            min_N = 1000
        else:
            raise ValueError(f"Unknown qkv type {args.qkv}")
        
        # will accumulate the actual attention weights
        attention_weights = torch.ones(len(layers), num_graphs, max_N, dtype=torch.float32)
        # will accumulate the attention weights if we follow the assumption that queries and keys are normally distributed, and graphs have the same size
        attention_weights_normal = torch.ones(len(layers), num_graphs, max_N, dtype=torch.float32)

        for layer_id, layer in enumerate(layers):
            num_graphs_in_N_range = 0
            for token_file in tqdm(os.listdir(osp.join(token_dir, layer))[:num_graphs]):
                token_dict = torch.load(osp.join(token_dir, layer, token_file))
                queries = token_dict['q']
                keys = token_dict['k']

                N = queries.shape[-2]
                if N < min_N or N > max_N:
                    continue

                attention_weights_single = get_full_cum_weights(queries, keys)
                attention_weights[layer_id, num_graphs_in_N_range, :N] = attention_weights_single

                # if we assume that queries and keys are normally distributed, and the graph also has N nodes
                queries = torch.randn_like(queries)
                keys = torch.randn_like(keys)

                attention_weights_normal_single = get_full_cum_weights(queries, keys)
                attention_weights_normal[layer_id, num_graphs_in_N_range, :N] = attention_weights_normal_single

                num_graphs_in_N_range += 1

            print(num_graphs_in_N_range)
        
        attention_weights = attention_weights[:, :num_graphs_in_N_range]
        attention_weights_normal = attention_weights_normal[:, :num_graphs_in_N_range]

        # averaging over all graphs
        # [num_layers, max_N]
        avg_attention_weights = attention_weights.mean(dim=1)
        avg_attention_weights_normal = attention_weights_normal.mean(dim=1)
        std_attention_weights = attention_weights.std(dim=1)
        std_attention_weights_normal = attention_weights_normal.std(dim=1)

        # save attention weights
        torch.save(attention_weights, osp.join('approx-output', f'full_attention_weights_{timestamp}.pt'))
        torch.save(attention_weights_normal, osp.join('approx-output', f'full_attention_weights_normal_{timestamp}.pt'))
        torch.save(avg_attention_weights, osp.join('approx-output', f'avg_attention_weights_{timestamp}.pt'))
        torch.save(avg_attention_weights_normal, osp.join('approx-output', f'avg_attention_weights_normal_{timestamp}.pt'))
        torch.save(std_attention_weights, osp.join('approx-output', f'std_attention_weights_{timestamp}.pt'))
        torch.save(std_attention_weights_normal, osp.join('approx-output', f'std_attention_weights_normal_{timestamp}.pt'))
        
        fig = plt.figure()
        for layer_id, layer in enumerate(layers):
            plt.plot(avg_attention_weights[layer_id], label=layer)
            plt.plot(avg_attention_weights_normal[layer_id], label=f"{layer} normal")
            plt.xscale('log')
        plt.legend()
        plt.show()

    else:
        raise ValueError(f"Unknown experiment type {args.experiment_type}")
