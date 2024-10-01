"""
This file contains the post processing that is needed after sparse_symbolic or sparse_cpp is applied.
I.e. based on the result containing the indices of the k nearest keys for each query,
we need to gather the queries, keys and values for these indices and compute the output of sparse attention.
"""

from argparse import Namespace
import torch
import torch.nn.functional as F
import math


def batched_post_processing(
    nearest_key_indices: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    args: Namespace,
    biggest_allocation_memory: int,
):
    """
    Post processing for sparse attention.
    :param nearest_key_indices: [B, H, N, k]
    :param queries: [B, H, N, kq_dim]
    :param keys: [B, H, N, kq_dim]
    :param values: [B, H, N, val_dim]
    :param args: Namespace containing the arguments for the model
    :param biggest_allocation_memory: int, the biggest possible allocation in bytes
    """
    B = keys.shape[0]
    H = keys.shape[1]
    N = keys.shape[2]
    batch_size = math.ceil(
        biggest_allocation_memory
        / (4 * B * H * args.k * max(args.kq_dim, args.val_dim) * 3)
    )  # choose batch size such that the number of bytes in the LazyTensors is below biggest_allocation_memory
    num_batches = math.ceil(N / batch_size)

    out = []
    for i in range(num_batches):
        queries_batch = queries[:, :, i * batch_size : (i + 1) * batch_size]
        nearest_key_indices_batch = nearest_key_indices[
            :, :, i * batch_size : (i + 1) * batch_size
        ]

        out.append(
            post_processing_(
                nearest_key_indices_batch,
                queries_batch,
                keys,
                values,
                args,
            )
        )

    return torch.cat(out, dim=2)


def post_processing_(
    nearest_key_indices_batch: torch.Tensor,
    queries_batch: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    args: Namespace,
):
    """
    Post processing for sparse attention.
    :param nearest_key_indices_batch: [B, H, N_batch, k]
    :param queries_batch: [B, H, N_batch, kq_dim]
    :param keys: [B, H, N, kq_dim]
    :param values: [B, H, N, val_dim]
    :param args: Namespace containing the arguments for the model
    """

    # [**, H, N_batch, k, kq_dim]
    nearest_keys = torch.gather(
        input=keys.unsqueeze(-2).expand(
            *keys.shape[:-1], args.k, args.kq_dim
        ),  # [**, H, N, k, kq_dim]
        dim=-3,
        index=nearest_key_indices_batch.unsqueeze(-1).expand(
            *nearest_key_indices_batch.shape, args.kq_dim
        ),  # [**, H, N_batch, k, kq_dim]
        # sparse_grad=True,
    )
    # [**, H, N_batch, k, val_dim]
    nearest_values = torch.gather(
        input=values.unsqueeze(-2).expand(
            *keys.shape[:-1], args.k, args.val_dim
        ),  # [**, H, N, k, val_dim]
        dim=-3,
        index=nearest_key_indices_batch.unsqueeze(-1).expand(
            *nearest_key_indices_batch.shape, args.val_dim
        ),  # [**, H, N_batch, k, val_dim]
        # sparse_grad=True,
    )
    # [**, H, N_batch, k, kq_dim]
    queries_extended = queries_batch.unsqueeze(-2).expand(
        *queries_batch.shape[:-1], args.k, args.kq_dim
    )
    # [**, H, N_batch, k]
    largest_attention_weights = (queries_extended * nearest_keys).sum(-1) / math.sqrt(
        args.kq_dim
    )
    largest_attention_weights = F.softmax(largest_attention_weights, dim=-1)
    # [**, H, N_batch, val_dim]
    out = (largest_attention_weights.unsqueeze(-1) * nearest_values).sum(
        dim=-2
    )  # sum over k nearest keys for each query

    return out
