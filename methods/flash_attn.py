import torch
import flash_attn


def flash_attn_mech(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the output of a full multi-head attention layer using Flashattention.

    Args:
        queries: [B, H, N, kq_dim]
        keys: [B, H, N, kq_dim]
        values: [B, H, N, val_dim]

    Returns:
        [B, H, N, val_dim]
    """
    
    # switch H and N dimensions, as this is required by flash_attn
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    window_size = (-1, -1) # infinite window size
    
    return flash_attn.flash_attn_func(queries, keys, values, dropout_p=0.0, causal=False, window_size=window_size, deterministic=True)
