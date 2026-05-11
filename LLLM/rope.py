"""
Primitive for RoPE (Rotary Positional Encoding)
See :
- https://huggingface.co/blog/designing-positional-encoding
- https://arxiv.org/pdf/2104.09864
"""

import torch


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: int = 10000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute RoPE rotations matrix.
    seq_len is the maximum dimension for interpretable context. A sliding context buffer
    using a larger position space is possible thanks to RoPE. In this case seq_len is equal
    to this larger position space length.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    # Number of 2D pairs
    half_dim = head_dim // 2

    # Frequencies θ_i
    i = torch.arange(half_dim, device=device)
    theta = 1.0 / (base ** (2 * i / head_dim))

    # Positions p
    positions = torch.arange(seq_len, device=device)

    # angles[p, i] = p * θ_i
    angles = positions[:, None] * theta[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE on x.

    x shape: [batch, heads, seq_len, head_dim]
    cos/sin shape: [seq_len, head_dim // 2]

    return rotated x.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Broadcast cos/sin over batch and heads
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Interleave the result back into the original shape
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x_even_rot
    x_rot[..., 1::2] = x_odd_rot

    return x_rot
