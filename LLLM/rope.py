"""
Primitive for RoPE (Rotary Positional Encoding)
See :
- https://huggingface.co/blog/designing-positional-encoding
- https://arxiv.org/pdf/2104.09864

Support adjacent-pair (interleaved) layout and split-half layout.
"""

from typing import TypedDict

import torch


class RopeFrequencyConfig(TypedDict):
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_context_len: int


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    freq_config: RopeFrequencyConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute RoPE rotations matrix.
    seq_len is the maximum dimension for interpretable context. A sliding context buffer
    using a larger position space is possible thanks to RoPE. In this case seq_len is
    equal to this larger position space length.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"

    # Number of 2D pairs
    half_dim = head_dim // 2

    # Frequencies θ_i
    i = torch.arange(half_dim, device=device)
    theta = 1.0 / (base ** (2 * i / head_dim))

    if freq_config is not None:
        theta = _apply_rope_frequency_config(theta, freq_config)

    # Positions p
    positions = torch.arange(seq_len, device=device)

    # angles[p, i] = p * θ_i
    angles = positions[:, None] * theta[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def _apply_rope_frequency_config(
    theta: torch.Tensor,
    freq_config: RopeFrequencyConfig,
) -> torch.Tensor:
    """
    Apply Llama 3 style RoPE frequency scaling to inverse frequencies.
    Introduced by Llama3.1 and 3.2.
    """
    low_freq_wavelen = (
        freq_config["original_context_len"] / freq_config["low_freq_factor"]
    )
    high_freq_wavelen = (
        freq_config["original_context_len"] / freq_config["high_freq_factor"]
    )

    wavelen = 2 * torch.pi / theta
    scaled_theta = torch.where(
        wavelen > low_freq_wavelen,
        theta / freq_config["factor"],
        theta,
    )

    smooth_factor = (
        freq_config["original_context_len"] / wavelen - freq_config["low_freq_factor"]
    ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])
    smoothed_theta = (1 - smooth_factor) * (
        theta / freq_config["factor"]
    ) + smooth_factor * theta

    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_theta, scaled_theta)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, use_interleaved: bool = True
) -> torch.Tensor:
    """
    Apply RoPE on x.

    Support interleaved and split-half layout.

    x shape: [batch, heads, seq_len, head_dim]
    cos/sin shape: [seq_len, head_dim // 2]

    return rotated x.
    """
    if use_interleaved:
        return apply_rope_interleaved(x, cos, sin)
    else:
        return apply_rope_split_half(x, cos, sin)


def apply_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE on x usign interleaved pairing.

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


def apply_rope_split_half(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE on x using split-half pairing.

    x shape: [batch, heads, seq_len, head_dim]
    cos/sin shape: [seq_len, head_dim // 2]

    return rotated x.
    """
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Broadcast cos/sin over batch and heads.
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return torch.cat((x1_rot, x2_rot), dim=-1)
