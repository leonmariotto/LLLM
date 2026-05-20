"""
Implement normalization functions.
Shared between models.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Replace LayerNorm in LLama2.
    RMSNorm uses only the root mean square, which improves computational efficiency
    (over LayerNorm which use mean and variance).
    See https://arxiv.org/abs/1910.07467.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor with shape ``[..., emb_dim]``.

        Returns:
            Normalized tensor with shape ``[..., emb_dim]``.
        """
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


class LayerNorm(nn.Module):
    """LayerNorm: normalize output data to have a mean of 0 and a variance of 1."""

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape ``[..., emb_dim]``.

        Returns:
            Normalized tensor with shape ``[..., emb_dim]``.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
