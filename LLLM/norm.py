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
        The norm computation need to be done 32bits wide to avoid precision loss.
        This is because small errors are amplified in next matrice multiplication, because
        norm "affects the scale of the hidden state before attention and MLP blocks".
        Args:
            x: Input tensor with shape ``[..., emb_dim]``.

        Returns:
            Normalized tensor with shape ``[..., emb_dim]``.
        """
        input_dtype = x.dtype
        x_float = x.float()
        means = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight.float()).to(dtype=input_dtype)


class LayerNorm(nn.Module):
    """LayerNorm: normalize output data to have a mean of 0 and a variance of 1."""

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The norm computation need to be done 32bits wide to avoid precision loss.
        This is because small errors are amplified in next matrice multiplication, because
        norm "affects the scale of the hidden state before attention and MLP blocks".
        Args:
            x: Input tensor with shape ``[..., emb_dim]``.

        Returns:
            Normalized tensor with shape ``[..., emb_dim]``.
        """
        input_dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x_float - mean) / torch.sqrt(var + self.eps)
        return (self.scale.float() * norm_x + self.shift.float()).to(dtype=input_dtype)
