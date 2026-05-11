"""
Transformer implementation.

Transformer use the following components :
- MultiHeadAttention: attention mechanism, implemented in attention.py
- FeedForward: a small nn that do an expansion (* 4), an activation function pass
(GeLu, not ReLu), and a contraction back to the original input dimension.
- LayerNorm: normalization is used to improve model math efficiency, without it
the model will struggle to find weights that minimize its loss function due to problems
like vanishing or exploding gradients. Normalization layer adjust the output of a nn to
have a mean of 0 and a variance of 1 (== "unit variance").
- Shortcuts: another technics used to improve training. It mitigate the problem of
vanishing gradient, when they become prgressivly smaller as they propagate through
layers, making it difficult to train earlier weights (first layers got gradients
too small). So we create shortcut connection that add the N layer input to N+1 layer
input (so N+1 input become N output + N input).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from .attention import MultiHeadAttention

# avoids a runtime circular import, except for type checking.
if TYPE_CHECKING:
    from .gpt import GPTConfig


# TODO use torch GeLu and remove this, or move some in comments.
class GELU(nn.Module):
    """
    Implement the GeLu activation function approximation (computationally cheaper).
    An optimized version is present torch.nn.functional.gelu but keep it here
    for illustration purpose.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """
    FeedForward: expansion -> activation (GeLu) -> contraction.
    """

    def __init__(self, embedded_dimension: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedded_dimension, expansion_factor * embedded_dimension),
            GELU(),
            nn.Linear(expansion_factor * embedded_dimension, embedded_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LayerNorm(nn.Module):
    """
    LayerNorm: normalize output data to have a mean of 0 and a variance of 1.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut: nn.Module = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor, pos: Optional[int] = None) -> torch.Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, pos)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
