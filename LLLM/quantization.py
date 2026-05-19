"""Quantized weight containers and modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType
from torch import nn
from torch.nn import functional as F


WeightMode = Literal["dense", "quantized"]
QuantizedTransform = Literal["llama_attention_unpermute"]


@dataclass(frozen=True)
class QuantizedWeight:
    """Packed GGUF weight plus enough metadata to materialize a dense tensor."""

    name: str
    tensor_type: GGMLQuantizationType
    data: np.ndarray[Any, Any]
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float16
    transform: QuantizedTransform | None = None
    n_heads: int | None = None
    head_dim: int | None = None

    def dequantize(self, *, device: torch.device | None = None) -> torch.Tensor:
        if self.tensor_type == GGMLQuantizationType.BF16:
            array = self.data.view(np.uint16).astype(np.uint32) << 16
            array = array.view(np.float32)
        else:
            array = gguf.dequantize(self.data, self.tensor_type)

        dense_array = cast(np.ndarray[Any, Any], np.array(array, copy=True))
        torch_any = cast(Any, torch)
        from_numpy = torch_any.from_numpy
        weight = cast(torch.Tensor, from_numpy(dense_array))
        if weight.is_floating_point():
            weight = weight.to(dtype=self.dtype)
        if tuple(weight.shape) != self.shape:
            raise ValueError(
                f"dequantized weight {self.name!r} has shape {tuple(weight.shape)}, "
                f"expected {self.shape}"
            )
        if self.transform == "llama_attention_unpermute":
            weight = _unpermute_llama_attention_weight(
                weight, self._required_n_heads(), self._required_head_dim()
            )
        if device is not None:
            weight = weight.to(device=device)
        return weight

    def _required_n_heads(self) -> int:
        if self.n_heads is None:
            raise ValueError(f"missing n_heads for transform on {self.name!r}")
        return self.n_heads

    def _required_head_dim(self) -> int:
        if self.head_dim is None:
            raise ValueError(f"missing head_dim for transform on {self.name!r}")
        return self.head_dim


DenseOrQuantizedWeight = torch.Tensor | QuantizedWeight


class QuantizedLinear(nn.Module):
    """Linear layer backed by a packed quantized weight."""

    def __init__(
        self,
        weight: QuantizedWeight,
        *,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if weight.shape != (out_features, in_features):
            raise ValueError(
                f"quantized linear shape mismatch: expected "
                f"{(out_features, in_features)}, got {weight.shape}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        if bias is None:
            self.bias = None
        else:
            if tuple(bias.shape) != (out_features,):
                raise ValueError(
                    f"bias shape mismatch: expected {(out_features,)}, "
                    f"got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.dequantize(device=x.device).to(dtype=x.dtype)
        bias = (
            None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        )
        return F.linear(x, weight, bias)


def _unpermute_llama_attention_weight(
    weight: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Convert llama.cpp's GGUF Q/K row layout into this model's RoPE layout.

    GGUF stores Llama query/key projection rows in the pair-adjacent layout used
    by llama.cpp's RoPE kernels.  This project applies RoPE in split-half layout
    when ``use_interleaved=False``: all first-half coordinates, then all
    second-half coordinates.  Quantized Q/K weights therefore need the same row
    permutation that the dense GGUF loader applies before ``F.linear`` sees the
    dequantized temporary weight.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2D attention weight, got {tuple(weight.shape)}")
    expected_rows = n_heads * head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"attention weight has {weight.shape[0]} rows, expected {expected_rows}"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"attention head_dim must be even, got {head_dim}")

    return (
        weight.reshape(n_heads, head_dim // 2, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape(weight.shape)
    )
