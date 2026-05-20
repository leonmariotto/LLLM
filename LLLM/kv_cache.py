"""Reusable key/value cache primitives for autoregressive attention."""

from __future__ import annotations

import torch


class KVCache:
    """
    Store per-layer key/value tensors for incremental decoding.

    The cache stores tensors in multi-head attention layout:
    ``[batch, heads, tokens, head_dim]``.
    """

    def __init__(self, max_seq_len: int | None = None) -> None:
        self.max_seq_len = max_seq_len
        self._layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self) -> None:
        """Drop all cached layer state."""
        self._layers.clear()

    def layer_seq_len(self, layer_idx: int) -> int:
        """Return the number of cached tokens for one layer."""
        cached = self._layers.get(layer_idx)
        if cached is None:
            return 0
        keys, _ = cached
        return int(keys.shape[-2])

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached ``(keys, values)`` for a layer, if present."""
        return self._layers.get(layer_idx)

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value tensors and return the complete layer cache."""
        self._validate_new_tensors(keys, values)

        cached = self._layers.get(layer_idx)
        if cached is None:
            keys_all = keys
            values_all = values
        else:
            keys_cached, values_cached = cached
            self._validate_cached_tensors(keys_cached, values_cached, keys, values)
            keys_all = torch.cat((keys_cached, keys), dim=-2)
            values_all = torch.cat((values_cached, values), dim=-2)

        if self.max_seq_len is not None and keys_all.shape[-2] > self.max_seq_len:
            raise ValueError(
                f"KV cache length {keys_all.shape[-2]} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        self._layers[layer_idx] = (keys_all, values_all)
        return keys_all, values_all

    @staticmethod
    def _validate_new_tensors(keys: torch.Tensor, values: torch.Tensor) -> None:
        """Sanity check before adding tensors to KV cache."""
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("KV cache tensors must be 4D [batch, heads, tokens, dim]")
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("KV cache key and value tensors must have matching shapes")
        if keys.device != values.device:
            raise ValueError(
                "KV cache key and value tensors must be on the same device"
            )
        if keys.dtype != values.dtype:
            raise ValueError("KV cache key and value tensors must have the same dtype")

    @staticmethod
    def _validate_cached_tensors(
        keys_cached: torch.Tensor,
        values_cached: torch.Tensor,
        keys_new: torch.Tensor,
        values_new: torch.Tensor,
    ) -> None:
        """
        Used to validate cached tensor and new tensor before concatenate the
        whole and giving it back to the model.
        """
        if (
            keys_cached.device != keys_new.device
            or values_cached.device != values_new.device
        ):
            raise ValueError("cannot append KV cache tensors from a different device")
        if (
            keys_cached.dtype != keys_new.dtype
            or values_cached.dtype != values_new.dtype
        ):
            raise ValueError("cannot append KV cache tensors with a different dtype")
        if tuple(keys_cached.shape[:2]) != tuple(keys_new.shape[:2]):
            raise ValueError(
                "cannot append KV cache tensors with different batch/head shape"
            )
        if tuple(values_cached.shape[:2]) != tuple(values_new.shape[:2]):
            raise ValueError(
                "cannot append KV cache tensors with different batch/head shape"
            )
        if int(keys_cached.shape[-1]) != int(keys_new.shape[-1]):
            raise ValueError("cannot append KV cache keys with a different head_dim")
        if int(values_cached.shape[-1]) != int(values_new.shape[-1]):
            raise ValueError("cannot append KV cache values with a different head_dim")
