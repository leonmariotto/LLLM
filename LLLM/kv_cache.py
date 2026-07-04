"""
Reusable key/value cache primitives for autoregressive attention.
Support for sliding cache window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheView:
    """Keys/values retained for attention plus their absolute start position."""

    keys: torch.Tensor
    values: torch.Tensor
    start_pos: int

    def __iter__(self):
        """Allow legacy tuple-unpacking as ``keys, values = cache.update(...)``."""
        yield self.keys
        yield self.values


@dataclass
class _KVCacheStorage:
    keys: torch.Tensor
    values: torch.Tensor
    capacity: int


class KVCache:
    """
    Store per-layer key/value tensors for incremental decoding.

    The cache stores tensors in multi-head attention layout:
    ``[batch, heads, tokens, head_dim]``.
    """

    def __init__(
        self, cache_length: int | None = None, *, sliding: bool = True
    ) -> None:
        if cache_length is not None and cache_length <= 0:
            raise ValueError("cache_length must be positive")
        self.cache_length = cache_length
        self.sliding = sliding
        self._layers: dict[int, KVCacheView] = {}
        self._storage: dict[int, _KVCacheStorage] = {}
        self._next_pos: dict[int, int] = {}

    def reset(self) -> None:
        """Drop all cached layer state."""
        self._layers.clear()
        self._storage.clear()
        self._next_pos.clear()

    def layer_seq_len(self, layer_idx: int) -> int:
        """Return the retained cache length for one layer."""
        cached = self._layers.get(layer_idx)
        if cached is None:
            return 0
        return int(cached.keys.shape[-2])

    def layer_start_pos(self, layer_idx: int) -> int:
        """Return the absolute position of the first retained token."""
        cached = self._layers.get(layer_idx)
        if cached is None:
            return self.layer_next_pos(layer_idx)
        return cached.start_pos

    def layer_next_pos(self, layer_idx: int) -> int:
        """Return the absolute position for the next token appended to a layer."""
        return self._next_pos.get(layer_idx, 0)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached ``(keys, values)`` for a layer, if present."""
        cached = self._layers.get(layer_idx)
        if cached is None:
            return None
        return cached.keys, cached.values

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start_pos: int | None = None,
    ) -> KVCacheView:
        """Append new key/value tensors and return the retained layer cache."""
        self._validate_new_tensors(keys, values)
        if start_pos is None:
            start_pos = self.layer_next_pos(layer_idx)
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        expected_pos = self.layer_next_pos(layer_idx)
        if start_pos != expected_pos:
            raise ValueError(
                f"cannot append KV cache at absolute position {start_pos}; "
                f"expected {expected_pos}"
            )

        cached = self._layers.get(layer_idx)
        cached_length = 0
        if cached is not None:
            self._validate_cached_tensors(cached.keys, cached.values, keys, values)
            cached_length = int(cached.keys.shape[-2])

        next_pos = start_pos + int(keys.shape[-2])
        combined_length = cached_length + int(keys.shape[-2])
        if (
            self.cache_length is not None
            and combined_length > self.cache_length
            and not self.sliding
        ):
            raise ValueError(
                f"KV cache length {combined_length} exceeds cache_length "
                f"{self.cache_length}"
            )

        retained_length = (
            combined_length
            if self.cache_length is None
            else min(combined_length, self.cache_length)
        )
        previous_storage = self._storage.get(layer_idx)
        storage = self._ensure_capacity(
            layer_idx,
            keys,
            values,
            retained_length,
            cached,
        )

        incoming_length = int(keys.shape[-2])
        if incoming_length >= retained_length:
            storage.keys[:, :, :retained_length, :].copy_(
                keys[:, :, -retained_length:, :]
            )
            storage.values[:, :, :retained_length, :].copy_(
                values[:, :, -retained_length:, :]
            )
        else:
            retained_cached_length = retained_length - incoming_length
            if retained_cached_length:
                assert cached is not None
                if not (
                    storage is previous_storage
                    and retained_cached_length == cached_length
                ):
                    cached_keys = cached.keys[:, :, -retained_cached_length:, :]
                    cached_values = cached.values[:, :, -retained_cached_length:, :]
                    if storage is previous_storage:
                        cached_keys = cached_keys.clone()
                        cached_values = cached_values.clone()
                    storage.keys[:, :, :retained_cached_length, :].copy_(cached_keys)
                    storage.values[:, :, :retained_cached_length, :].copy_(
                        cached_values
                    )
            storage.keys[:, :, retained_cached_length:retained_length, :].copy_(keys)
            storage.values[:, :, retained_cached_length:retained_length, :].copy_(
                values
            )

        keys_all = storage.keys[:, :, :retained_length, :]
        values_all = storage.values[:, :, :retained_length, :]
        cache_start_pos = next_pos - retained_length

        view = KVCacheView(keys=keys_all, values=values_all, start_pos=cache_start_pos)
        self._layers[layer_idx] = view
        self._next_pos[layer_idx] = next_pos
        return view

    def _ensure_capacity(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        required: int,
        cached: KVCacheView | None,
    ) -> _KVCacheStorage:
        storage = self._storage.get(layer_idx)
        if storage is not None and storage.capacity >= required:
            return storage

        capacity = 1 if storage is None else storage.capacity
        while capacity < required:
            capacity *= 2
        if self.cache_length is not None:
            capacity = min(capacity, self.cache_length)

        shape = (*keys.shape[:-2], capacity, keys.shape[-1])
        new_storage = _KVCacheStorage(
            keys=keys.new_empty(shape),
            values=values.new_empty(shape),
            capacity=capacity,
        )
        if cached is not None:
            cached_length = int(cached.keys.shape[-2])
            new_storage.keys[:, :, :cached_length, :].copy_(cached.keys)
            new_storage.values[:, :, :cached_length, :].copy_(cached.values)
        self._storage[layer_idx] = new_storage
        return new_storage

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
