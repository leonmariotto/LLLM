import pytest
import torch

from ..LLLM.kv_cache import KVCache


def test_kv_cache_appends_per_layer_state() -> None:
    cache = KVCache()
    keys_a = torch.ones(1, 2, 3, 4)
    values_a = torch.full((1, 2, 3, 4), 2.0)
    keys_b = torch.full((1, 2, 1, 4), 3.0)
    values_b = torch.full((1, 2, 1, 4), 4.0)

    keys, values = cache.update(0, keys_a, values_a)

    assert cache.layer_seq_len(0) == 3
    torch.testing.assert_close(keys, keys_a)
    torch.testing.assert_close(values, values_a)

    keys, values = cache.update(0, keys_b, values_b)

    assert cache.layer_seq_len(0) == 4
    torch.testing.assert_close(keys[:, :, :3, :], keys_a)
    torch.testing.assert_close(keys[:, :, 3:, :], keys_b)
    torch.testing.assert_close(values[:, :, :3, :], values_a)
    torch.testing.assert_close(values[:, :, 3:, :], values_b)


def test_kv_cache_enforces_max_seq_len() -> None:
    cache = KVCache(max_seq_len=2)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        cache.update(0, torch.ones(1, 1, 3, 2), torch.ones(1, 1, 3, 2))


def test_kv_cache_rejects_mismatched_shapes() -> None:
    cache = KVCache()

    with pytest.raises(ValueError, match="matching shapes"):
        cache.update(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 2, 2))

    cache.update(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))

    with pytest.raises(ValueError, match="batch/head"):
        cache.update(0, torch.ones(1, 2, 1, 2), torch.ones(1, 2, 1, 2))
