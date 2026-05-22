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
    assert cache.layer_start_pos(0) == 0
    assert cache.layer_next_pos(0) == 3
    torch.testing.assert_close(keys, keys_a)
    torch.testing.assert_close(values, values_a)

    keys, values = cache.update(0, keys_b, values_b)

    assert cache.layer_seq_len(0) == 4
    assert cache.layer_start_pos(0) == 0
    assert cache.layer_next_pos(0) == 4
    torch.testing.assert_close(keys[:, :, :3, :], keys_a)
    torch.testing.assert_close(keys[:, :, 3:, :], keys_b)
    torch.testing.assert_close(values[:, :, :3, :], values_a)
    torch.testing.assert_close(values[:, :, 3:, :], values_b)


def test_kv_cache_slides_by_default_at_cache_length() -> None:
    cache = KVCache(cache_length=2)

    keys = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)
    values = keys + 10

    view = cache.update(0, keys, values)

    assert cache.layer_seq_len(0) == 2
    assert cache.layer_start_pos(0) == 1
    assert cache.layer_next_pos(0) == 3
    torch.testing.assert_close(view.keys, keys[:, :, 1:, :])
    torch.testing.assert_close(view.values, values[:, :, 1:, :])


def test_kv_cache_can_enforce_cache_length_without_sliding() -> None:
    cache = KVCache(cache_length=2, sliding=False)

    with pytest.raises(ValueError, match="exceeds cache_length"):
        cache.update(0, torch.ones(1, 1, 3, 2), torch.ones(1, 1, 3, 2))


def test_kv_cache_rejects_non_contiguous_absolute_position() -> None:
    cache = KVCache()
    cache.update(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))

    with pytest.raises(ValueError, match="expected 1"):
        cache.update(
            0,
            torch.ones(1, 1, 1, 2),
            torch.ones(1, 1, 1, 2),
            start_pos=3,
        )


def test_kv_cache_rejects_mismatched_shapes() -> None:
    cache = KVCache()

    with pytest.raises(ValueError, match="matching shapes"):
        cache.update(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 2, 2))

    cache.update(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))

    with pytest.raises(ValueError, match="batch/head"):
        cache.update(0, torch.ones(1, 2, 1, 2), torch.ones(1, 2, 1, 2))
