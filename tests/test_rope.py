import pytest
import torch

torch = pytest.importorskip("torch")

from ..LLLM.rope import (
    apply_rope,
    apply_rope_interleaved,
    apply_rope_split_half,
    precompute_rope_cache,
)


def test_precompute_rope_cache_returns_expected_shape_and_values() -> None:
    cos, sin = precompute_rope_cache(seq_len=3, head_dim=4)

    assert cos.shape == (3, 2)
    assert sin.shape == (3, 2)

    expected_angles = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.01],
            [2.0, 0.02],
        ]
    )
    torch.testing.assert_close(cos, torch.cos(expected_angles))
    torch.testing.assert_close(sin, torch.sin(expected_angles))


def test_apply_rope_interleaved_rotates_adjacent_pairs() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    cos, sin = precompute_rope_cache(seq_len=2, head_dim=4)

    rotated = apply_rope_interleaved(x, cos, sin)

    c0, c1 = cos[1]
    s0, s1 = sin[1]
    expected = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [
                        5.0 * c0 - 6.0 * s0,
                        5.0 * s0 + 6.0 * c0,
                        7.0 * c1 - 8.0 * s1,
                        7.0 * s1 + 8.0 * c1,
                    ],
                ]
            ]
        ]
    )

    torch.testing.assert_close(rotated, expected)


def test_apply_rope_split_half_rotates_first_half_against_second_half() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
    cos, sin = precompute_rope_cache(seq_len=2, head_dim=4)

    rotated = apply_rope_split_half(x, cos, sin)

    c0, c1 = cos[1]
    s0, s1 = sin[1]
    expected = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [
                        5.0 * c0 - 7.0 * s0,
                        6.0 * c1 - 8.0 * s1,
                        5.0 * s0 + 7.0 * c0,
                        6.0 * s1 + 8.0 * c1,
                    ],
                ]
            ]
        ]
    )

    torch.testing.assert_close(rotated, expected)


def test_apply_rope_dispatches_to_selected_layout() -> None:
    x = torch.randn(2, 3, 4, 6)
    cos, sin = precompute_rope_cache(seq_len=4, head_dim=6)

    torch.testing.assert_close(
        apply_rope(x, cos, sin),
        apply_rope_interleaved(x, cos, sin),
    )
    torch.testing.assert_close(
        apply_rope(x, cos, sin, use_interleaved=False),
        apply_rope_split_half(x, cos, sin),
    )


def test_apply_rope_handles_empty_sequence() -> None:
    x = torch.empty(2, 3, 0, 4)
    cos, sin = precompute_rope_cache(seq_len=0, head_dim=4)

    interleaved = apply_rope_interleaved(x, cos, sin)
    split_half = apply_rope_split_half(x, cos, sin)

    assert interleaved.shape == x.shape
    assert split_half.shape == x.shape
    assert interleaved.numel() == 0
    assert split_half.numel() == 0


def test_precompute_rope_cache_rejects_odd_head_dim() -> None:
    with pytest.raises(AssertionError, match="even head_dim"):
        precompute_rope_cache(seq_len=2, head_dim=3)


def test_apply_rope_split_half_rejects_odd_head_dim() -> None:
    x = torch.empty(1, 1, 2, 3)
    cos = torch.empty(2, 1)
    sin = torch.empty(2, 1)

    with pytest.raises(AssertionError, match="even head_dim"):
        apply_rope_split_half(x, cos, sin)


def test_apply_rope_rejects_mismatched_cache_shape() -> None:
    x = torch.empty(1, 1, 2, 4)
    cos = torch.empty(2, 3)
    sin = torch.empty(2, 3)

    with pytest.raises(RuntimeError):
        apply_rope_interleaved(x, cos, sin)

