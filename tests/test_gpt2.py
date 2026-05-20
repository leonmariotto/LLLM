from typing import Any, Callable, cast

import pytest
import torch
from torch import nn

from ..LLLM.gpt2 import (
    GPT2Config,
    GPT2FeedForward,
    GPT2Model,
    GPT2MultiHeadAttention,
    GPT2TransformerBlock,
)
from ..LLLM.hf_loader import model_ir_from_hf
from ..LLLM.norm import LayerNorm


_manual_seed = cast(Callable[[int], torch.Generator], cast(Any, torch).manual_seed)


def _tiny_gpt_config() -> GPT2Config:
    """Return a minimal GPT configuration used by the tests in this module."""
    return {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 3,
        "n_heads": 1,
        "n_layers": 0,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "gpt2",
    }


def _tiny_transformer_gpt_config() -> GPT2Config:
    """Return a tiny GPT configuration with one transformer block for cache tests."""
    cfg = _tiny_gpt_config().copy()
    cfg["emb_dim"] = 4
    cfg["n_layers"] = 1
    return cfg


def _tiny_rope_gpt_config() -> GPT2Config:
    """Return a tiny GPT configuration that uses RoPE in attention."""
    cfg = _tiny_transformer_gpt_config().copy()
    cfg["positional_encoding"] = "rope"
    return cfg


def _tiny_hf_gpt2_config() -> dict[str, int | str | float]:
    return {
        "model_type": "gpt2",
        "vocab_size": 5,
        "n_positions": 4,
        "n_embd": 4,
        "n_head": 1,
        "n_layer": 1,
        "resid_pdrop": 0.0,
    }

def test_multi_head_attention() -> None:
    torch.manual_seed(42)  # Let there be order among chaos.
    d_in = 3
    d_out = 8
    num_heads = 2
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ],
        requires_grad=False,
    )
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    mha = GPT2MultiHeadAttention(
        d_in,
        d_out,
        context_length,
        dropout=0.0,
        num_heads=num_heads,
        use_rope=True,
    )
    context_vecs = mha(batch)
    expected_context_vecs = torch.tensor(
        [
            [
                [
                    -0.5691065192222595,
                    -0.26063844561576843,
                    0.2583349347114563,
                    -0.23653504252433777,
                    0.00406038761138916,
                    -0.3905044496059418,
                    0.2901259660720825,
                    -0.43187201023101807,
                ],
                [
                    -0.5106597542762756,
                    -0.31405866146087646,
                    0.19704855978488922,
                    -0.29567795991897583,
                    0.022763952612876892,
                    -0.4193854331970215,
                    0.2607108950614929,
                    -0.33293426036834717,
                ],
                [
                    -0.4933360815048218,
                    -0.3250187635421753,
                    0.1774100661277771,
                    -0.309073269367218,
                    0.031104832887649536,
                    -0.4243485927581787,
                    0.2536696195602417,
                    -0.30401110649108887,
                ],
                [
                    -0.4708804488182068,
                    -0.33434367179870605,
                    0.16520822048187256,
                    -0.30085235834121704,
                    0.03795764595270157,
                    -0.4146912693977356,
                    0.2468728870153427,
                    -0.2860690951347351,
                ],
                [
                    -0.45166879892349243,
                    -0.29535219073295593,
                    0.13360558450222015,
                    -0.26248160004615784,
                    0.07704773545265198,
                    -0.3824581503868103,
                    0.25940313935279846,
                    -0.26681721210479736,
                ],
                [
                    -0.441005140542984,
                    -0.33338701725006104,
                    0.14248201251029968,
                    -0.2785763442516327,
                    0.05836986005306244,
                    -0.39788827300071716,
                    0.24770411849021912,
                    -0.2636151909828186,
                ],
            ],
            [
                [
                    -0.5691065192222595,
                    -0.26063844561576843,
                    0.2583349347114563,
                    -0.23653504252433777,
                    0.00406038761138916,
                    -0.3905044496059418,
                    0.2901259660720825,
                    -0.43187201023101807,
                ],
                [
                    -0.5106597542762756,
                    -0.31405866146087646,
                    0.19704855978488922,
                    -0.29567795991897583,
                    0.022763952612876892,
                    -0.4193854331970215,
                    0.2607108950614929,
                    -0.33293426036834717,
                ],
                [
                    -0.4933360815048218,
                    -0.3250187635421753,
                    0.1774100661277771,
                    -0.309073269367218,
                    0.031104832887649536,
                    -0.4243485927581787,
                    0.2536696195602417,
                    -0.30401110649108887,
                ],
                [
                    -0.4708804488182068,
                    -0.33434367179870605,
                    0.16520822048187256,
                    -0.30085235834121704,
                    0.03795764595270157,
                    -0.4146912693977356,
                    0.2468728870153427,
                    -0.2860690951347351,
                ],
                [
                    -0.45166879892349243,
                    -0.29535219073295593,
                    0.13360558450222015,
                    -0.26248160004615784,
                    0.07704773545265198,
                    -0.3824581503868103,
                    0.25940313935279846,
                    -0.26681721210479736,
                ],
                [
                    -0.441005140542984,
                    -0.33338701725006104,
                    0.14248201251029968,
                    -0.2785763442516327,
                    0.05836986005306244,
                    -0.39788827300071716,
                    0.24770411849021912,
                    -0.2636151909828186,
                ],
            ],
        ]
    )
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)
    torch.testing.assert_close(context_vecs, expected_context_vecs)


def test_multi_head_attention_gpt2_scaled() -> None:
    d_in = 768
    d_out = 768
    num_heads = 12
    context_length = 1024
    batch_count = 5
    batch = torch.rand(batch_count, context_length, d_in)
    mha = GPT2MultiHeadAttention(
        d_in, d_out, context_length, dropout=0.0, num_heads=num_heads
    )
    context_vecs = mha(batch)
    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)


def _manual_layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute layer normalization directly for expected-value comparisons."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

class AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        """Store the constant added to every tensor element."""
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        """Return the input tensor with the configured constant added."""
        return x + self.value


class ScaleBy(nn.Module):
    def __init__(self, factor: float) -> None:
        """Store the multiplicative factor applied in the forward pass."""
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor scaled by the configured factor."""
        return x * self.factor


def test_feed_forward_matches_manual_two_layer_computation() -> None:
    """Check FeedForward matches an explicit two-linear-layer GELU computation."""
    feed_forward = GPT2FeedForward(embedded_dimension=2, expansion_factor=2)

    # Pull out the internal linear layers so the expected result can be computed step by step.
    first_linear = feed_forward.fc1
    second_linear = feed_forward.fc2

    with torch.no_grad():
        first_linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
        first_linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        second_linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.5, -0.5],
                    [0.0, 1.0, -0.5, 0.5],
                ]
            )
        )
        second_linear.bias.copy_(torch.tensor([0.2, -0.1]))

    inputs = torch.tensor([[[0.5, -1.0], [1.5, 0.25]]])

    outputs = feed_forward(inputs)

    # Rebuild the forward pass explicitly: linear -> GELU -> linear.
    hidden = inputs @ first_linear.weight.T + first_linear.bias
    activated = torch.nn.functional.gelu(hidden, approximate="tanh")
    expected_outputs = activated @ second_linear.weight.T + second_linear.bias

    torch.testing.assert_close(outputs, expected_outputs)


def test_layer_norm_matches_manual_normalization_with_scale_and_shift() -> None:
    """Confirm LayerNorm applies normalization, scaling, and shifting correctly."""
    layer_norm = LayerNorm(emb_dim=3, eps=1e-5)
    inputs = torch.tensor([[[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]])

    with torch.no_grad():
        layer_norm.scale.copy_(torch.tensor([1.5, -2.0, 0.5]))
        layer_norm.shift.copy_(torch.tensor([0.2, 0.3, -0.4]))

    outputs = layer_norm(inputs)

    # Match the implementation details, including population variance.
    mean = inputs.mean(dim=-1, keepdim=True)
    var = inputs.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) / torch.sqrt(var + layer_norm.eps)
    expected_outputs = normalized * layer_norm.scale + layer_norm.shift

    torch.testing.assert_close(outputs, expected_outputs)


def test_transformer_block_applies_both_residual_paths() -> None:
    """Ensure GPT2TransformerBlock applies attention and feed-forward residual paths."""
    cfg = {
        "emb_dim": 2,
        "context_length": 3,
        "n_heads": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "gpt2",
    }
    block = GPT2TransformerBlock(cfg)
    # Replace submodules with deterministic operations so the residual math is easy to verify.
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.att = AddConstant(1.0)
    block.ff = ScaleBy(2.0)
    block.drop_shortcut = nn.Identity()

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    outputs = block(inputs)
    # First residual path: x + (x + 1) = 2x + 1. Second path: (2x + 1) + 2(2x + 1) = 6x + 3.
    expected_outputs = 6 * inputs + 3

    torch.testing.assert_close(outputs, expected_outputs)

def test_gpt_config_from_ir_translates_hugging_face_names() -> None:
    ir = model_ir_from_hf(_tiny_hf_gpt2_config(), {}, architecture="gpt2")
    cfg = GPT2Model.config_from_ir(ir)

    assert cfg == {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 4,
        "n_heads": 1,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "positional_encoding": "gpt2",
    }


def test_gpt_forward_matches_manual_computation_without_transformer_blocks() -> None:
    """Verify GPT logits match a manual embedding, norm, and output-head computation."""
    cfg = _tiny_gpt_config()
    model = GPT2Model(cfg)

    with torch.no_grad():
        model.tok_emb.weight.copy_(
            torch.tensor(
                [
                    [0.10, 0.20, 0.30],
                    [0.40, 0.50, 0.60],
                    [0.70, 0.80, 0.90],
                    [1.00, 1.10, 1.20],
                    [1.30, 1.40, 1.50],
                ]
            )
        )
        assert model.pos_emb is not None
        model.pos_emb.weight.copy_(
            torch.tensor(
                [
                    [0.01, 0.02, 0.03],
                    [0.04, 0.05, 0.06],
                    [0.07, 0.08, 0.09],
                    [0.10, 0.11, 0.12],
                ]
            )
        )
        model.final_norm.scale.fill_(1.0)
        model.final_norm.shift.zero_()
        model.out_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )

    in_idx = torch.tensor([[0, 1, 2], [2, 1, 0]])

    logits = model(in_idx)

    # GPT-2 positional encoding adds learned position embeddings before the
    # transformer blocks.
    tok_embeds = model.tok_emb.weight[in_idx]
    assert model.pos_emb is not None
    pos_embeds = model.pos_emb.weight[: in_idx.shape[1]]
    x = tok_embeds + pos_embeds
    x = _manual_layer_norm(x)
    expected_logits = x @ model.out_head.weight.T

    assert logits.shape == (2, 3, cfg["vocab_size"])
    torch.testing.assert_close(logits, expected_logits)


def test_gpt_rope_is_invariant_to_global_position_shift() -> None:
    """Check RoPE depends on relative positions, not absolute position embeddings."""
    cfg = _tiny_rope_gpt_config()
    model = GPT2Model(cfg)

    with torch.no_grad():
        model.tok_emb.weight.copy_(
            torch.tensor(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.50, 0.60, 0.70, 0.80],
                    [0.90, 1.00, 1.10, 1.20],
                    [1.30, 1.40, 1.50, 1.60],
                    [1.70, 1.80, 1.90, 2.00],
                ]
            )
        )
        model.final_norm.scale.fill_(1.0)
        model.final_norm.shift.zero_()

    in_idx = torch.tensor([[0, 1, 0]])

    logits_at_zero = model(in_idx, pos=0)
    logits_at_one = model(in_idx, pos=1)

    assert logits_at_zero.shape == (1, 3, cfg["vocab_size"])
    torch.testing.assert_close(logits_at_zero, logits_at_one)


def test_gpt_rejects_sequences_longer_than_context_length() -> None:
    """Ensure GPT raises IndexError for inputs longer than its context length."""
    cfg = _tiny_rope_gpt_config()
    model = GPT2Model(cfg)
    in_idx = torch.tensor([[0, 1, 2, 3, 4]])

    with pytest.raises(AssertionError):
        model(in_idx)
