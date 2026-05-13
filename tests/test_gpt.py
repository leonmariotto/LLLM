import pytest
import torch
from torch import nn
from typing import Any, Callable, cast

from ..LLLM.gpt import (
    GPT_CONFIG_124M,
    GPTConfig,
    GPTModel,
    gpt_config_from_fetched,
)


_manual_seed = cast(Callable[[int], torch.Generator], cast(Any, torch).manual_seed)


def _tiny_gpt_config() -> GPTConfig:
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


def _tiny_transformer_gpt_config() -> GPTConfig:
    """Return a tiny GPT configuration with one transformer block for cache tests."""
    cfg = _tiny_gpt_config().copy()
    cfg["emb_dim"] = 4
    cfg["n_layers"] = 1
    return cfg


def _tiny_rope_gpt_config() -> GPTConfig:
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


def _manual_layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute layer normalization directly for expected-value comparisons."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def test_gpt_config_from_fetched_translates_hugging_face_names() -> None:
    cfg = gpt_config_from_fetched(_tiny_hf_gpt2_config())

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
    model = GPTModel(cfg)

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
    model = GPTModel(cfg)

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
    model = GPTModel(cfg)
    in_idx = torch.tensor([[0, 1, 2, 3, 4]])

    with pytest.raises(AssertionError):
        model(in_idx)


def test_gpt_check_gpt124_size() -> None:
    """Check GPT_CONFIG_124M size, we see that model is 124M without output layer weight, but this
    are valid trainable parameters. A model branded XXX size is in reality a bit bigger.
    """
    _manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT_CONFIG_124M: Total number of parameters: {total_params:,}")
    # assert total_params == 163009536 without qv_bias
    assert total_params == 163037184
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(
        f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}"
    )
    # assert total_params_gpt2 == 124412160 # without qv_bias
    assert total_params_gpt2 == 124439808
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
