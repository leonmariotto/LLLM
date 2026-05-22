from collections.abc import Sequence

import pytest
import torch
from torch import nn

from ..LLLM.llama2 import (
    Llama2Config,
    Llama2FeedForward,
    Llama2Model,
    Llama2MultiHeadAttention,
    Llama2Tokenizer,
    Llama2TransformerBlock,
)
from ..LLLM.kv_cache import KVCache
from ..LLLM.norm import RMSNorm
from ..LLLM.hf_loader import model_ir_from_hf


def _tiny_llama_config() -> Llama2Config:
    """Return a minimal Llama configuration used by the tests in this module."""
    return {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 3,
        "n_heads": 1,
        "n_layers": 0,
        "hidden_dim": 4,
        "rope_theta": 10000.0,
        "dtype": torch.float32,
    }


def _tiny_transformer_llama_config() -> Llama2Config:
    """Return a tiny Llama configuration with one transformer block."""
    cfg = _tiny_llama_config().copy()
    cfg["emb_dim"] = 4
    cfg["n_layers"] = 1
    return cfg


def _tiny_hf_llama_config() -> dict[str, int | float | str]:
    return {
        "model_type": "llama",
        "vocab_size": 5,
        "max_position_embeddings": 4,
        "hidden_size": 4,
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "rope_theta": 10000.0,
    }


def _manual_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute RMSNorm directly for expected-value comparisons."""
    means = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(means + eps)


class AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        return x + self.value


class ScaleBy(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class FakeSentencePieceProcessor:
    def __init__(self) -> None:
        self.loaded_path: str | None = None

    def load(self, tokenizer_file: str) -> bool:
        self.loaded_path = tokenizer_file
        return True

    def encode(self, text: str, *, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [ord(char) for char in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(idx) for idx in ids)


def test_feed_forward_matches_manual_swiglu_computation() -> None:
    feed_forward = Llama2FeedForward(emb_dim=2, hidden_dim=3, dtype=None)

    with torch.no_grad():
        feed_forward.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        )
        feed_forward.fc2.weight.copy_(
            torch.tensor(
                [
                    [0.5, 1.0],
                    [1.5, -0.5],
                    [-1.0, 0.25],
                ]
            )
        )
        feed_forward.fc3.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, -0.5],
                    [0.25, -1.0, 0.5],
                ]
            )
        )

    inputs = torch.tensor([[[0.5, -1.0], [1.5, 0.25]]])

    outputs = feed_forward(inputs)

    x_fc1 = inputs @ feed_forward.fc1.weight.T
    x_fc2 = inputs @ feed_forward.fc2.weight.T
    hidden = torch.nn.functional.silu(x_fc1) * x_fc2
    expected_outputs = hidden @ feed_forward.fc3.weight.T

    torch.testing.assert_close(outputs, expected_outputs)


def test_rms_norm_matches_manual_normalization_with_weight() -> None:
    norm = RMSNorm(emb_dim=3, eps=1e-5)
    inputs = torch.tensor([[[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]])

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.5, -2.0, 0.5]))

    outputs = norm(inputs)

    expected_outputs = _manual_rms_norm(inputs, norm.eps) * norm.weight
    torch.testing.assert_close(outputs, expected_outputs)


def test_multi_head_attention_without_rope_matches_manual_causal_attention() -> None:
    attention = Llama2MultiHeadAttention(
        d_in=2,
        d_out=2,
        context_length=3,
        num_heads=1,
        dropout=0.0,
        qkv_bias=False,
        use_rope=False,
    )

    with torch.no_grad():
        identity = torch.eye(2)
        attention.W_query.weight.copy_(identity)
        attention.W_key.weight.copy_(identity)
        attention.W_value.weight.copy_(identity)
        attention.out_proj.weight.copy_(identity)

    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])

    outputs = attention(inputs)

    queries = inputs
    keys = inputs
    values = inputs
    scores = queries @ keys.transpose(1, 2)
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, -torch.inf)
    weights = torch.softmax(scores / 2**0.5, dim=-1)
    expected_outputs = weights @ values

    torch.testing.assert_close(outputs, expected_outputs)


def test_multi_head_attention_with_kv_cache_matches_full_attention_last_token() -> None:
    attention = Llama2MultiHeadAttention(
        d_in=2,
        d_out=2,
        context_length=4,
        num_heads=1,
        dropout=0.0,
        qkv_bias=False,
        use_rope=True,
    )

    with torch.no_grad():
        identity = torch.eye(2)
        attention.W_query.weight.copy_(identity)
        attention.W_key.weight.copy_(identity)
        attention.W_value.weight.copy_(identity)
        attention.out_proj.weight.copy_(identity)

    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])

    full_outputs = attention(inputs)
    cache = KVCache()
    attention(inputs[:, :2, :], kv_cache=cache, layer_idx=0)
    cached_outputs = attention(inputs[:, 2:, :], kv_cache=cache, layer_idx=0)

    assert cache.layer_seq_len(0) == 3
    torch.testing.assert_close(cached_outputs, full_outputs[:, 2:, :])


def test_multi_head_attention_sliding_cache_uses_absolute_rope_positions() -> None:
    attention = Llama2MultiHeadAttention(
        d_in=2,
        d_out=2,
        context_length=4,
        num_heads=1,
        dropout=0.0,
        qkv_bias=False,
        use_rope=True,
    )

    with torch.no_grad():
        identity = torch.eye(2)
        attention.W_query.weight.copy_(identity)
        attention.W_key.weight.copy_(identity)
        attention.W_value.weight.copy_(identity)
        attention.out_proj.weight.copy_(identity)

    inputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    cache = KVCache(cache_length=2)

    attention(inputs[:, :2, :], kv_cache=cache, layer_idx=0)
    cached_outputs = attention(inputs[:, 2:, :], kv_cache=cache, layer_idx=0)
    expected_outputs = attention(inputs[:, 1:, :], pos=1)

    assert cache.layer_seq_len(0) == 2
    assert cache.layer_start_pos(0) == 1
    assert cache.layer_next_pos(0) == 3
    torch.testing.assert_close(cached_outputs, expected_outputs[:, -1:, :])


def test_multi_head_attention_scaled_shape() -> None:
    d_in = 16
    d_out = 16
    num_heads = 4
    context_length = 8
    batch = torch.rand(2, context_length, d_in)
    attention = Llama2MultiHeadAttention(
        d_in,
        d_out,
        context_length,
        dropout=0.0,
        num_heads=num_heads,
    )

    context_vecs = attention(batch)

    assert context_vecs.shape == (batch.shape[0], batch.shape[1], d_out)


def test_multi_head_attention_rejects_invalid_dimensions() -> None:
    with pytest.raises(AssertionError, match="num_head shall not be 0"):
        Llama2MultiHeadAttention(
            d_in=2,
            d_out=2,
            context_length=3,
            num_heads=0,
        )

    with pytest.raises(AssertionError, match="d_out must be divisible"):
        Llama2MultiHeadAttention(
            d_in=2,
            d_out=3,
            context_length=3,
            num_heads=2,
        )


def test_multi_head_attention_rejects_invalid_input_embedding_size() -> None:
    attention = Llama2MultiHeadAttention(
        d_in=2,
        d_out=2,
        context_length=3,
        num_heads=1,
    )
    inputs = torch.empty(1, 3, 3)

    with pytest.raises(AssertionError, match="invalid d_in"):
        attention(inputs)


def test_multi_head_attention_rejects_rope_position_beyond_context_length() -> None:
    attention = Llama2MultiHeadAttention(
        d_in=2,
        d_out=2,
        context_length=3,
        num_heads=1,
    )
    inputs = torch.empty(1, 2, 2)

    with pytest.raises(AssertionError, match="RoPE position exceeds"):
        attention(inputs, pos=2)


def test_transformer_block_applies_both_residual_paths() -> None:
    cfg = _tiny_transformer_llama_config()
    block = Llama2TransformerBlock(cfg)
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.att = AddConstant(1.0)
    block.ff = ScaleBy(2.0)

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ]
    )

    outputs = block(inputs)

    expected_outputs = 6 * inputs + 3
    torch.testing.assert_close(outputs, expected_outputs)


def test_llama_config_from_ir_translates_hugging_face_names() -> None:
    ir = model_ir_from_hf(_tiny_hf_llama_config(), {}, architecture="llama2")
    cfg = Llama2Model.config_from_ir(ir)

    assert cfg == {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 4,
        "n_heads": 1,
        "n_layers": 1,
        "hidden_dim": 8,
        "rope_theta": 10000.0,
        "dtype": torch.float32,
    }


def test_llama_config_from_ir_rejects_missing_or_wrong_types() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        Llama2Model.config_from_ir(
            model_ir_from_hf({"model_type": "llama"}, {}, architecture="llama2")
        )

    bad_config = _tiny_hf_llama_config()
    bad_config["rope_theta"] = "10000"

    with pytest.raises(ValueError, match="rope_theta"):
        Llama2Model.config_from_ir(
            model_ir_from_hf(bad_config, {}, architecture="llama2")
        )


def test_llama_forward_matches_manual_computation_without_transformer_blocks() -> None:
    cfg = _tiny_llama_config()
    model = Llama2Model(cfg)

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
        model.final_norm.weight.fill_(1.0)
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

    tok_embeds = model.tok_emb.weight[in_idx]
    x = _manual_rms_norm(tok_embeds)
    expected_logits = x @ model.out_head.weight.T

    assert logits.shape == (2, 3, cfg["vocab_size"])
    torch.testing.assert_close(logits, expected_logits)


def test_llama_model_with_transformer_blocks_returns_expected_shape() -> None:
    cfg = _tiny_transformer_llama_config()
    model = Llama2Model(cfg)
    in_idx = torch.tensor([[0, 1, 2], [2, 1, 0]])

    logits = model(in_idx)

    assert logits.shape == (2, 3, cfg["vocab_size"])


def test_llama_model_with_kv_cache_matches_full_forward() -> None:
    cfg = _tiny_transformer_llama_config()
    model = Llama2Model(cfg)
    model.eval()
    in_idx = torch.tensor([[0, 1, 2]])

    with torch.no_grad():
        full_logits = model(in_idx)

        cache = KVCache()
        model(in_idx[:, :2], kv_cache=cache)
        cached_logits = model(in_idx[:, 2:], kv_cache=cache)

    torch.testing.assert_close(cached_logits, full_logits[:, 2:, :])

def test_llama_tokenizer_wraps_sentencepiece_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "LLLM.llama2.spm.SentencePieceProcessor",
        FakeSentencePieceProcessor,
    )

    tokenizer = Llama2Tokenizer("tokenizer.model")

    assert tokenizer.encode("Az") == [65, 122]
    assert tokenizer.decode([65, 122]) == "Az"
    assert tokenizer.tokenizer.loaded_path == "tokenizer.model"
