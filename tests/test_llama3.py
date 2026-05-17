import pytest
import torch
from torch import nn

torch = pytest.importorskip("torch")

from ..LLLM.kv_cache import KVCache
from ..LLLM.llama3 import (
    Llama3Config,
    Llama3GroupedQueryAttention,
    Llama3Model,
    Llama3TransformerBlock,
    llama3_config_from_fetched,
)
from ..LLLM.rope import precompute_rope_cache


def _tiny_hf_llama3_config() -> dict[str, object]:
    return {
        "vocab_size": 5,
        "max_position_embeddings": 4,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "rope_theta": 10000.0,
    }


def _tiny_llama3_config() -> Llama3Config:
    return {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 4,
        "n_heads": 2,
        "n_kv_groups": 1,
        "n_layers": 0,
        "hidden_dim": 8,
        "rope_theta": 10000.0,
        "rope_interleaved": False,
        "freq_config": None,
        "dtype": torch.float32,
    }


def _tiny_transformer_llama3_config() -> Llama3Config:
    cfg = _tiny_llama3_config().copy()
    cfg["n_layers"] = 1
    return cfg


def test_llama3_config_from_fetched_translates_hugging_face_names() -> None:
    cfg = llama3_config_from_fetched(_tiny_hf_llama3_config())

    assert cfg == {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 4,
        "n_heads": 2,
        "n_kv_groups": 1,
        "n_layers": 1,
        "hidden_dim": 8,
        "rope_theta": 10000.0,
        "rope_interleaved": False,
        "freq_config": None,
        "dtype": torch.float32,
    }


def test_llama3_config_from_fetched_uses_remote_rope_interleaved() -> None:
    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_interleaved"] = True

    cfg = llama3_config_from_fetched(hf_config)

    assert cfg["rope_interleaved"] is True


def test_llama3_config_from_fetched_translates_hugging_face_rope_scaling() -> None:
    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_scaling"] = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }

    cfg = llama3_config_from_fetched(hf_config)

    assert cfg["freq_config"] == {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
    }


def test_llama3_config_from_fetched_accepts_legacy_rope_scaling_type() -> None:
    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_scaling"] = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "type": "llama3",
    }

    cfg = llama3_config_from_fetched(hf_config)

    assert cfg["freq_config"] == {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
    }


def test_llama3_config_from_fetched_rejects_unsupported_rope_scaling() -> None:
    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_scaling"] = {
        "factor": 2.0,
        "type": "linear",
    }

    with pytest.raises(ValueError, match="unsupported rope_scaling"):
        llama3_config_from_fetched(hf_config)


def test_llama3_config_from_fetched_rejects_bad_rope_scaling_types() -> None:
    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_scaling"] = "llama3"

    with pytest.raises(ValueError, match="rope_scaling"):
        llama3_config_from_fetched(hf_config)

    hf_config = _tiny_hf_llama3_config()
    hf_config["rope_scaling"] = {
        "factor": 8,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }

    with pytest.raises(ValueError, match="factor"):
        llama3_config_from_fetched(hf_config)


def _manual_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    means = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(means + eps)


class AddConstantAttention(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        del cos, sin, pos, kv_cache, layer_idx
        return x + self.value


class ScaleBy(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


def test_grouped_query_attention_matches_manual_causal_attention_without_rope_rotation() -> None:
    attention = Llama3GroupedQueryAttention(
        d_in=4,
        d_out=4,
        context_length=3,
        num_heads=2,
        num_kv_groups=1,
        dropout=0.0,
        qkv_bias=False,
    )

    with torch.no_grad():
        attention.W_query.weight.copy_(torch.eye(4))
        attention.W_key.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
        attention.W_value.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        attention.out_proj.weight.copy_(torch.eye(4))

    inputs = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 2.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 2.0],
            ]
        ]
    )
    cos = torch.ones(3, 1)
    sin = torch.zeros(3, 1)

    outputs = attention(inputs, cos, sin)

    b, num_tokens, _ = inputs.shape
    queries = (inputs @ attention.W_query.weight.T).view(b, num_tokens, 2, 2)
    queries = queries.transpose(1, 2)
    keys = (inputs @ attention.W_key.weight.T).view(b, num_tokens, 1, 2)
    keys = keys.transpose(1, 2).repeat_interleave(2, dim=1)
    values = (inputs @ attention.W_value.weight.T).view(b, num_tokens, 1, 2)
    values = values.transpose(1, 2).repeat_interleave(2, dim=1)
    scores = queries @ keys.transpose(2, 3)
    mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, -torch.inf)
    weights = torch.softmax(scores / 2**0.5, dim=-1)
    expected_outputs = (weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, 4)

    torch.testing.assert_close(outputs, expected_outputs)


def test_grouped_query_attention_with_kv_cache_matches_full_attention_last_token() -> None:
    attention = Llama3GroupedQueryAttention(
        d_in=4,
        d_out=4,
        context_length=4,
        num_heads=2,
        num_kv_groups=1,
        dropout=0.0,
        qkv_bias=False,
    )

    with torch.no_grad():
        attention.W_query.weight.copy_(torch.eye(4))
        attention.W_key.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
        attention.W_value.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        attention.out_proj.weight.copy_(torch.eye(4))

    inputs = torch.tensor([[[1.0, 0.0, 0.5, 0.0], [0.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 2.0]]])
    cos, sin = precompute_rope_cache(seq_len=4, head_dim=2)

    full_outputs = attention(inputs, cos, sin)
    cache = KVCache()
    attention(inputs[:, :2, :], cos, sin, kv_cache=cache, layer_idx=0)
    cached_outputs = attention(inputs[:, 2:, :], cos, sin, kv_cache=cache, layer_idx=0)

    assert cache.layer_seq_len(0) == 3
    torch.testing.assert_close(cached_outputs, full_outputs[:, 2:, :])


def test_grouped_query_attention_rejects_invalid_dimensions() -> None:
    with pytest.raises(AssertionError, match="num_head shall not be 0"):
        Llama3GroupedQueryAttention(
            d_in=4,
            d_out=4,
            context_length=3,
            num_heads=0,
            num_kv_groups=1,
        )

    with pytest.raises(AssertionError, match="d_out must be divisible"):
        Llama3GroupedQueryAttention(
            d_in=4,
            d_out=5,
            context_length=3,
            num_heads=2,
            num_kv_groups=1,
        )

    with pytest.raises(AssertionError, match="num_heads must be divisible"):
        Llama3GroupedQueryAttention(
            d_in=4,
            d_out=4,
            context_length=3,
            num_heads=2,
            num_kv_groups=3,
        )


def test_grouped_query_attention_rejects_invalid_input_embedding_size() -> None:
    attention = Llama3GroupedQueryAttention(
        d_in=4,
        d_out=4,
        context_length=3,
        num_heads=2,
        num_kv_groups=1,
    )
    cos, sin = precompute_rope_cache(seq_len=3, head_dim=2)

    with pytest.raises(AssertionError, match="invalid d_in"):
        attention(torch.empty(1, 3, 3), cos, sin)


def test_grouped_query_attention_rejects_rope_position_beyond_context_length() -> None:
    attention = Llama3GroupedQueryAttention(
        d_in=4,
        d_out=4,
        context_length=3,
        num_heads=2,
        num_kv_groups=1,
    )
    cos, sin = precompute_rope_cache(seq_len=3, head_dim=2)

    with pytest.raises(AssertionError, match="RoPE position exceeds"):
        attention(torch.empty(1, 2, 4), cos, sin, pos=2)


def test_transformer_block_applies_both_residual_paths() -> None:
    cfg = _tiny_transformer_llama3_config()
    block = Llama3TransformerBlock(cfg)
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.att = AddConstantAttention(1.0)
    block.ff = ScaleBy(2.0)

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ]
    )
    cos, sin = precompute_rope_cache(seq_len=cfg["context_length"], head_dim=2)

    outputs = block(inputs, cos, sin)

    expected_outputs = 6 * inputs + 3
    torch.testing.assert_close(outputs, expected_outputs)


def test_llama3_forward_matches_manual_computation_without_transformer_blocks() -> None:
    cfg = _tiny_llama3_config()
    model = Llama3Model(cfg)

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
        model.final_norm.weight.fill_(1.0)
        model.out_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            )
        )

    in_idx = torch.tensor([[0, 1, 2], [2, 1, 0]])

    logits = model(in_idx)

    tok_embeds = model.tok_emb.weight[in_idx]
    expected_logits = _manual_rms_norm(tok_embeds) @ model.out_head.weight.T

    assert logits.shape == (2, 3, cfg["vocab_size"])
    torch.testing.assert_close(logits, expected_logits)


def test_llama3_model_with_transformer_blocks_returns_expected_shape() -> None:
    cfg = _tiny_transformer_llama3_config()
    model = Llama3Model(cfg)
    in_idx = torch.tensor([[0, 1, 2], [2, 1, 0]])

    logits = model(in_idx)

    assert logits.shape == (2, 3, cfg["vocab_size"])


def test_llama3_model_with_kv_cache_matches_full_forward() -> None:
    cfg = _tiny_transformer_llama3_config()
    model = Llama3Model(cfg)
    model.eval()
    in_idx = torch.tensor([[0, 1, 2]])

    with torch.no_grad():
        full_logits = model(in_idx)

        cache = KVCache()
        model(in_idx[:, :2], kv_cache=cache)
        cached_logits = model(in_idx[:, 2:], kv_cache=cache)

    torch.testing.assert_close(cached_logits, full_logits[:, 2:, :])
