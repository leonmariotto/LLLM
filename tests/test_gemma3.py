from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from ..LLLM.fetch import FetchedModel
from ..LLLM.gemma3 import (
    Gemma3Config,
    Gemma3GroupedQueryAttention,
    Gemma3Model,
    Gemma3TransformerBlock,
    gemma3_config_from_fetched,
)
from ..LLLM.rope import precompute_rope_cache


def _tiny_hf_gemma3_config() -> dict[str, object]:
    return {
        "model_type": "gemma3_text",
        "vocab_size": 13,
        "max_position_embeddings": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "head_dim": 4,
        "query_pre_attn_scalar": 4,
        "sliding_window": 4,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "final_logit_softcapping": None,
        "attn_logit_softcapping": None,
    }


def _tiny_gemma3_config() -> Gemma3Config:
    return gemma3_config_from_fetched(_tiny_hf_gemma3_config())


def _tiny_transformers_gemma3_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        query_pre_attn_scalar=4,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        attention_bias=False,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )


def test_gemma3_config_from_fetched_translates_hugging_face_names() -> None:
    cfg = _tiny_gemma3_config()

    assert cfg["vocab_size"] == 13
    assert cfg["emb_dim"] == 8
    assert cfg["hidden_dim"] == 16
    assert cfg["n_heads"] == 2
    assert cfg["n_kv_groups"] == 1
    assert cfg["head_dim"] == 4
    assert cfg["query_pre_attn_scalar"] == 4
    assert cfg["layer_types"] == ["sliding_attention", "full_attention"]


def test_gemma3_config_from_fetched_accepts_nested_text_config() -> None:
    hf_config = {
        "model_type": "gemma3",
        "text_config": {
            **_tiny_hf_gemma3_config(),
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            },
        },
    }

    cfg = gemma3_config_from_fetched(hf_config)

    assert cfg["rope_base"] == 1000000.0
    assert cfg["rope_local_base"] == 10000.0


def test_gemma3_attention_supports_head_dim_different_from_embedding_split() -> None:
    attention = Gemma3GroupedQueryAttention(
        d_in=8,
        d_out=8,
        head_dim=6,
        context_length=8,
        num_heads=2,
        num_kv_groups=1,
        sliding_window=None,
        query_pre_attn_scalar=6,
    )
    cos, sin = precompute_rope_cache(seq_len=8, head_dim=6)
    x = torch.randn(2, 3, 8)

    out = attention(x, cos, sin)

    assert out.shape == (2, 3, 8)


def test_gemma3_load_fetched_model_copies_hugging_face_weights() -> None:
    cfg = _tiny_gemma3_config()
    model = Gemma3Model(cfg)
    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(13, 8),
        "model.norm.weight": torch.randn(8),
        "lm_head.weight": torch.randn(13, 8),
    }
    for layer_idx in range(cfg["n_layers"]):
        weights.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight": torch.randn(
                    8, 8
                ),
                f"model.layers.{layer_idx}.self_attn.k_proj.weight": torch.randn(
                    4, 8
                ),
                f"model.layers.{layer_idx}.self_attn.v_proj.weight": torch.randn(
                    4, 8
                ),
                f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(
                    8, 8
                ),
                f"model.layers.{layer_idx}.self_attn.q_norm.weight": torch.randn(4),
                f"model.layers.{layer_idx}.self_attn.k_norm.weight": torch.randn(4),
                f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(8),
                f"model.layers.{layer_idx}.post_attention_layernorm.weight": (
                    torch.randn(8)
                ),
                f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": (
                    torch.randn(8)
                ),
                f"model.layers.{layer_idx}.post_feedforward_layernorm.weight": (
                    torch.randn(8)
                ),
                f"model.layers.{layer_idx}.mlp.gate_proj.weight": torch.randn(16, 8),
                f"model.layers.{layer_idx}.mlp.up_proj.weight": torch.randn(16, 8),
                f"model.layers.{layer_idx}.mlp.down_proj.weight": torch.randn(8, 16),
            }
        )
    fetched = FetchedModel(
        path=Path("/tmp/fake-gemma3"),
        config=_tiny_hf_gemma3_config(),
        weights=weights,
    )

    model.load_fetched_model(fetched)

    torch.testing.assert_close(model.tok_emb.weight, weights["model.embed_tokens.weight"])
    torch.testing.assert_close(model.final_norm.scale, weights["model.norm.weight"])
    torch.testing.assert_close(model.out_head.weight, weights["lm_head.weight"])
    first_block = cast(Gemma3TransformerBlock, model.trf_blocks[0])
    torch.testing.assert_close(
        first_block.att.W_query.weight,
        weights["model.layers.0.self_attn.q_proj.weight"],
    )


def test_gemma3_load_fetched_model_rejects_bad_shapes() -> None:
    cfg = _tiny_gemma3_config()
    model = Gemma3Model(cfg)
    fetched = FetchedModel(
        path=Path("/tmp/fake-gemma3"),
        config=_tiny_hf_gemma3_config(),
        weights={"model.embed_tokens.weight": torch.randn(12, 8)},
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_fetched_model(fetched)


def test_gemma3_tiny_model_matches_transformers_reference_model() -> None:
    cast(Any, torch).manual_seed(1234)
    reference_config = _tiny_transformers_gemma3_config()
    reference = Gemma3ForCausalLM(reference_config)
    reference.eval()
    fetched = FetchedModel(
        path=Path("."),
        config=reference_config.to_dict(),
        weights=reference.state_dict(),
    )
    model = Gemma3Model(gemma3_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)

    input_ids = torch.tensor([[2, 4, 6, 8, 10]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        reference_logits = reference(input_ids=input_ids).logits

    assert logits.shape == reference_logits.shape
    torch.testing.assert_close(logits, reference_logits, rtol=1e-5, atol=1e-6)
