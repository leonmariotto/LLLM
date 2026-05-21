import torch
from typing import Any, Callable, cast
from transformers import Qwen3Config as TransformersQwen3Config
from transformers import Qwen3ForCausalLM

from ..LLLM.hf_loader import model_ir_from_hf
from ..LLLM.qwen3 import Qwen3Config, Qwen3Model


def _tiny_hf_qwen3_config() -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "head_dim": 4,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
    }


def _tiny_qwen3_config() -> Qwen3Config:
    return {
        "vocab_size": 32,
        "context_length": 16,
        "emb_dim": 16,
        "n_heads": 4,
        "n_kv_groups": 2,
        "n_layers": 2,
        "hidden_dim": 32,
        "head_dim": 4,
        "rope_theta": 10000.0,
        "rope_interleaved": False,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "dtype": torch.float32,
    }


def test_hf_loader_infers_qwen3_architecture() -> None:
    ir = model_ir_from_hf(_tiny_hf_qwen3_config(), {})

    assert ir.architecture == "qwen3"


def test_qwen3_config_from_ir_translates_hugging_face_names() -> None:
    ir = model_ir_from_hf(_tiny_hf_qwen3_config(), {}, architecture="qwen3")

    assert Qwen3Model.config_from_ir(ir) == _tiny_qwen3_config()


def test_qwen3_tiny_model_matches_transformers_reference_model() -> None:
    manual_seed = cast(Callable[[int], Any], cast(Any, torch).manual_seed)
    reference_config = cast(Any, TransformersQwen3Config)(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        head_dim=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    manual_seed(123)
    reference = Qwen3ForCausalLM(reference_config).eval()
    ir = model_ir_from_hf(
        reference_config.to_dict(),
        reference.state_dict(),
        architecture="qwen3",
    )
    model = Qwen3Model(Qwen3Model.config_from_ir(ir)).eval()
    model.load_ir_weights(ir)

    idx = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        actual = model(idx)
        expected = reference(idx).logits

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
