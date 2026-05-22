import torch
from typing import Any, Callable, cast
from transformers import Qwen2Config as TransformersQwen2Config
from transformers import Qwen2ForCausalLM

from ..LLLM.hf_loader import model_ir_from_hf
from ..LLLM.qwen2 import Qwen2Config, Qwen2Model, Qwen2Tokenizer


def _tiny_hf_qwen2_config() -> dict[str, object]:
    return {
        "model_type": "qwen2",
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": True,
    }


def _tiny_qwen2_config() -> Qwen2Config:
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
        "attention_bias": True,
        "dtype": torch.float32,
    }


def test_hf_loader_infers_qwen2_architecture() -> None:
    ir = model_ir_from_hf(_tiny_hf_qwen2_config(), {})

    assert ir.architecture == "qwen2"


def test_qwen2_config_from_ir_translates_hugging_face_names() -> None:
    ir = model_ir_from_hf(_tiny_hf_qwen2_config(), {}, architecture="qwen2")

    assert Qwen2Model.config_from_ir(ir) == _tiny_qwen2_config()


def test_qwen2_tiny_model_matches_transformers_reference_model() -> None:
    manual_seed = cast(Callable[[int], Any], cast(Any, torch).manual_seed)
    reference_config = cast(Any, TransformersQwen2Config)(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
        tie_word_embeddings=False,
    )
    manual_seed(123)
    reference = Qwen2ForCausalLM(reference_config).eval()
    ir = model_ir_from_hf(
        reference_config.to_dict(),
        reference.state_dict(),
        architecture="qwen2",
    )
    model = Qwen2Model(Qwen2Model.config_from_ir(ir)).eval()
    model.load_ir_weights(ir)

    idx = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        actual = model(idx)
        expected = reference(idx).logits

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_qwen2_chat_template_can_disable_thinking() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert prompt == (
        "<|im_start|>user\n"
        "Answer briefly.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def test_qwen2_chat_template_keeps_existing_thinking_default() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == (
        "<|im_start|>user\n"
        "Answer briefly.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
