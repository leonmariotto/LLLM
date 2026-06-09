import pytest
import torch
from typing import Any, Callable, cast
from transformers import Qwen2Config as TransformersQwen2Config
from transformers import Qwen2ForCausalLM

from ..LLLM.hf_loader import model_ir_from_hf
from ..LLLM.kv_cache import KVCache
from ..LLLM.qwen2 import Qwen2Config, Qwen2Model, Qwen2Tokenizer
from ..LLLM.tool_common import ToolCall


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


def test_qwen2_sliding_cache_uses_absolute_rope_positions() -> None:
    manual_seed = cast(Callable[[int], Any], cast(Any, torch).manual_seed)
    manual_seed(123)
    model = Qwen2Model(_tiny_qwen2_config()).eval()
    idx = torch.tensor([[1, 2, 3, 4]])
    cache = KVCache(cache_length=2)

    with torch.no_grad():
        model(idx[:, :2], kv_cache=cache)
        model(idx[:, 2:3], kv_cache=cache)
        model(idx[:, 3:], kv_cache=cache)

    assert cache.layer_seq_len(0) == 2
    assert cache.layer_start_pos(0) == 2
    assert cache.layer_next_pos(0) == 4


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
        "<|im_start|>user\nAnswer briefly.<|im_end|>\n<|im_start|>assistant\n"
    )


def test_qwen2_chat_template_renders_tools_calls_and_responses() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Read weather.",
                "parameters": {"type": "object"},
            },
        }
    ]

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [ToolCall(name="weather", arguments={"city": "Paris"})],
            },
            {"role": "tool", "content": "warm"},
            {"role": "tool", "content": "dry"},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(prompt, str)
    assert prompt.startswith("<|im_start|>system\nBe precise.\n\n# Tools\n")
    assert '"name": "weather"' in prompt
    assert "<|im_start|>user\nWeather?<|im_end|>\n" in prompt
    assert (
        '<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call><|im_end|>\n" in prompt
    )
    assert (
        "<|im_start|>user\n<tool_response>\nwarm\n</tool_response>"
        "\n<tool_response>\ndry\n</tool_response><|im_end|>\n" in prompt
    )
    assert prompt.endswith("<|im_start|>assistant\n")


def test_qwen2_chat_template_adds_default_system_for_tools() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Use tools."}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tokenize=False,
    )

    assert isinstance(prompt, str)
    assert prompt.startswith(
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
        "# Tools"
    )


def test_qwen2_chat_template_renders_explicit_empty_tool_context() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    prompt = tokenizer.apply_chat_template(
        [{"role": "tool", "content": "Tool error: unknown tool"}],
        tools=[],
        tokenize=False,
    )

    assert isinstance(prompt, str)
    assert "<tools>\n</tools>" in prompt
    assert "<tool_response>\nTool error: unknown tool\n</tool_response>" in prompt


def test_qwen2_parses_assistant_text_and_multiple_tool_calls() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    output = tokenizer.parse_assistant_output(
        "Checking.\n"
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )

    assert output.content == "Checking."
    assert output.tool_calls == (
        ToolCall(name="a", arguments={"x": 1}),
        ToolCall(name="b", arguments={}),
    )


def test_qwen2_parse_assistant_output_strips_thinking_blocks() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    output = tokenizer.parse_assistant_output(
        "<think>hidden reasoning</think>\n"
        "Checking.\n"
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
    )

    assert output.content == "Checking."
    assert output.tool_calls == (ToolCall(name="a", arguments={"x": 1}),)


def test_qwen2_parse_assistant_output_strips_trailing_unclosed_thinking() -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    output = tokenizer.parse_assistant_output("FINAL ANSWER: Paris\n<think>hidden")

    assert output.content == "FINAL ANSWER: Paris"
    assert output.tool_calls == ()


@pytest.mark.parametrize(
    "completion",
    [
        "<tool_call>{broken}</tool_call>",
        '<tool_call>{"name": "lookup", "arguments": []}</tool_call>',
        '<tool_call>{"name": "lookup", "arguments": {}}',
    ],
)
def test_qwen2_rejects_invalid_tool_call_output(completion: str) -> None:
    tokenizer = Qwen2Tokenizer.__new__(Qwen2Tokenizer)

    with pytest.raises(ValueError):
        tokenizer.parse_assistant_output(completion)
