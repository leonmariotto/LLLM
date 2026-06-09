from collections.abc import Sequence
from typing import Any

import pytest
from loguru import logger

from ..LLLM.generator import ChatCompletion, CompletionParseError
from ..LLLM.generator_with_tool import (
    AssistantOutput,
    GeneratorWithTool,
    ChatMessage,
)
from ..LLLM.tool_common import Tool, ToolCall


class FakeToolTokenizer:
    def __init__(self, parsed_outputs: dict[str, AssistantOutput | ValueError]) -> None:
        self.parsed_outputs = parsed_outputs
        self.histories: list[list[ChatMessage]] = []
        self.schemas: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]] | str:
        self.histories.append(
            [
                {
                    "role": message["role"],
                    "content": message["content"],
                    **(
                        {"tool_calls": list(message["tool_calls"])}
                        if "tool_calls" in message
                        else {}
                    ),
                }
                for message in messages
            ]
        )
        self.schemas.append(list(tools or []))
        return {"input_ids": [len(self.histories)]}

    def parse_assistant_output(self, completion: str) -> AssistantOutput:
        output = self.parsed_outputs[completion]
        if isinstance(output, ValueError):
            raise output
        return output


class FakeTextGenerator:
    def __init__(
        self,
        outputs: Sequence[str],
        parsed_outputs: dict[str, AssistantOutput | ValueError],
    ) -> None:
        self.outputs = list(outputs)
        self.tokenizer = FakeToolTokenizer(parsed_outputs)
        self.calls: list[dict[str, Any]] = []

    def generate_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> ChatCompletion:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        assert isinstance(encoded, dict)
        input_ids = encoded["input_ids"]
        self.calls.append(
            {
                "prompt_tokens": input_ids,
                "stop_at_eos": stop_at_eos,
                "max_generated_token": max_generated_token,
                "cache_length": cache_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
        raw_completion = self.outputs[len(self.calls) - 1]
        try:
            message = self.tokenizer.parse_assistant_output(raw_completion)
        except ValueError as error:
            raise CompletionParseError(raw_completion, error) from error
        return ChatCompletion(
            message=message,
            raw_completion=raw_completion,
            prompt_tokens=len(input_ids),
            generated_tokens=len(raw_completion),
            finish_reason="stop",
        )


def tool(name: str, execute: Any) -> Tool:
    return Tool(
        schema={
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        },
        execute=execute,
    )


def test_tool_generator_returns_final_response_without_mutating_input() -> None:
    generator = FakeTextGenerator(["done"], {"done": AssistantOutput("finished")})
    tool_generator = GeneratorWithTool(generator, [])
    messages: list[ChatMessage] = [{"role": "user", "content": "question"}]

    result = tool_generator.generate(
        messages,
        max_generated_token=12,
        cache_length=44,
        temperature=0.5,
        top_k=2,
        top_p=0.9,
    )

    assert result == "finished"
    assert messages == [{"role": "user", "content": "question"}]
    assert generator.calls == [
        {
            "prompt_tokens": [1],
            "stop_at_eos": True,
            "max_generated_token": 12,
            "cache_length": 44,
            "temperature": 0.5,
            "top_k": 2,
            "top_p": 0.9,
        }
    ]


def test_tool_generator_executes_multiple_calls_in_order_and_regenerates() -> None:
    seen: list[str] = []
    logs: list[str] = []
    sink_id = logger.add(lambda message: logs.append(str(message)), level="INFO")

    def execute_a(arguments: dict[str, object]) -> str:
        seen.append(f"a:{arguments['value']}")
        return "result-a"

    def execute_b(arguments: dict[str, object]) -> str:
        seen.append(f"b:{arguments['value']}")
        return "result-b"

    generator = FakeTextGenerator(
        ["calls", "answer"],
        {
            "calls": AssistantOutput(
                "",
                (ToolCall("a", {"value": 1}), ToolCall("b", {"value": 2})),
            ),
            "answer": AssistantOutput("complete"),
        },
    )
    tool_generator = GeneratorWithTool(
        generator,
        [tool("a", execute_a), tool("b", execute_b)],
    )

    try:
        result = tool_generator.generate([{"role": "user", "content": "go"}])
    finally:
        logger.remove(sink_id)

    assert result == "complete"
    assert seen == ["a:1", "b:2"]
    text = "".join(logs)
    assert "Assistant requested 2 tool calls on round 0" in text
    assert "Executing tool a" in text
    assert "Tool b completed" in text
    assert "Tool generation completed after 1 tool rounds" in text
    assert generator.tokenizer.histories[1] == [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall("a", {"value": 1}), ToolCall("b", {"value": 2})],
        },
        {"role": "tool", "content": "result-a"},
        {"role": "tool", "content": "result-b"},
    ]


@pytest.mark.parametrize(
    ("first_output", "parsed_output", "expected_error"),
    [
        ("bad", ValueError("invalid JSON"), "invalid tool call output: invalid JSON"),
        (
            "unknown",
            AssistantOutput("", (ToolCall("missing", {}),)),
            "unknown tool 'missing'",
        ),
        (
            "failed",
            AssistantOutput("", (ToolCall("explode", {}),)),
            "'explode' failed: failure",
        ),
    ],
)
def test_tool_generator_feeds_failures_back_for_recovery(
    first_output: str,
    parsed_output: AssistantOutput | ValueError,
    expected_error: str,
) -> None:
    def explode(arguments: dict[str, object]) -> str:
        raise RuntimeError("failure")

    generator = FakeTextGenerator(
        [first_output, "answer"],
        {
            first_output: parsed_output,
            "answer": AssistantOutput("recovered"),
        },
    )
    tool_generator = GeneratorWithTool(generator, [tool("explode", explode)])

    assert tool_generator.generate([{"role": "user", "content": "go"}]) == "recovered"
    if isinstance(parsed_output, ValueError):
        assert generator.tokenizer.histories[1][1]["content"] == first_output
    assert expected_error in generator.tokenizer.histories[1][-1]["content"]


def test_tool_generator_rejects_invalid_and_duplicate_tool_registration() -> None:
    generator = FakeTextGenerator([], {})

    with pytest.raises(ValueError, match="function object"):
        GeneratorWithTool(generator, [Tool({"type": "function"}, lambda _: "")])
    with pytest.raises(ValueError, match="duplicate tool name"):
        GeneratorWithTool(generator, [tool("a", lambda _: ""), tool("a", lambda _: "")])


def test_tool_generator_raises_when_tool_round_limit_is_exhausted() -> None:
    generator = FakeTextGenerator(
        ["call", "call"],
        {"call": AssistantOutput("", (ToolCall("a", {}),))},
    )
    tool_generator = GeneratorWithTool(
        generator,
        [tool("a", lambda _: "again")],
        max_tool_rounds=1,
    )

    with pytest.raises(RuntimeError, match="maximum tool rounds exceeded"):
        tool_generator.generate([{"role": "user", "content": "loop"}])
