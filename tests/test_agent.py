from collections.abc import Sequence

import pytest

from ..LLLM.agent import Agent
from ..LLLM.agent_context import ExecutionContext, Message, AgentToolCall, ToolResult
from ..LLLM.agent_llm import LlmClient
from ..LLLM.generator import (
    AssistantOutput,
    ChatCompletion,
    ChatMessage,
    ToolCall as GeneratorToolCall,
)
from ..LLLM.tool_common import Tool


class FakeGenerator:
    def __init__(self, outputs: Sequence[AssistantOutput]) -> None:
        self.outputs = list(outputs)
        self.messages: list[list[ChatMessage]] = []
        self.tool_schemas: list[list[dict[str, object]]] = []

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
        enable_thinking: bool = True,
    ) -> ChatCompletion:
        self.messages.append([dict(message) for message in messages])
        self.tool_schemas.append(list(tools or []))
        output = self.outputs[len(self.messages) - 1]
        return ChatCompletion(
            message=output,
            raw_completion="raw",
            prompt_tokens=3,
            generated_tokens=4,
            finish_reason="stop",
        )


def tool(name: str, result: str | Exception) -> Tool:
    def execute(arguments: dict[str, object]) -> str:
        if isinstance(result, Exception):
            raise result
        return f"{result}:{arguments.get('q', '')}"

    return Tool(
        schema={
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        },
        execute=execute,
    )


def test_agent_run_returns_simple_answer_and_updates_context() -> None:
    context = ExecutionContext()
    generator = FakeGenerator([AssistantOutput("finished")])
    llm = LlmClient(generator, max_generated_token=11, temperature=0.2)
    agent = Agent(llm, [], instructions="be brief")

    assert agent.run("question", context=context) == "finished"

    assert context.final_result == "finished"
    assert context.messages() == [
        Message(role="user", content="question"),
        Message(role="assistant", content="finished"),
    ]
    assert generator.messages == [
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "question"},
        ]
    ]


def test_agent_run_executes_successful_tool_round_then_final_answer() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput(
                "checking",
                (GeneratorToolCall("lookup", {"q": "x"}),),
            ),
            AssistantOutput("answer"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    assert agent.run("question") == "answer"

    assert generator.messages[1] == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [GeneratorToolCall("lookup", {"q": "x"})],
        },
        {"role": "tool", "content": "Tool result: found:x"},
    ]


@pytest.mark.parametrize(
    ("first_call", "registered_tools", "expected_tool_message"),
    [
        (
            GeneratorToolCall("missing", {}),
            [],
            "Tool error: unknown tool 'missing'",
        ),
        (
            GeneratorToolCall("explode", {}),
            [tool("explode", RuntimeError("failure"))],
            "Tool error: 'explode' failed: failure",
        ),
    ],
)
def test_agent_feeds_unknown_tool_and_exceptions_back_for_recovery(
    first_call: GeneratorToolCall,
    registered_tools: list[Tool],
    expected_tool_message: str,
) -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (first_call,)),
            AssistantOutput("recovered"),
        ]
    )
    agent = Agent(LlmClient(generator), registered_tools)

    assert agent.run("go") == "recovered"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": expected_tool_message,
    }


def test_agent_raises_when_tool_round_limit_is_exhausted() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (GeneratorToolCall("again", {}),)),
            AssistantOutput("", (GeneratorToolCall("again", {}),)),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("again", "loop")], max_tool_rounds=1)

    with pytest.raises(RuntimeError, match="maximum tool rounds exceeded"):
        agent.run("loop")


def test_agent_context_records_tool_events_and_final_result() -> None:
    context = ExecutionContext()
    generator = FakeGenerator(
        [
            AssistantOutput("", (GeneratorToolCall("lookup", {"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    assert agent.run("go", context=context) == "done"

    assert context.items() == [
        Message(role="user", content="go"),
        AgentToolCall(tool_call_id="call_0_0", name="lookup", arguments={"q": "x"}),
        ToolResult(
            tool_call_id="call_0_0",
            name="lookup",
            status="success",
            content=["found:x"],
        ),
        Message(role="assistant", content="done"),
    ]
    assert context.final_result == "done"
