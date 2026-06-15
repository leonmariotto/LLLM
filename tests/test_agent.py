from collections.abc import Sequence

import pytest

from ..LLLM.agent import Agent
from ..LLLM.agent_context import ExecutionContext, Event, Message, AgentToolResult
from ..LLLM.agent_llm import LlmClient
from ..LLLM.generator import (
    AssistantOutput,
    ChatCompletion,
    ChatMessage,
)
from ..LLLM.tool_common import Tool, ToolCall, ToolContextPolicy


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
    agent = Agent(llm, [], instruction="be brief")

    result = agent.run("question", context=context)

    assert result.output == "finished"
    assert result.context is context
    assert context.final_result == "finished"
    assert context.messages() == [
        Message(role="user", content="question"),
        Message(role="assistant", content="finished"),
    ]
    assert generator.messages == [
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "question"},
        ],
    ]


def test_agent_run_executes_successful_tool_round_then_final_answer() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput(
                "checking",
                (ToolCall(name="lookup", arguments={"q": "x"}),),
            ),
            AssistantOutput("answer"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    result = agent.run("question")

    assert result.output == "answer"

    assert generator.messages[1] == [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [ToolCall(name="lookup", arguments={"q": "x"})],
        },
        {"role": "tool", "content": "Tool result: found:x"},
    ]


@pytest.mark.parametrize(
    ("first_call", "registered_tools", "expected_tool_message"),
    [
        (
            ToolCall(name="missing", arguments={}),
            [],
            "Tool error: unknown tool 'missing'",
        ),
        (
            ToolCall(name="explode", arguments={}),
            [tool("explode", RuntimeError("failure"))],
            "Tool error: 'explode' failed: failure",
        ),
    ],
)
def test_agent_feeds_unknown_tool_and_exceptions_back_for_recovery(
    first_call: ToolCall,
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

    result = agent.run("go")

    assert result.output == "recovered"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": expected_tool_message,
    }


def test_agent_returns_without_final_result_when_tool_round_limit_is_exhausted() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="again", arguments={}),)),
            AssistantOutput("", (ToolCall(name="again", arguments={}),)),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("again", "loop")], max_step=1)

    result = agent.run("loop")

    assert result.output is None
    assert result.context.final_result is None
    assert result.context.current_step == 1


def test_agent_context_records_tool_events_and_final_result() -> None:
    context = ExecutionContext()
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )
    agent = Agent(LlmClient(generator), [tool("lookup", "found")])

    result = agent.run("go", context=context)

    assert result.output == "done"
    assert result.context is context

    assert context.items() == [
        Message(role="user", content="go"),
        ToolCall(tool_call_id="call_0", name="lookup", arguments={"q": "x"}),
        AgentToolResult(
            tool_call_id="call_0",
            name="lookup",
            status="success",
            content=["found:x"],
        ),
        Message(role="assistant", content="done"),
    ]
    assert context.final_result == "done"


def test_agent_keeps_latest_tool_answer_raw_in_llm_request() -> None:
    raw_answer = "raw answer with lots of detail"
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_answer(result: AgentToolResult) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content=["compact answer"],
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _: raw_answer,
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": f"Tool result: {raw_answer}",
    }
    assert result.context.items()[2] == AgentToolResult(
        tool_call_id="call_0",
        name="lookup",
        status="success",
        content=[raw_answer],
    )


def test_agent_compacts_previous_tool_answer_only_in_llm_request() -> None:
    context = ExecutionContext()
    context.add_user_message("go")
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="agent",
            content=[
                ToolCall(
                    tool_call_id="call_0",
                    name="lookup",
                    arguments={"q": "x"},
                )
            ],
        )
    )
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="tool",
            content=[
                AgentToolResult(
                    tool_call_id="call_0",
                    name="lookup",
                    status="success",
                    content=["raw answer with lots of detail"],
                )
            ],
        )
    )
    context.add_user_message("continue")
    generator = FakeGenerator([AssistantOutput("done")])

    def compact_answer(result: AgentToolResult) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content=["compact answer"],
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _: "unused",
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    agent.step(context)

    assert generator.messages[0][-2] == {
        "role": "tool",
        "content": "Tool result: compact answer",
    }
    assert generator.messages[0][-1] == {"role": "user", "content": "continue"}
    assert context.items()[2] == AgentToolResult(
        tool_call_id="call_0",
        name="lookup",
        status="success",
        content=["raw answer with lots of detail"],
    )


def test_agent_compacts_tool_call_only_in_llm_request() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "raw"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_call(call: ToolCall) -> ToolCall:
        return ToolCall(
            tool_call_id=call.tool_call_id,
            name=call.name,
            arguments={"q": "compact"},
        )

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _: "answer",
                context_policy=ToolContextPolicy(compact_call=compact_call),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assistant_message = generator.messages[1][1]
    assert assistant_message["tool_calls"] == [
        ToolCall(name="lookup", arguments={"q": "compact"})
    ]
    assert result.context.items()[1] == ToolCall(
        tool_call_id="call_0",
        name="lookup",
        arguments={"q": "raw"},
    )


def test_agent_uses_raw_item_when_context_policy_fails() -> None:
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
    )

    def compact_answer(_: AgentToolResult) -> AgentToolResult:
        raise RuntimeError("bad policy")

    agent = Agent(
        LlmClient(generator),
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
                execute=lambda _: "raw answer",
                context_policy=ToolContextPolicy(compact_answer=compact_answer),
            )
        ],
    )

    result = agent.run("go")

    assert result.output == "done"
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": "Tool result: raw answer",
    }
