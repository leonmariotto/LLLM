from collections.abc import Sequence

import pytest

from ..LLLM.agent import (
    Agent,
    SUMMARY_PROMPT,
    SUM_KEEP_RECENTS,
    SUMMARIZE_TOKEN_THRESHOLD,
)
from ..LLLM.agent_context import ExecutionContext, Event, Message, AgentToolResult
from ..LLLM.agent_llm import LlmClient
from ..LLLM.generator import (
    AssistantOutput,
    ChatCompletion,
    ChatMessage,
)
from ..LLLM.tool_common import Tool, ToolCall, ToolContextPolicy


class FakeTokenizer:
    def __init__(self, token_count: int = 0) -> None:
        self.token_count = token_count
        self.messages: list[list[ChatMessage]] = []
        self.tools: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> dict[str, list[int]]:
        self.messages.append([dict(message) for message in messages])
        self.tools.append(list(tools or []))
        return {"input_ids": list(range(self.token_count))}

    def parse_assistant_output(self, output: str) -> AssistantOutput:
        return AssistantOutput(output)


class FakeGenerator:
    def __init__(
        self,
        outputs: Sequence[AssistantOutput | Exception],
        *,
        token_count: int = 0,
    ) -> None:
        self.outputs = list(outputs)
        self.messages: list[list[ChatMessage]] = []
        self.tool_schemas: list[list[dict[str, object]]] = []
        self.tokenizer = FakeTokenizer(token_count)

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
        if isinstance(output, Exception):
            raise output
        return ChatCompletion(
            message=output,
            raw_completion="raw",
            prompt_tokens=3,
            generated_tokens=4,
            finish_reason="stop",
        )

    def count_completion_tokens(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        enable_thinking: bool = True,
    ) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return len(encoded["input_ids"])


def tool(name: str, result: str | Exception) -> Tool:
    def execute(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        del container_env
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


def long_message_context() -> ExecutionContext:
    context = ExecutionContext()
    for index in range(SUM_KEEP_RECENTS + 3):
        message = (
            Message(role="user", content=f"item {index}")
            if index % 2 == 0
            else Message(role="assistant", content=f"item {index}")
        )
        context.add_event(
            Event(
                execution_id=context.execution_id,
                author=message.role,
                content=[message],
            )
        )
    return context


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


def test_agent_run_forwards_container_env_to_tools() -> None:
    class FakeContainerEnv:
        pass

    env = FakeContainerEnv()
    seen_envs: list[object] = []

    def execute(
        arguments: dict[str, object],
        container_env: object | None = None,
    ) -> str:
        seen_envs.append(container_env)
        return f"ok:{arguments['q']}"

    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="lookup", arguments={"q": "x"}),)),
            AssistantOutput("done"),
        ]
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
                execute=execute,
            )
        ],
    )

    result = agent.run("go", container_env=env)  # type: ignore[arg-type]

    assert result.output == "done"
    assert seen_envs == [env]
    assert generator.messages[1][-1] == {
        "role": "tool",
        "content": "Tool result: ok:x",
    }


def test_agent_does_not_manage_container_env_lifecycle() -> None:
    class FakeContainerEnv:
        def __init__(self) -> None:
            self.started = False
            self.closed = False

        def start(self, *_args: object, **_kwargs: object) -> None:
            self.started = True

        def close(self) -> None:
            self.closed = True

    env = FakeContainerEnv()
    generator = FakeGenerator([AssistantOutput("done")])
    agent = Agent(LlmClient(generator), [])

    result = agent.run("go", container_env=env)  # type: ignore[arg-type]

    assert result.output == "done"
    assert env.started is False
    assert env.closed is False


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
                execute=lambda _, container_env=None: raw_answer,
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


def test_agent_does_not_summarize_short_history() -> None:
    context = ExecutionContext()
    for index in range(SUM_KEEP_RECENTS + 1):
        context.add_user_message(f"item {index}")
    generator = FakeGenerator([AssistantOutput("done")])
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 1
    assert generator.messages[0] == [
        {"role": "user", "content": f"item {index}"}
        for index in range(SUM_KEEP_RECENTS + 1)
    ]


def test_agent_does_not_summarize_long_history_below_token_threshold() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [AssistantOutput("done")],
        token_count=SUMMARIZE_TOKEN_THRESHOLD,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 1
    assert generator.messages[0] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]


def test_agent_summarizes_middle_history_only_in_llm_request() -> None:
    context = long_message_context()
    raw_items = context.items()
    generator = FakeGenerator(
        [
            AssistantOutput("summary text"),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.tool_schemas == [[], []]
    assert generator.messages[0] == [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
    ]
    assert generator.messages[1] == [
        {"role": "user", "content": "item 0"},
        {
            "role": "system",
            "content": "Conversation summary so far:\nsummary text",
        },
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.items() == [*raw_items, Message(role="assistant", content="done")]


def test_agent_falls_back_to_unsummarized_history_on_summary_error() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [
            ValueError("summary failed"),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.messages[1] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.messages()[-1] == Message(role="assistant", content="done")


def test_agent_falls_back_to_unsummarized_history_without_assistant_summary() -> None:
    context = long_message_context()
    generator = FakeGenerator(
        [
            AssistantOutput("", (ToolCall(name="noop", arguments={}),)),
            AssistantOutput("done"),
        ],
        token_count=SUMMARIZE_TOKEN_THRESHOLD + 1,
    )
    agent = Agent(LlmClient(generator), [])

    agent.step(context)

    assert len(generator.messages) == 2
    assert generator.messages[1] == [
        {"role": "user", "content": "item 0"},
        {"role": "assistant", "content": "item 1"},
        {"role": "user", "content": "item 2"},
        {"role": "assistant", "content": "item 3"},
        {"role": "user", "content": "item 4"},
        {"role": "assistant", "content": "item 5"},
        {"role": "user", "content": "item 6"},
        {"role": "assistant", "content": "item 7"},
    ]
    assert context.messages()[-1] == Message(role="assistant", content="done")


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
                execute=lambda _, container_env=None: "unused",
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
                execute=lambda _, container_env=None: "answer",
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
                execute=lambda _, container_env=None: "raw answer",
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
