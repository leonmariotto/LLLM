from collections.abc import Sequence

from ..LLLM.agent_context import Message, AgentToolCall, AgentToolResult
from ..LLLM.agent_llm import LlmClient, LlmRequest, build_messages
from ..LLLM.generator import (
    AssistantOutput,
    ChatCompletion,
    ChatMessage,
    CompletionParseError,
    ToolCall as GeneratorToolCall,
)


class FakeGenerator:
    def __init__(self, outputs: Sequence[AssistantOutput | ValueError]) -> None:
        self.outputs = list(outputs)
        self.messages: list[list[ChatMessage]] = []
        self.tool_schemas: list[list[dict[str, object]]] = []
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "stop_at_eos": stop_at_eos,
                "max_generated_token": max_generated_token,
                "cache_length": cache_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
            }
        )
        output = self.outputs[len(self.messages) - 1]
        if isinstance(output, ValueError):
            raise CompletionParseError("bad raw", output)
        return ChatCompletion(
            message=output,
            raw_completion="raw",
            prompt_tokens=3,
            generated_tokens=4,
            finish_reason="stop",
        )


def test_build_messages_converts_context_items() -> None:
    request = LlmRequest(
        instructions=["be useful"],
        content=[
            Message(role="user", content="question"),
            Message(role="assistant", content="checking"),
            AgentToolCall(tool_call_id="call_2_0", name="lookup", arguments={"q": "x"}),
            AgentToolResult(
                tool_call_id="call_2_0",
                name="lookup",
                status="success",
                content=["found"],
            ),
        ],
    )

    assert build_messages(request) == [
        {"role": "system", "content": "be useful"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [GeneratorToolCall("lookup", {"q": "x"})],
        },
        {"role": "tool", "content": "Tool result: found"},
    ]


def test_llm_client_complete_returns_text_tool_calls_and_usage() -> None:
    schema: dict[str, object] = {"type": "function"}
    generator = FakeGenerator(
        [
            AssistantOutput(
                "I will check",
                (GeneratorToolCall("lookup", {"q": "x"}),),
            )
        ]
    )

    response = LlmClient(generator, max_generated_token=9).complete(
        LlmRequest(
            content=[Message(role="user", content="question")],
            tool_schemas=[schema],
        )
    )

    assert response.error_message is None
    assert response.raw_completion == "raw"
    assert response.usage_metadata == {
        "prompt_tokens": 3,
        "generated_tokens": 4,
        "finish_reason": "stop",
    }
    assert response.content == [
        Message(role="assistant", content="I will check"),
        AgentToolCall(tool_call_id="call_0", name="lookup", arguments={"q": "x"}),
    ]
    assert generator.messages == [[{"role": "user", "content": "question"}]]
    assert generator.tool_schemas == [[schema]]
    assert generator.calls[0]["max_generated_token"] == 9


def test_llm_client_preserves_parse_error_raw_completion() -> None:
    response = LlmClient(FakeGenerator([ValueError("invalid")])).complete(LlmRequest())

    assert response.error_message == "invalid"
    assert response.raw_completion == "bad raw"
    assert response.content == [Message(role="assistant", content="bad raw")]
