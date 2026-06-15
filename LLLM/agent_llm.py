"""
LLM communication layer for the context-aware agent.
"""

from __future__ import annotations

from typing import Literal, cast

from dataclasses import dataclass, field

from .agent_context import ContentItem, Message
from .tool_common import ToolCall
from .agent_context import AgentToolResult
from .generator import (
    Generator,
    ChatMessage,
    CompletionParseError,
    ToolCall as GeneratorToolCall,
)


def _empty_instructions() -> list[str]:
    return []


def _empty_content() -> list[ContentItem]:
    return []


def _empty_tool_schemas() -> list[dict[str, object]]:
    return []


def _empty_usage_metadata() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class LlmRequest:
    """One assistant-turn request."""

    instructions: list[str] = field(default_factory=_empty_instructions)
    content: list[ContentItem] = field(default_factory=_empty_content)
    tool_schemas: list[dict[str, object]] = field(default_factory=_empty_tool_schemas)


@dataclass(frozen=True)
class LlmResponse:
    """One assistant-turn response."""

    content: list[ContentItem] = field(default_factory=_empty_content)
    raw_completion: str = ""
    usage_metadata: dict[str, object] = field(default_factory=_empty_usage_metadata)
    error_message: str | None = None
    # TODO add confidence hint: logprobe


def build_messages(request: LlmRequest) -> list[ChatMessage]:
    """Convert agent history into chat messages accepted by local generators."""
    messages: list[ChatMessage] = [
        cast(ChatMessage, {"role": "system", "content": instruction})
        for instruction in request.instructions
    ]

    for item in request.content:
        if isinstance(item, Message):
            messages.append(
                cast(ChatMessage, {"role": item.role, "content": item.content})
            )
        elif isinstance(item, ToolCall):
            tool_call = GeneratorToolCall(
                name=item.name,
                arguments=dict(item.arguments),
            )
            if messages and messages[-1]["role"] == "assistant":
                # If the last message is an assistant, and the current a ToolCall
                # we shall merge the two.
                # Agent history store messages and tool call as separate Event, but
                # messages API put it in the same message.
                last_message = messages[-1]
                tool_calls = last_message.get("tool_calls")
                if tool_calls is None:
                    tool_calls = []
                    last_message["tool_calls"] = tool_calls
                tool_calls.append(tool_call)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call],
                    }
                )
        else:
            messages.append(
                cast(
                    ChatMessage,
                    {
                        "role": "tool",
                        "content": _format_tool_result(item),
                    },
                )
            )
    return messages


class LlmClient:
    """Adapter from agent requests to ``Generator.generate_completion``."""

    def __init__(
        self,
        generator: Generator,
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        self.generator = generator
        self.stop_at_eos = stop_at_eos
        self.max_generated_token = max_generated_token
        self.cache_length = cache_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate and parse one assistant turn."""
        chat_messages = build_messages(request)
        try:
            completion = self.generator.generate_completion(
                chat_messages,
                tools=request.tool_schemas or None,
                stop_at_eos=self.stop_at_eos,
                max_generated_token=self.max_generated_token,
                cache_length=self.cache_length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        except CompletionParseError as error:
            return LlmResponse(
                content=[Message(role="assistant", content=error.raw_completion)],
                raw_completion=error.raw_completion,
                error_message=str(error),
            )
        except ValueError as error:
            return LlmResponse(error_message=str(error))

        assistant_messages: list[Message] = []
        if completion.message.content:
            assistant_messages.append(
                Message(role="assistant", content=completion.message.content)
            )
        return LlmResponse(
            content=[
                *assistant_messages,
                *[
                    ToolCall(
                        tool_call_id=f"call_{index}",
                        name=tool_call.name,
                        arguments=dict(tool_call.arguments),
                    )
                    for index, tool_call in enumerate(completion.message.tool_calls)
                ],
            ],
            raw_completion=completion.raw_completion,
            usage_metadata={
                "prompt_tokens": completion.prompt_tokens,
                "generated_tokens": completion.generated_tokens,
                "finish_reason": completion.finish_reason,
            },
        )


def _format_tool_result(result: AgentToolResult) -> str:
    prefix_by_status: dict[Literal["success", "error"], str] = {
        "success": "Tool result",
        "error": "Tool error",
    }
    content = "\n".join(str(item) for item in result.content)
    if content:
        return f"{prefix_by_status[result.status]}: {content}"
    return f"{prefix_by_status[result.status]}:"
