"""
LLM communication layer for the context-aware agent.
"""

from __future__ import annotations

from typing import Literal, cast

from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .agent_context import ContentItem, Message
from .tool_common import ToolCall
from .agent_context import AgentToolResult
from .generator import (
    Generator,
    ChatMessage,
    CompletionParseError,
    ToolCall as GeneratorToolCall,
)


def _empty_content() -> list[ContentItem]:
    return []


def _empty_tool_schemas() -> list[dict[str, object]]:
    return []


def _empty_usage_metadata() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class LlmRequest:
    """One assistant-turn request."""

    content: list[ContentItem] = field(default_factory=_empty_content)
    tool_schemas: list[dict[str, object]] = field(default_factory=_empty_tool_schemas)
    response_format: type[BaseModel] | None = None
    trace_enabled: bool = False


@dataclass(frozen=True)
class LlmResponse:
    """One assistant-turn response."""

    content: list[ContentItem] = field(default_factory=_empty_content)
    raw_completion: str = ""
    usage_metadata: dict[str, object] = field(default_factory=_empty_usage_metadata)
    error_message: str | None = None
    parsed: BaseModel | None = None
    trace: dict[str, object] | None = None
    # TODO add confidence hint: logprobe


def build_messages(request: LlmRequest) -> list[ChatMessage]:
    """Convert agent history into chat messages accepted by local generators."""
    messages: list[ChatMessage] = []

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
        enable_thinking: bool = True,
    ) -> None:
        self.generator = generator
        self.stop_at_eos = stop_at_eos
        self.max_generated_token = max_generated_token
        self.cache_length = cache_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_thinking = enable_thinking

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate and parse one assistant turn."""
        chat_messages = build_messages(request)
        request_trace = (
            self._request_trace(chat_messages, request)
            if request.trace_enabled
            else None
        )
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
                enable_thinking=self.enable_thinking,
                response_format=request.response_format,
                trace_enabled=request.trace_enabled,
            )
        except CompletionParseError as error:
            trace = None
            if request.trace_enabled:
                trace = {
                    **(request_trace or {}),
                    "completion": error.trace,
                    "error": {
                        "type": type(error.parse_error).__name__,
                        "message": str(error),
                    },
                }
            return LlmResponse(
                content=[Message(role="assistant", content=error.raw_completion)],
                raw_completion=error.raw_completion,
                error_message=str(error),
                trace=trace,
            )
        except ValueError as error:
            trace = None
            if request.trace_enabled:
                trace = {
                    **(request_trace or {}),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            return LlmResponse(error_message=str(error), trace=trace)

        assistant_messages: list[ContentItem] = []
        if completion.message.content:
            assistant_messages.append(
                Message(role="assistant", content=completion.message.content)
            )
        parsed: BaseModel | None = None
        # parse the returned string, if typed sctructured answer parsed successfuly
        # it's returned in the parsed field of LlmResponse.
        if request.response_format is not None:
            payload = _structured_json_payload(
                completion.message.content,
                completion.raw_completion,
            )
            try:
                parsed = request.response_format.model_validate_json(payload)
            except ValidationError as error:
                trace = None
                if request.trace_enabled:
                    trace = self._response_trace(
                        request_trace,
                        completion,
                        assistant_messages,
                        parsed=None,
                        error=error,
                    )
                return LlmResponse(
                    content=assistant_messages,
                    raw_completion=completion.raw_completion,
                    usage_metadata={
                        "prompt_tokens": completion.prompt_tokens,
                        "generated_tokens": completion.generated_tokens,
                        "finish_reason": completion.finish_reason,
                    },
                    error_message=str(error),
                    trace=trace,
                )
        content: list[ContentItem] = [
            *assistant_messages,
            *[
                ToolCall(
                    tool_call_id=f"call_{index}",
                    name=tool_call.name,
                    arguments=dict(tool_call.arguments),
                )
                for index, tool_call in enumerate(completion.message.tool_calls)
            ],
        ]
        trace = None
        if request.trace_enabled:
            trace = self._response_trace(
                request_trace,
                completion,
                content,
                parsed=parsed,
                error=None,
            )
        return LlmResponse(
            content=content,
            raw_completion=completion.raw_completion,
            usage_metadata={
                "prompt_tokens": completion.prompt_tokens,
                "generated_tokens": completion.generated_tokens,
                "finish_reason": completion.finish_reason,
            },
            parsed=parsed,
            trace=trace,
        )

    def count_tokens(self, request: LlmRequest) -> int:
        """Count prompt tokens for one request using the generator chat template."""
        chat_messages = build_messages(request)
        return self.generator.count_completion_tokens(
            chat_messages,
            tools=request.tool_schemas or None,
            enable_thinking=self.enable_thinking,
        )

    def _request_trace(
        self,
        chat_messages: list[ChatMessage],
        request: LlmRequest,
    ) -> dict[str, object]:
        """
        Build the trace request: that will be logged in a json file if trace enabled.
        """
        return {
            "request": {
                "messages": chat_messages,
                "tool_schemas": request.tool_schemas,
                "response_format": (
                    request.response_format.__name__
                    if request.response_format is not None
                    else None
                ),
            },
            "client_config": {
                "stop_at_eos": self.stop_at_eos,
                "max_generated_token": self.max_generated_token,
                "cache_length": self.cache_length,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "enable_thinking": self.enable_thinking,
            },
        }

    def _response_trace(
        self,
        request_trace: dict[str, object] | None,
        completion: object,
        content: list[ContentItem],
        *,
        parsed: BaseModel | None,
        error: Exception | None,
    ) -> dict[str, object]:
        """
        Build the trace response: that will be logged in a json file if trace enabled.
        """
        completion_trace = getattr(completion, "trace", None)
        trace: dict[str, object] = {
            **(request_trace or {}),
            "completion": completion_trace,
            "parsed_content": [item.model_dump(mode="json") for item in content],
            "parsed_structured_response": (
                parsed.model_dump(mode="json") if parsed is not None else None
            ),
        }
        if error is not None:
            trace["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        return trace


def _format_tool_result(result: AgentToolResult) -> str:
    prefix_by_status: dict[Literal["success", "error"], str] = {
        "success": "Tool result",
        "error": "Tool error",
    }
    content = "\n".join(str(item) for item in result.content)
    if content:
        return f"{prefix_by_status[result.status]}: {content}"
    return f"{prefix_by_status[result.status]}:"


def _structured_json_payload(content: str, raw_completion: str) -> str:
    """Return JSON content from plain or optional-think structured output."""
    for candidate in (content, raw_completion):
        payload = _strip_optional_think(candidate).strip()
        if payload:
            return payload
    return ""


def _strip_optional_think(text: str) -> str:
    close_tag = "</think>"
    close_index = text.find(close_tag)
    if text.lstrip().startswith("<think>") and close_index != -1:
        return text[close_index + len(close_tag) :]
    return text
