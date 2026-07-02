"""
Small OpenAI-compatible HTTP server for local LLLM inference.
Provide a command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
from threading import Lock
import time
from typing import Annotated, Literal, Protocol, Self, cast
from uuid import uuid4

import click
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import uvicorn

from .fetch import fetch_model_ir
from .generator import Generator
from .generator import ChatCompletion as LocalChatCompletion
from .generator import ChatMessage as LocalChatMessage
from .model_ir import ModelIR
from .tool_common import ToolCall as LocalToolCall
from .utils import get_device


DEFAULT_CACHE_LENGTH = 16384
DEFAULT_SERVER_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_SERVED_MODEL_NAME = "lllm"
SERVER_LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


class CompletionGenerator(Protocol):
    """Generator operations used by the HTTP adapter."""

    def generate_completion(
        self,
        messages: Sequence[LocalChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        max_generated_token: int = 20,
        temperature: float = 0.0,
        top_p: float | None = None,
        enable_thinking: bool = True,
    ) -> LocalChatCompletion: ...


def _tokenizer_artifact_path(ir: ModelIR) -> Path:
    path = Path(str(ir.metadata["path"]))
    return path if path.suffix.lower() == ".gguf" else path / "tokenizer.json"


def _build_qwen3_generator(ir: ModelIR, *, cache_length: int) -> Generator:
    from .qwen3 import Qwen3Model, Qwen3Tokenizer

    model = Qwen3Model(Qwen3Model.config_from_ir(ir))
    tokenizer = Qwen3Tokenizer(str(_tokenizer_artifact_path(ir)))
    model.load_ir_weights(ir)
    model.to(get_device())
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_gemma3_generator(ir: ModelIR, *, cache_length: int) -> Generator:
    from .gemma3 import Gemma3Model, Gemma3Tokenizer

    model = Gemma3Model(Gemma3Model.config_from_ir(ir))
    tokenizer = Gemma3Tokenizer(str(_tokenizer_artifact_path(ir)))
    model.load_ir_weights(ir)
    model.to(get_device())
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def build_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
) -> CompletionGenerator:
    """Load a supported model architecture into the shared generator."""
    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    if ir.architecture == "qwen3":
        return _build_qwen3_generator(ir, cache_length=cache_length)
    if ir.architecture == "gemma3":
        return _build_gemma3_generator(ir, cache_length=cache_length)
    raise ValueError(
        f"unsupported inference server architecture {ir.architecture!r}; "
        "supported architectures: qwen3, gemma3"
    )


def _configure_logging(verbosity: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextMessage(_StrictModel):
    role: Literal["system", "developer", "user"]
    content: str


class FunctionCall(_StrictModel):
    name: str = Field(min_length=1, max_length=64)
    arguments: str

    @field_validator("arguments")
    @classmethod
    def arguments_must_be_json_object(cls, value: str) -> str:
        try:
            arguments = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("function arguments must be valid JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("function arguments must be a JSON object")
        return value


class FunctionToolCall(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


class AssistantInputMessage(_StrictModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[FunctionToolCall] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> Self:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class ToolInputMessage(_StrictModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str = Field(min_length=1)


ChatCompletionMessage = Annotated[
    TextMessage | AssistantInputMessage | ToolInputMessage,
    Field(discriminator="role"),
]


class FunctionDefinition(_StrictModel):
    name: str = Field(min_length=1, max_length=64)
    description: str | None = None
    parameters: dict[str, object] | None = None
    strict: bool | None = None


class FunctionTool(_StrictModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedFunctionChoice(_StrictModel):
    name: str = Field(min_length=1, max_length=64)


class NamedFunctionToolChoice(_StrictModel):
    type: Literal["function"] = "function"
    function: NamedFunctionChoice


class ChatCompletionRequest(_StrictModel):
    model: str
    messages: list[ChatCompletionMessage] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedFunctionToolChoice | None = (
        None
    )
    parallel_tool_calls: bool = True


class AssistantMessage(_StrictModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[FunctionToolCall] | None = None


class ChatCompletionChoice(_StrictModel):
    index: int
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"]
    logprobs: None = None


class CompletionUsage(_StrictModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(_StrictModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


def _local_messages(
    messages: Sequence[ChatCompletionMessage],
) -> list[LocalChatMessage]:
    local_messages: list[LocalChatMessage] = []
    for message in messages:
        if isinstance(message, AssistantInputMessage):
            local_calls = [
                LocalToolCall(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=cast(
                        dict[str, object],
                        json.loads(tool_call.function.arguments),
                    ),
                )
                for tool_call in message.tool_calls or []
            ]
            assistant_message: LocalChatMessage = {
                "role": "assistant",
                "content": message.content or "",
            }
            if local_calls:
                assistant_message["tool_calls"] = local_calls
            local_messages.append(assistant_message)
        elif isinstance(message, ToolInputMessage):
            # The tokenizer groups tool results by order. The call ID is validated
            # at the HTTP boundary but is not part of the internal prompt syntax.
            local_messages.append({"role": "tool", "content": message.content})
        else:
            local_messages.append(
                {
                    # Internal chat templates use system for developer instructions.
                    "role": "system" if message.role == "developer" else message.role,
                    "content": message.content,
                }
            )
    return local_messages


def _tool_schemas(request: ChatCompletionRequest) -> list[dict[str, object]] | None:
    if request.tool_choice == "none":
        has_tool_history = any(
            isinstance(message, ToolInputMessage)
            or (
                isinstance(message, AssistantInputMessage)
                and bool(message.tool_calls)
            )
            for message in request.messages
        )
        # An empty list keeps the tool-aware history template active without
        # advertising any callable tools for the next assistant turn.
        return [] if has_tool_history else None
    return (
        [
            cast(dict[str, object], tool.model_dump(exclude_none=True))
            for tool in request.tools
        ]
        if request.tools
        else None
    )


def _response_tool_calls(
    completion: LocalChatCompletion,
) -> list[FunctionToolCall] | None:
    if not completion.message.tool_calls:
        return None
    return [
        FunctionToolCall(
            id=f"call_{uuid4().hex}",
            function=FunctionCall(
                name=tool_call.name,
                arguments=json.dumps(
                    tool_call.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )
        for tool_call in completion.message.tool_calls
    ]


def create_app(
    generator: CompletionGenerator,
    *,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    enable_thinking: bool = True,
) -> FastAPI:
    """Create an inference app around one already-loaded model."""
    app = FastAPI(title="LLLM inference server")
    generation_lock = Lock()

    def _create_chat_completion(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if request.model != served_model_name:
            raise HTTPException(
                status_code=404,
                detail=f"model {request.model!r} is not served",
            )
        if request.stream:
            # TODO: Replace this guard with token-yielding generation and SSE chunks.
            raise HTTPException(status_code=501, detail="streaming is not implemented")
        if request.tool_choice == "required" or isinstance(
            request.tool_choice,
            NamedFunctionToolChoice,
        ):
            raise HTTPException(
                status_code=501,
                detail="required or named tool_choice is not implemented",
            )
        if not request.parallel_tool_calls:
            raise HTTPException(
                status_code=501,
                detail="disabling parallel tool calls is not implemented",
            )
        if request.tools and any(tool.function.strict for tool in request.tools):
            raise HTTPException(
                status_code=501,
                detail="strict function schemas are not implemented",
            )

        # Generator metrics and model state are mutable, so one model serves one
        # generation at a time.
        with generation_lock:
            completion = generator.generate_completion(
                _local_messages(request.messages),
                tools=_tool_schemas(request),
                max_generated_token=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                enable_thinking=enable_thinking,
            )

        tool_calls = _response_tool_calls(completion)
        usage = CompletionUsage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.generated_tokens,
            total_tokens=completion.prompt_tokens + completion.generated_tokens,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time.time()),
            model=served_model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        content=completion.message.content or None,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=(
                        "tool_calls" if tool_calls else completion.finish_reason
                    ),
                )
            ],
            usage=usage,
        )

    app.add_api_route(
        "/v1/chat/completions",
        _create_chat_completion,
        methods=["POST"],
        response_model=ChatCompletionResponse,
    )
    return app


@click.command(help="Run the OpenAI-compatible LLLM inference server.")
@click.option(
    "--model",
    default=DEFAULT_SERVER_MODEL_REPO_ID,
    show_default=True,
    help="Hugging Face repo id or local path for a supported model.",
)
@click.option(
    "--served-model-name",
    default=DEFAULT_SERVED_MODEL_NAME,
    show_default=True,
    help="Model alias accepted by the HTTP API.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=click.IntRange(1, 65535))
@click.option(
    "--cache-length",
    default=DEFAULT_CACHE_LENGTH,
    show_default=True,
    type=click.IntRange(min=1),
    help="KV cache length used by the generator.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    help="Only use models already present in the local Hugging Face cache.",
)
@click.option(
    "--no-think",
    is_flag=True,
    help="Disable model thinking mode when supported.",
)
@click.option(
    "--verbosity",
    default="warning",
    show_default=True,
    type=click.Choice(SERVER_LOG_LEVELS, case_sensitive=False),
)
def server_cli(
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    cache_length: int,
    local_files_only: bool,
    no_think: bool,
    verbosity: str,
) -> None:
    """Load one model and serve it until Uvicorn exits."""
    _configure_logging(verbosity)
    generator = build_generator(
        model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    app = create_app(
        generator,
        served_model_name=served_model_name,
        enable_thinking=not no_think,
    )
    uvicorn.run(app, host=host, port=port, log_level=verbosity)
