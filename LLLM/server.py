"""
Small OpenAI-compatible HTTP server for local LLLM inference.
Provide a command line interface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
from pathlib import Path
import sys
import time
from typing import Annotated, Literal, Protocol, Self, cast
from uuid import uuid4

import click
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
import torch
import uvicorn

from .fetch import fetch_model_ir
from .generator import Generator
from .generator import ChatCompletion as LocalChatCompletion
from .generator import ChatMessage as LocalChatMessage
from .generator import JsonObjectSpec, schema_from_json_schema
from .model_ir import ModelIR
from .generator import ToolCall
from .utils import get_device


DEFAULT_CACHE_LENGTH = 16384
DEFAULT_SERVER_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_SERVED_MODEL_NAME = "lllm"
SERVER_LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")
SERVER_DTYPES = ("auto", "float16", "float32")


class CompletionGenerator(Protocol):
    """Generator operations used by the HTTP adapter."""

    def generate_completion(
        self,
        messages: Sequence[LocalChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        max_generated_token: int = 20,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        enable_thinking: bool = True,
        response_format: JsonObjectSpec | None = None,
    ) -> LocalChatCompletion: ...


def _tokenizer_artifact_path(ir: ModelIR) -> Path:
    path = Path(str(ir.metadata["path"]))
    return path if path.suffix.lower() == ".gguf" else path / "tokenizer.json"


def _sentencepiece_artifact_path(ir: ModelIR) -> Path:
    path = Path(str(ir.metadata["path"]))
    if path.suffix.lower() == ".gguf":
        raise ValueError("Llama 2 server inference requires HF tokenizer.model")
    tokenizer_path = path / "tokenizer.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.model in {path}")
    return tokenizer_path


def _move_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.to(device=device, dtype=dtype)


def _build_gpt2_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .gpt2 import GeneratorGPT2, GPT2Model, GPT2Tokenizer

    model = GPT2Model(GPT2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    return GeneratorGPT2(
        model=model,
        tokenizer=GPT2Tokenizer(),
        cache_length=min(cache_length, model.context_length),
    )


def _build_llama2_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .llama2 import Llama2Model, Llama2Tokenizer

    model = Llama2Model(Llama2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    tokenizer = Llama2Tokenizer(str(_sentencepiece_artifact_path(ir)))
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_llama3_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .llama3 import Llama3Model, Llama3Tokenizer

    model = Llama3Model(Llama3Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    tokenizer_path = _tokenizer_artifact_path(ir)
    tokenizer = (
        Llama3Tokenizer.from_gguf(str(tokenizer_path))
        if tokenizer_path.suffix.lower() == ".gguf"
        else Llama3Tokenizer(str(tokenizer_path))
    )
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_qwen2_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .qwen2 import Qwen2Model, Qwen2Tokenizer

    model = Qwen2Model(Qwen2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    tokenizer = Qwen2Tokenizer(str(_tokenizer_artifact_path(ir)))
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_qwen3_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .qwen3 import Qwen3Model, Qwen3Tokenizer

    model = Qwen3Model(Qwen3Model.config_from_ir(ir))
    tokenizer = Qwen3Tokenizer(str(_tokenizer_artifact_path(ir)))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_gemma3_generator(
    ir: ModelIR,
    *,
    cache_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Generator:
    from .gemma3 import Gemma3Model, Gemma3Tokenizer

    model = Gemma3Model(Gemma3Model.config_from_ir(ir))
    tokenizer = Gemma3Tokenizer(str(_tokenizer_artifact_path(ir)))
    model.load_ir_weights(ir)
    _move_model(model, device=device, dtype=dtype)
    model.dtype = dtype
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _inference_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "float16":
        if device.type == "cpu":
            raise ValueError("float16 server inference requires a GPU device")
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported inference dtype {name!r}")


def build_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
    dtype: str = "auto",
) -> CompletionGenerator:
    """Load a supported model architecture into the shared generator."""
    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    device = get_device()
    torch_dtype = _inference_dtype(dtype, device)
    logger.info("Loading model on device={} dtype={}", device, torch_dtype)
    builders = {
        "gpt2": _build_gpt2_generator,
        "llama2": _build_llama2_generator,
        "llama3": _build_llama3_generator,
        "qwen2": _build_qwen2_generator,
        "qwen3": _build_qwen3_generator,
        "gemma3": _build_gemma3_generator,
    }
    builder = builders.get(ir.architecture)
    if builder is not None:
        return builder(
            ir,
            cache_length=cache_length,
            device=device,
            dtype=torch_dtype,
        )
    raise ValueError(
        f"unsupported inference server architecture {ir.architecture!r}; "
        f"supported architectures: {', '.join(builders)}"
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


class JsonSchemaDefinition(_StrictModel):
    name: str = Field(min_length=1)
    description: str | None = None
    schema_value: dict[str, object] = Field(alias="schema")
    strict: bool | None = None


class JsonSchemaResponseFormat(_StrictModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class ChatCompletionRequest(_StrictModel):
    model: str
    messages: list[ChatCompletionMessage] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    enable_thinking: bool | None = None
    stream: bool = False
    tools: list[FunctionTool] | None = None
    tool_choice: (
        Literal["none", "auto", "required"] | NamedFunctionToolChoice | None
    ) = None
    parallel_tool_calls: bool = True
    response_format: JsonSchemaResponseFormat | None = None


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
                ToolCall(
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
            or (isinstance(message, AssistantInputMessage) and bool(message.tool_calls))
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


def _response_format_spec(
    response_format: JsonSchemaResponseFormat | None,
) -> JsonObjectSpec | None:
    if response_format is None:
        return None
    definition = response_format.json_schema
    try:
        return schema_from_json_schema(
            definition.schema_value,
            name=definition.name,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def create_app(
    generator: CompletionGenerator,
    *,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    enable_thinking: bool = True,
) -> FastAPI:
    """Create an inference app around one already-loaded model."""
    app = FastAPI(title="LLLM inference server")
    generation_lock = asyncio.Lock()

    async def _create_chat_completion(
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

        logger.debug("Receiving new request")

        local_messages = _local_messages(request.messages)
        tool_schemas = _tool_schemas(request)
        response_format = _response_format_spec(request.response_format)
        request_enable_thinking = (
            enable_thinking
            if request.enable_thinking is None
            else request.enable_thinking
        )

        def generate() -> LocalChatCompletion:
            return generator.generate_completion(
                local_messages,
                tools=tool_schemas,
                max_generated_token=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                enable_thinking=request_enable_thinking,
                response_format=response_format,
            )

        # Waiting requests stay suspended on the event loop instead of occupying
        # worker threads or allocating inference state. Only the lock holder is
        # dispatched to a worker because generation is synchronous and CPU/GPU
        # intensive.
        async with generation_lock:
            logger.debug("Running request")
            completion = await run_in_threadpool(generate)

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
    app.add_api_route(
        "/chat/completions",
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
    "--dtype",
    default="auto",
    show_default=True,
    type=click.Choice(SERVER_DTYPES, case_sensitive=False),
    help="Inference dtype. Auto uses float16 on CUDA and float32 otherwise.",
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
    dtype: str,
    no_think: bool,
    verbosity: str,
) -> None:
    """Load one model and serve it until Uvicorn exits."""
    _configure_logging(verbosity)
    generator = build_generator(
        model,
        cache_length=cache_length,
        local_files_only=local_files_only,
        dtype=dtype,
    )
    app = create_app(
        generator,
        served_model_name=served_model_name,
        enable_thinking=not no_think,
    )
    uvicorn.run(app, host=host, port=port, log_level=verbosity)
