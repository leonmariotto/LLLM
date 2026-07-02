from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner
from fastapi.testclient import TestClient
import pytest

from ..LLLM import server as server_module
from ..LLLM.generator import (
    AssistantOutput,
    ChatCompletion,
    ChatMessage,
)
from ..LLLM.tool_common import ToolCall


class FakeGenerator:
    def __init__(self, output: AssistantOutput | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.output = output or AssistantOutput("Hello")

    def generate_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        max_generated_token: int = 20,
        temperature: float = 0.0,
        top_p: float | None = None,
        enable_thinking: bool = True,
    ) -> ChatCompletion:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
                "max_generated_token": max_generated_token,
                "temperature": temperature,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
            }
        )
        return ChatCompletion(
            message=self.output,
            raw_completion="Hello",
            prompt_tokens=12,
            generated_tokens=3,
            finish_reason="stop",
        )


def test_chat_completion_maps_request_and_response() -> None:
    generator = FakeGenerator()
    client = TestClient(
        server_module.create_app(
            generator,
            served_model_name="local-model",
            enable_thinking=False,
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 32,
            "temperature": 0.4,
            "top_p": 0.8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["object"] == "chat.completion"
    assert body["model"] == "local-model"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello",
                "tool_calls": None,
            },
            "finish_reason": "stop",
            "logprobs": None,
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    assert generator.calls == [
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
            "tools": None,
            "max_generated_token": 32,
            "temperature": 0.4,
            "top_p": 0.8,
            "enable_thinking": False,
        }
    ]


def test_chat_completion_rejects_unknown_model() -> None:
    client = TestClient(server_module.create_app(FakeGenerator()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "other",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "model 'other' is not served"}


def test_chat_completion_marks_streaming_not_implemented() -> None:
    client = TestClient(server_module.create_app(FakeGenerator()))
    request: dict[str, object] = {
        "model": "lllm",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 501
    assert response.json() == {"detail": "streaming is not implemented"}


def test_chat_completion_returns_openai_function_tool_calls() -> None:
    generator = FakeGenerator(
        AssistantOutput(
            "",
            (
                ToolCall(
                    name="get_weather",
                    arguments={"city": "Paris", "units": "celsius"},
                ),
            ),
        )
    )
    client = TestClient(server_module.create_app(generator))
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lllm",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [tool],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["type"] == "function"
    assert calls[0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Paris","units":"celsius"}',
    }
    assert generator.calls[0]["tools"] == [tool]


def test_chat_completion_accepts_tool_call_and_result_history() -> None:
    generator = FakeGenerator(AssistantOutput("It is sunny."))
    client = TestClient(server_module.create_app(generator))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lllm",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": '{"condition":"sunny"}',
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert generator.calls[0]["messages"] == [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(
                    tool_call_id="call_weather",
                    name="get_weather",
                    arguments={"city": "Paris"},
                )
            ],
        },
        {"role": "tool", "content": '{"condition":"sunny"}'},
    ]


@pytest.mark.parametrize(
    ("request_update", "detail"),
    [
        (
            {"tool_choice": "required"},
            "required or named tool_choice is not implemented",
        ),
        (
            {"parallel_tool_calls": False},
            "disabling parallel tool calls is not implemented",
        ),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "strict": True},
                    }
                ]
            },
            "strict function schemas are not implemented",
        ),
    ],
)
def test_chat_completion_rejects_unsupported_tool_controls(
    request_update: dict[str, object],
    detail: str,
) -> None:
    client = TestClient(server_module.create_app(FakeGenerator()))
    request: dict[str, object] = {
        "model": "lllm",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    request.update(request_update)

    response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 501
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    "request_update",
    [
        {"messages": [{"role": "tool", "content": "result"}]},
        {"messages": [{"role": "user", "content": ["not", "text"]}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "not json",
                            },
                        }
                    ],
                }
            ]
        },
        {"temperature": 3.0},
        {"top_p": 0.0},
        {"max_tokens": 0},
    ],
)
def test_chat_completion_validates_minimal_request(
    request_update: dict[str, Any],
) -> None:
    client = TestClient(server_module.create_app(FakeGenerator()))
    request: dict[str, object] = {
        "model": "lllm",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request.update(request_update)

    response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 422


def test_server_cli_loads_model_and_runs_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeGenerator()
    captured: dict[str, object] = {}

    def fake_build_generator(repo_id: str, **kwargs: object) -> FakeGenerator:
        captured["repo_id"] = repo_id
        captured["build_kwargs"] = kwargs
        return generator

    def fake_uvicorn_run(app: object, **kwargs: object) -> None:
        captured["app"] = app
        captured["uvicorn_kwargs"] = kwargs

    monkeypatch.setattr(server_module, "build_generator", fake_build_generator)
    monkeypatch.setattr(server_module, "_configure_logging", lambda _: None)
    monkeypatch.setattr(server_module.uvicorn, "run", fake_uvicorn_run)

    result = CliRunner().invoke(
        server_module.server_cli,
        [
            "--model",
            "repo/model",
            "--served-model-name",
            "local",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--cache-length",
            "512",
            "--local-files-only",
            "--no-think",
            "--verbosity",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["repo_id"] == "repo/model"
    assert captured["build_kwargs"] == {
        "cache_length": 512,
        "local_files_only": True,
    }
    assert captured["uvicorn_kwargs"] == {
        "host": "0.0.0.0",
        "port": 9000,
        "log_level": "debug",
    }
    assert captured["app"] is not None


@pytest.mark.parametrize(
    ("architecture", "builder_name"),
    [
        ("qwen3", "_build_qwen3_generator"),
        ("gemma3", "_build_gemma3_generator"),
    ],
)
def test_build_generator_dispatches_by_ir_architecture(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    builder_name: str,
) -> None:
    generator = FakeGenerator()
    ir = SimpleNamespace(architecture=architecture)
    captured: dict[str, object] = {}

    def fake_fetch(repo_id: str, **kwargs: object) -> object:
        captured["repo_id"] = repo_id
        captured["fetch_kwargs"] = kwargs
        return ir

    def fake_builder(model_ir: object, *, cache_length: int) -> FakeGenerator:
        captured["ir"] = model_ir
        captured["cache_length"] = cache_length
        return generator

    monkeypatch.setattr(server_module, "fetch_model_ir", fake_fetch)
    monkeypatch.setattr(server_module, builder_name, fake_builder)

    result = server_module.build_generator(
        "repo/model",
        cache_length=512,
        local_files_only=True,
    )

    assert result is generator
    assert captured == {
        "repo_id": "repo/model",
        "fetch_kwargs": {"local_files_only": True},
        "ir": ir,
        "cache_length": 512,
    }


def test_build_generator_rejects_unsupported_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = SimpleNamespace(architecture="llama3")
    monkeypatch.setattr(
        server_module,
        "fetch_model_ir",
        lambda *args, **kwargs: ir,
    )

    with pytest.raises(ValueError, match="supported architectures: qwen3, gemma3"):
        server_module.build_generator(
            "repo/model",
            cache_length=512,
            local_files_only=False,
        )
