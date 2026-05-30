from collections.abc import Sequence
from pathlib import Path

import pytest
import requests

from ..LLLM.docker_generator import ContainerizedGeneratorWithTool, DockerMount
from ..LLLM.generator_container_worker import execute_generate_payload, load_factory
from ..LLLM.generator_with_tool import ToolCall, ToolMessage


class FakeContainer:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {"State": {"Running": True}}
        self.stopped = False

    def reload(self) -> None:
        pass

    def stop(self, *, timeout: int) -> None:
        del timeout
        self.stopped = True


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.calls: list[dict[str, object]] = []

    def run(self, image: str, **kwargs: object) -> FakeContainer:
        self.calls.append({"image": image, **kwargs})
        return self.container


class FakeDockerClient:
    def __init__(self) -> None:
        self.container = FakeContainer()
        self.containers = FakeContainers(self.container)


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> object:
        return self.payload


class FakeGeneratorWithTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        messages: Sequence[ToolMessage],
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "stop_at_eos": stop_at_eos,
                "max_generated_token": max_generated_token,
                "cache_length": cache_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
        return "container result"


def test_docker_mount_formats_volume_spec() -> None:
    mount = DockerMount("/tmp/data", "/data", read_only=True)

    assert mount.as_volume_spec() == {"bind": "/data", "mode": "ro"}


def test_containerized_generator_starts_container_and_forwards_generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    posted: list[dict[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        del kwargs
        assert url == "http://127.0.0.1:33333/health"
        return FakeResponse({"ok": True})

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        assert url == "http://127.0.0.1:33333/generate"
        posted.append(kwargs)
        return FakeResponse({"result": "answer"})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        factory_kwargs={"repo_id": "tiny"},
        docker_image="image",
        mount_points=[DockerMount(tmp_path, "/data", read_only=True)],
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    result = proxy.generate(
        [{"role": "user", "content": "hi"}],
        max_generated_token=12,
        temperature=0.2,
    )

    assert result == "answer"
    call = client.containers.calls[0]
    assert call["image"] == "image"
    assert call["network_mode"] == "host"
    assert call["working_dir"] == "/workspace/LLLM"
    assert call["environment"] == {
        "LLLM_GENERATOR_FACTORY": "tests.fake:create_generator",
        "LLLM_GENERATOR_FACTORY_KWARGS": '{"repo_id": "tiny"}',
        "LLLM_GENERATOR_WORKER_HOST": "127.0.0.1",
        "LLLM_GENERATOR_WORKER_PORT": "33333",
        "UV_CACHE_DIR": "/tmp/lllm-uv-cache",
        "UV_PROJECT_ENVIRONMENT": "/tmp/lllm-uv-venv",
    }
    assert posted[0]["json"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "stop_at_eos": True,
        "max_generated_token": 12,
        "cache_length": None,
        "temperature": 0.2,
        "top_k": None,
        "top_p": None,
    }


def test_containerized_generator_raises_remote_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDockerClient()
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: FakeResponse({}))
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            {"error": {"type": "RuntimeError", "message": "remote failed"}}
        ),
    )
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        client=client,
        worker_port=33333,
    )

    with pytest.raises(RuntimeError, match="remote failed"):
        proxy.generate([{"role": "user", "content": "hi"}])


def test_containerized_generator_context_manager_stops_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDockerClient()
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: FakeResponse({}))

    with ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        client=client,
        worker_port=33333,
    ) as proxy:
        assert proxy is not None

    assert client.container.stopped


def test_worker_execute_generate_payload_forwards_options() -> None:
    generator = FakeGeneratorWithTool()

    result = execute_generate_payload(
        generator,  # type: ignore[arg-type]
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stop_at_eos": False,
            "max_generated_token": 7,
            "cache_length": 128,
            "temperature": 0.5,
            "top_k": 3,
            "top_p": 0.9,
        },
    )

    assert result == "container result"
    assert generator.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stop_at_eos": False,
            "max_generated_token": 7,
            "cache_length": 128,
            "temperature": 0.5,
            "top_k": 3,
            "top_p": 0.9,
        }
    ]


def test_worker_execute_generate_payload_validates_messages() -> None:
    with pytest.raises(ValueError, match="messages"):
        execute_generate_payload(FakeGeneratorWithTool(), {})  # type: ignore[arg-type]


def test_load_factory_rejects_invalid_path() -> None:
    with pytest.raises(ValueError, match="module:callable"):
        load_factory("not.a.factory")


def test_tool_call_dataclass_stays_importable_for_rpc_payloads() -> None:
    call = ToolCall("compute", {"expression": "2 + 2"})

    assert call.name == "compute"
