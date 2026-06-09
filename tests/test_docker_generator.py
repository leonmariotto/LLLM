from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import requests

from ..LLLM.docker_generator import (
    DEFAULT_CONTAINER_REPO_PATH,
    DEFAULT_CONTAINER_REPO_SOURCE_PATH,
    DEFAULT_CONTAINER_UV_PROJECT_ENVIRONMENT_PATH,
    ContainerizedGeneratorWithTool,
    DockerMount,
)
from ..LLLM.generator_container_worker import (
    execute_generate_payload,
    load_factory,
)
from ..LLLM.generator import ChatMessage
from ..LLLM.tool_common import ToolCall


class FakeContainer:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {"State": {"Running": True}}
        self.stopped = False
        self.log_output = b""

    def reload(self) -> None:
        pass

    def stop(self, *, timeout: int) -> None:
        del timeout
        self.stopped = True

    def logs(self, **kwargs: object) -> bytes:
        del kwargs
        return self.log_output


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
        messages: Sequence[ChatMessage],
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
    data_path = tmp_path / "data"
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        factory_kwargs={"repo_id": "tiny"},
        docker_image="image",
        mount_points=[DockerMount(data_path, "/data", read_only=True)],
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
    assert call["user"] == "1000:1000"
    assert call["working_dir"] == "/workspace"
    command = cast(list[str], call["command"])
    assert command[0:2] == ["sh", "-lc"]
    assert (
        f"cp -R {DEFAULT_CONTAINER_REPO_SOURCE_PATH} "
        f"{DEFAULT_CONTAINER_REPO_PATH}"
    ) in command[2]
    assert f"cd {DEFAULT_CONTAINER_REPO_PATH}" in command[2]
    assert "uv sync" in command[2]
    assert "uv run --no-sync python -m LLLM.generator_container_worker" in (
        command[2]
    )
    assert call["environment"] == {
        "LLLM_GENERATOR_FACTORY": "tests.fake:create_generator",
        "LLLM_GENERATOR_FACTORY_KWARGS": '{"repo_id": "tiny"}',
        "LLLM_GENERATOR_WORKER_HOST": "127.0.0.1",
        "LLLM_GENERATOR_WORKER_PORT": "33333",
        "HF_HOME": "/tmp/lllm-hf-cache",
        "UV_CACHE_DIR": "/tmp/lllm-uv-cache",
        "UV_PROJECT_ENVIRONMENT": DEFAULT_CONTAINER_UV_PROJECT_ENVIRONMENT_PATH,
    }
    volumes = cast(dict[str, dict[str, str]], call["volumes"])
    assert volumes[str(tmp_path)] == {
        "bind": DEFAULT_CONTAINER_REPO_SOURCE_PATH,
        "mode": "ro",
    }
    assert volumes[str(data_path)] == {"bind": "/data", "mode": "ro"}
    assert volumes[str(Path.home() / ".cache" / "uv")] == {
        "bind": "/tmp/lllm-uv-cache",
        "mode": "rw",
    }
    assert volumes[str(Path.home() / ".cache" / "huggingface")] == {
        "bind": "/tmp/lllm-hf-cache",
        "mode": "rw",
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
    assert proxy.container_log_path is not None
    assert proxy.container_log_path.parent == tmp_path / "container_logs"
    assert proxy.container_log_path.name.startswith("containerized-generator-")
    assert proxy.container_log_path.suffix == ".log"


def test_containerized_generator_records_container_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    client.container.log_output = b"boot\nready\n"

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({"ok": True})

    def fake_post(*_args: object, **_kwargs: object) -> FakeResponse:
        client.container.log_output = b"boot\nready\ngenerated\n"
        return FakeResponse({"result": "answer"})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    assert proxy.generate([{"role": "user", "content": "hi"}]) == "answer"

    assert proxy.container_log_path is not None
    assert proxy.container_log_path.read_bytes() == b"boot\nready\ngenerated\n"


def test_containerized_generator_can_disable_container_log_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    client.container.log_output = b"boot\nready\n"

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "get", fake_get)
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
        record_log=False,
    )

    proxy.start()

    assert proxy.container_log_path is None
    assert not (tmp_path / "container_logs").exists()


def test_containerized_generator_can_override_container_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({})

    monkeypatch.setattr(requests, "get", fake_get)
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
        container_user=None,
    )

    proxy.start()

    assert client.containers.calls[0]["image"] == "buildpack-deps:bookworm-curl"
    assert client.containers.calls[0]["user"] is None


def test_containerized_generator_raises_remote_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({})

    def fake_post(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse(
            {"error": {"type": "RuntimeError", "message": "remote failed"}}
        )

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        requests,
        "post",
        fake_post,
    )
    proxy = ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    with pytest.raises(RuntimeError, match="remote failed"):
        proxy.generate([{"role": "user", "content": "hi"}])


def test_containerized_generator_context_manager_stops_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({})

    monkeypatch.setattr(requests, "get", fake_get)

    with ContainerizedGeneratorWithTool(
        "tests.fake:create_generator",
        repo_path=tmp_path,
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
    call = ToolCall(name="compute", arguments={"expression": "2 + 2"})

    assert call.name == "compute"
