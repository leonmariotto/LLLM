from pathlib import Path
from typing import cast

import pytest
import requests

from ..LLLM.containerized_agent_client import (
    DEFAULT_CONTAINER_REPO_PATH,
    DEFAULT_CONTAINER_REPO_SOURCE_PATH,
    DEFAULT_CONTAINER_UV_PROJECT_ENVIRONMENT_PATH,
    ContainerizedAgent,
    DockerMount,
)
from ..LLLM.containerized_agent_server import (
    execute_run_payload,
    load_factory,
)
from ..LLLM.agent_context import AgentResult, ExecutionContext
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


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        prompt: str,
    ) -> AgentResult:
        self.calls.append({"prompt": prompt})
        context = ExecutionContext()
        context.add_user_message(prompt)
        context.final_result = "container result"
        return AgentResult(output="container result", context=context)


def agent_result_payload(output: str = "answer") -> dict[str, object]:
    return {
        "result": {
            "output": output,
            "status": "complete",
            "context": {
                "execution_id": "execution",
                "events": [],
                "current_step": 0,
                "state": {},
                "final_result": output,
            },
        }
    }


def test_docker_mount_formats_volume_spec() -> None:
    mount = DockerMount("/tmp/data", "/data", read_only=True)

    assert mount.as_volume_spec() == {"bind": "/data", "mode": "ro"}


def test_containerized_agent_starts_container_and_forwards_run(
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
        assert url == "http://127.0.0.1:33333/run"
        posted.append(kwargs)
        return FakeResponse(agent_result_payload())

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    data_path = tmp_path / "data"
    proxy = ContainerizedAgent(
        "tests.fake:create_generator",
        factory_kwargs={"repo_id": "tiny"},
        docker_image="image",
        mount_points=[DockerMount(data_path, "/data", read_only=True)],
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    result = proxy.run("hi")

    assert result.output == "answer"
    call = client.containers.calls[0]
    assert call["image"] == "image"
    assert call["network_mode"] == "host"
    assert call["user"] == "1000:1000"
    assert call["working_dir"] == "/tmp"
    command = cast(list[str], call["command"])
    assert command[0:2] == ["sh", "-lc"]
    assert (
        f"cp -R {DEFAULT_CONTAINER_REPO_SOURCE_PATH} "
        f"{DEFAULT_CONTAINER_REPO_PATH}"
    ) in command[2]
    assert f"cd {DEFAULT_CONTAINER_REPO_PATH}" in command[2]
    assert "uv sync" in command[2]
    assert f"mkdir -p {Path(DEFAULT_CONTAINER_REPO_PATH).parent}" in command[2]
    assert "uv run --no-sync python -m LLLM.containerized_agent_server" in (
        command[2]
    )
    assert call["environment"] == {
        "LLLM_AGENT_FACTORY": "tests.fake:create_generator",
        "LLLM_AGENT_FACTORY_KWARGS": '{"repo_id": "tiny"}',
        "LLLM_AGENT_WORKER_HOST": "127.0.0.1",
        "LLLM_AGENT_WORKER_PORT": "33333",
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
    assert posted[0]["json"] == {"prompt": "hi"}
    assert proxy.container_log_path is not None
    assert proxy.container_log_path.parent == tmp_path / "container_logs"
    assert proxy.container_log_path.name.startswith("containerized-agent-")
    assert proxy.container_log_path.suffix == ".log"


def test_containerized_agent_records_container_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    client.container.log_output = b"boot\nready\n"

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({"ok": True})

    def fake_post(*_args: object, **_kwargs: object) -> FakeResponse:
        client.container.log_output = b"boot\nready\ngenerated\n"
        return FakeResponse(agent_result_payload())

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    proxy = ContainerizedAgent(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    assert proxy.run("hi").output == "answer"

    assert proxy.container_log_path is not None
    assert proxy.container_log_path.read_bytes() == b"boot\nready\ngenerated\n"


def test_containerized_agent_can_disable_container_log_recording(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    client.container.log_output = b"boot\nready\n"

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "get", fake_get)
    proxy = ContainerizedAgent(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
        record_log=False,
    )

    proxy.start()

    assert proxy.container_log_path is None
    assert not (tmp_path / "container_logs").exists()


def test_containerized_agent_can_override_container_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({})

    monkeypatch.setattr(requests, "get", fake_get)
    proxy = ContainerizedAgent(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
        container_user=None,
    )

    proxy.start()

    assert client.containers.calls[0]["image"] == "buildpack-deps:bookworm-curl"
    assert client.containers.calls[0]["user"] is None


def test_containerized_agent_raises_remote_errors(
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
    proxy = ContainerizedAgent(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    )

    with pytest.raises(RuntimeError, match="remote failed"):
        proxy.run("hi")


def test_containerized_agent_context_manager_stops_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()

    def fake_get(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({})

    monkeypatch.setattr(requests, "get", fake_get)

    with ContainerizedAgent(
        "tests.fake:create_generator",
        repo_path=tmp_path,
        client=client,
        worker_port=33333,
    ) as proxy:
        assert proxy is not None

    assert client.container.stopped


def test_worker_execute_run_payload_forwards_prompt() -> None:
    agent = FakeAgent()

    result = execute_run_payload(
        agent,  # type: ignore[arg-type]
        {"prompt": "hi"},
    )

    assert result.output == "container result"
    assert agent.calls == [{"prompt": "hi"}]


def test_worker_execute_run_payload_validates_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        execute_run_payload(FakeAgent(), {})  # type: ignore[arg-type]


def test_load_factory_rejects_invalid_path() -> None:
    with pytest.raises(ValueError, match="module:callable"):
        load_factory("not.a.factory")


def test_tool_call_dataclass_stays_importable_for_rpc_payloads() -> None:
    call = ToolCall(name="compute", arguments={"expression": "2 + 2"})

    assert call.name == "compute"
