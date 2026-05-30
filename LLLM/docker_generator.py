"""
Host-side proxy for running ``GeneratorWithTool`` inside a container.

The caller owns this proxy outside the container.  The real generator is
constructed by an importable factory inside a long-lived Docker/Podman
container, and every ``generate`` call is transmitted over HTTP to that worker.

To use the caller must define a factory function that create the
GeneratorWithTool with optional serializable arguments. The factory function
will be executed in the newly created container.
We can't serialize an initialized instance of GeneratorWithTool to a container,
so the GeneratorWithTool instance have to be created in-place.

"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import requests
from loguru import logger

from .generator_with_tool import ToolMessage

DEFAULT_DOCKER_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
DEFAULT_CONTAINER_REPO_PATH = "/workspace/LLLM"
DEFAULT_REPO_PATH = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DockerMount:
    """A bind mount passed to the container engine.

    Args:
        host_path: File or directory on the host that should be mounted.
        container_path: Absolute path where ``host_path`` appears inside the
            container.
        read_only: Mount the path read-only when true, otherwise read-write.
    """

    host_path: str | Path
    container_path: str
    read_only: bool = False

    def as_volume_spec(self) -> dict[str, str]:
        return {
            "bind": self.container_path,
            "mode": "ro" if self.read_only else "rw",
        }


class ContainerizedGeneratorWithTool:
    """Proxy ``GeneratorWithTool.generate`` calls into a long-lived container.

    The host process owns this lightweight proxy. The real
    ``GeneratorWithTool`` is built inside the container by importing
    ``factory`` and calling it with ``factory_kwargs``. Requests to
    :meth:`generate` are serialized to JSON, sent to the worker HTTP server in
    the container, executed there, and returned as a string.

    ``factory`` must use ``"module:callable"`` syntax. For example,
    ``"tests.test_functional_containerized_generator:create_qwen3_06b_agent"``
    imports ``tests.test_functional_containerized_generator`` inside the
    container and calls ``create_qwen3_06b_agent(**factory_kwargs)``. The
    callable must return a ``GeneratorWithTool`` instance and must be importable
    from the container's working directory.

    Args:
        factory: Import path for the container-side generator factory in
            ``"module:callable"`` format.
        factory_kwargs: JSON-serializable keyword arguments passed to the
            factory inside the container.
        docker_image: Docker image used to run the worker. The image must have
            ``uv`` available because the worker command is ``uv run python -m
            LLLM.generator_container_worker``.
        mount_points: Extra bind mounts made available to the container, for
            example model caches or test scratch directories.
        repo_path: Host checkout mounted at ``/workspace/LLLM`` inside the
            container. Defaults to this repository.
        docker_base_url: Optional Docker daemon URL passed to
            ``docker.DockerClient``. When omitted, ``docker.from_env()`` is
            used.
        client: Optional prebuilt Docker client, mainly useful for tests.
        timeout_seconds: HTTP timeout for each ``generate`` request.
        startup_timeout_seconds: Maximum time to wait for the worker health
            endpoint after starting the container.
        worker_port: Host port used by the worker. When omitted, a free local
            port is selected.
        auto_remove: Whether Docker should remove the container after it stops.
    """

    def __init__(
        self,
        factory: str,
        *,
        factory_kwargs: dict[str, object] | None = None,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        mount_points: Sequence[DockerMount] = (),
        repo_path: str | Path | None = None,
        docker_base_url: str | None = None,
        client: Any | None = None,
        timeout_seconds: int = 600,
        startup_timeout_seconds: int = 120,
        worker_port: int | None = None,
        auto_remove: bool = True,
    ) -> None:
        if not factory:
            raise ValueError("factory must not be empty")
        if not docker_image:
            raise ValueError("docker_image must not be empty")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if startup_timeout_seconds < 1:
            raise ValueError("startup_timeout_seconds must be positive")
        self.factory = factory
        self.factory_kwargs = dict(factory_kwargs or {})
        self.docker_image = docker_image
        self.mount_points = tuple(mount_points)
        self.repo_path = Path(repo_path) if repo_path is not None else DEFAULT_REPO_PATH
        self.docker_base_url = docker_base_url
        self._client = client
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.worker_port = worker_port
        self.auto_remove = auto_remove
        self._container: Any | None = None

    def __enter__(self) -> ContainerizedGeneratorWithTool:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def start(self) -> None:
        """Start the worker container if it is not already running.

        Called in every generate().

        The repository is mounted into the container, the factory configuration
        is passed through environment variables, and this method blocks until
        the worker's ``/health`` endpoint responds. It returns ``None`` and
        raises ``RuntimeError`` if the container exits early or never becomes
        ready.
        """
        if self._container is not None:
            return
        if self.worker_port is None:
            self.worker_port = _find_free_local_port()
        client = self._get_client()
        logger.info(
            "Starting GeneratorWithTool container image={} factory={} port={}",
            self.docker_image,
            self.factory,
            self.worker_port,
        )
        self._container = client.containers.run(
            self.docker_image,
            command=[
                "uv",
                "run",
                "python",
                "-m",
                "LLLM.generator_container_worker",
            ],
            detach=True,
            remove=self.auto_remove,
            network_mode="host",
            working_dir=DEFAULT_CONTAINER_REPO_PATH,
            environment={
                "LLLM_GENERATOR_FACTORY": self.factory,
                "LLLM_GENERATOR_FACTORY_KWARGS": json.dumps(self.factory_kwargs),
                "LLLM_GENERATOR_WORKER_HOST": "127.0.0.1",
                "LLLM_GENERATOR_WORKER_PORT": str(self.worker_port),
                "UV_CACHE_DIR": "/tmp/lllm-uv-cache",
                "UV_PROJECT_ENVIRONMENT": "/tmp/lllm-uv-venv",
            },
            volumes=self._volumes(),
        )
        self._wait_until_ready()

    def close(self) -> None:
        """Stop the worker container if this proxy started one.

        Returns ``None``. Calling this method when no container is running is a
        no-op.
        """
        if self._container is None:
            return
        container = self._container
        self._container = None
        logger.info("Stopping GeneratorWithTool container")
        container.stop(timeout=5)

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
        """Forward a generation request into the container.

        Args:
            messages: Chat messages accepted by ``GeneratorWithTool.generate``.
                Each message is JSON-serialized and sent to the worker.
            stop_at_eos: Stop generation when the tokenizer EOS token is
                produced.
            max_generated_token: Maximum number of new tokens to generate for
                each assistant turn.
            cache_length: Optional per-request KV cache length override.
            temperature: Sampling temperature. ``0.0`` uses deterministic
                greedy decoding.
            top_k: Optional top-k sampling cutoff.
            top_p: Optional nucleus sampling cutoff. The underlying generator
                uses this instead of ``top_k`` when provided.

        Returns:
            Final assistant response returned by the container-side
            ``GeneratorWithTool`` after any tool rounds.

        Raises:
            RuntimeError: If the worker returns an HTTP error, a serialized
                remote exception, or an invalid response payload.
        """
        self.start()
        payload: dict[str, object] = {
            "messages": list(messages),
            "stop_at_eos": stop_at_eos,
            "max_generated_token": max_generated_token,
            "cache_length": cache_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        logger.info("Forwarding generate request to container")
        response = requests.post(
            self._url("/generate"),
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"container generate request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        return _decode_generate_response(cast(object, response.json()))

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import docker

        if self.docker_base_url is not None:
            self._client = docker.DockerClient(base_url=self.docker_base_url)
        else:
            self._client = docker.from_env()
        return self._client

    def _volumes(self) -> dict[str, dict[str, str]]:
        volumes = {
            str(self.repo_path.expanduser().resolve()): {
                "bind": DEFAULT_CONTAINER_REPO_PATH,
                "mode": "rw",
            }
        }
        for mount in self.mount_points:
            volumes[str(Path(mount.host_path).expanduser().resolve())] = (
                mount.as_volume_spec()
            )
        return volumes

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if not self._container_is_running():
                raise RuntimeError("generator container exited before becoming ready")
            try:
                response = requests.get(self._url("/health"), timeout=1)
                if response.status_code == 200:
                    logger.info("GeneratorWithTool container is ready")
                    return
            except requests.RequestException as error:
                last_error = error
            time.sleep(0.2)
        raise RuntimeError(
            f"generator container did not become ready within "
            f"{self.startup_timeout_seconds}s: {last_error}"
        )

    def _container_is_running(self) -> bool:
        if self._container is None:
            return False
        self._container.reload()
        attrs = cast(object, getattr(self._container, "attrs", {}))
        if not isinstance(attrs, dict):
            return True
        attrs_dict = cast(dict[str, object], attrs)
        state = attrs_dict.get("State")
        if not isinstance(state, dict):
            return True
        state_dict = cast(dict[str, object], state)
        running = state_dict.get("Running")
        return bool(running) if isinstance(running, bool) else True

    def _url(self, path: str) -> str:
        if self.worker_port is None:
            raise RuntimeError("worker port has not been assigned")
        return f"http://127.0.0.1:{self.worker_port}{path}"


def _decode_generate_response(payload: object) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("container returned invalid JSON payload")
    payload_dict = cast(dict[str, object], payload)
    result = payload_dict.get("result")
    if isinstance(result, str):
        return result
    error = payload_dict.get("error")
    if isinstance(error, dict):
        error_dict = cast(dict[str, object], error)
        message = error_dict.get("message")
        traceback = error_dict.get("traceback")
        if isinstance(traceback, str):
            logger.debug("Remote container traceback:\n{}", traceback)
        if isinstance(message, str):
            raise RuntimeError(message)
    raise RuntimeError("container returned invalid generate response")


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
