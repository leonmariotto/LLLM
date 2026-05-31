from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from LLLM.docker_generator import (
    DEFAULT_CONTAINER_HF_CACHE_PATH,
    DEFAULT_CONTAINER_UV_CACHE_PATH,
    DEFAULT_HOST_HF_CACHE_PATH,
    DEFAULT_HOST_UV_CACHE_PATH,
    ContainerizedGeneratorWithTool,
    DockerMount,
)
from LLLM.fetch import fetch_model_ir
from LLLM.generator import Generator
from LLLM.generator_with_tool import GeneratorWithTool, Tool
from LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"
CONTAINER_TMP_PATH = "/tmp/lllm-containerized-generator"


def create_qwen3_06b_agent(*, marker_path: str | None = None) -> GeneratorWithTool:
    """Factory imported by ``LLLM.generator_container_worker`` inside Docker."""
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    generator = Generator(model=model, tokenizer=tokenizer, cache_length=4096)

    def container_marker(arguments: dict[str, object]) -> str:
        if marker_path is not None:
            marker = Path(marker_path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(arguments, sort_keys=True), encoding="utf-8")
        return "containerized-qwen3-tool-ok"

    return GeneratorWithTool(
        generator,
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "container_marker",
                        "description": (
                            "Record that the containerized Qwen3 agent reached "
                            "tool execution, then return a fixed sentinel."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "proof": {
                                    "type": "string",
                                    "description": "The exact proof value requested.",
                                },
                            },
                            "required": ["proof"],
                        },
                    },
                },
                execute=container_marker,
            )
        ],
        max_tool_rounds=3,
    )


@pytest.mark.slow
def test_functional_qwen3_06b_agent_runs_inside_container(tmp_path: Path) -> None:
    docker_client = _require_docker_client()
    container_tmp = f"{CONTAINER_TMP_PATH}/{tmp_path.name}"
    marker_path = f"{container_tmp}/tool-call.json"

    with ContainerizedGeneratorWithTool(
        "tests.test_functional_containerized_generator:create_qwen3_06b_agent",
        factory_kwargs={"marker_path": marker_path},
        mount_points=[
            DockerMount(tmp_path, container_tmp),
            *_cache_mounts(),
        ],
        client=docker_client,
        timeout_seconds=900,
        startup_timeout_seconds=3000,
        auto_remove=False,
    ) as agent:
        response = agent.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "/no_think\n"
                        "Call the container_marker tool exactly once with "
                        '{"proof": "docker-worker-qwen3"}. First reply only '
                        "with a valid <tool_call></tool_call> block. After the "
                        "tool response, answer with the tool response exactly "
                        "and no extra text."
                    ),
                }
            ],
            max_generated_token=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

    assert (tmp_path / "tool-call.json").read_text(encoding="utf-8") == (
        '{"proof": "docker-worker-qwen3"}'
    )
    assert "containerized-qwen3-tool-ok" in response


def _require_docker_client() -> Any:
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"Docker is required for this functional test: {error}")
    return client


def _cache_mounts() -> list[DockerMount]:
    mounts: list[DockerMount] = []
    for host_path, container_path in [
        (DEFAULT_HOST_HF_CACHE_PATH, DEFAULT_CONTAINER_HF_CACHE_PATH),
        (DEFAULT_HOST_UV_CACHE_PATH, DEFAULT_CONTAINER_UV_CACHE_PATH),
    ]:
        host_path.mkdir(parents=True, exist_ok=True)
        mounts.append(DockerMount(host_path, container_path))
    return mounts
