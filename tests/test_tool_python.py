import subprocess
from typing import Any

import pytest
from loguru import logger

from ..LLLM.tool_common import Tool
from ..LLLM.tool_python import execute_python, python_tool


def test_python_tool_returns_registered_tool() -> None:
    tool = python_tool()

    assert isinstance(tool, Tool)
    assert tool.schema["type"] == "function"
    function = tool.schema["function"]
    assert isinstance(function, dict)
    assert function["name"] == "python"
    assert tool.execute({"code": "print(2 + 2)"}) == "Exit code: 0\nstdout:\n4"


def test_execute_python_runs_raw_python_code() -> None:
    output = execute_python(
        {
            "code": (
                "import math\n"
                "values = [1, 2, 3]\n"
                "print(sum(values))\n"
                "print(math.sqrt(81))"
            )
        }
    )

    assert output == "Exit code: 0\nstdout:\n6\n9.0"


def test_execute_python_returns_tracebacks_without_raising() -> None:
    output = execute_python({"code": "print('before')\n1 / 0"})

    assert "Exit code: 1" in output
    assert "stdout:\nbefore" in output
    assert "stderr:" in output
    assert "ZeroDivisionError" in output


def test_execute_python_reports_empty_output() -> None:
    assert execute_python({"code": "x = 1"}) == "Exit code: 0\nstdout: <empty>"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"code": 1},
        {"code": ""},
        {"code": "   "},
        {"code": "print(1)", "timeout_seconds": True},
        {"code": "print(1)", "timeout_seconds": 0},
        {"code": "print(1)", "timeout_seconds": 31},
    ],
)
def test_execute_python_validates_arguments(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        execute_python(arguments)


def test_execute_python_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["uv", "run", "python"], timeout=2)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="timed out"):
        execute_python({"code": "while True: pass", "timeout_seconds": 2})


def test_execute_python_truncates_large_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["uv", "run", "python", "-c", "..."],
            returncode=0,
            stdout="x" * 13000,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = execute_python({"code": "print('large')"})

    assert output.endswith("[truncated]")
    assert len(output) < 12100


def test_execute_python_logs_execution_summary() -> None:
    logs: list[str] = []
    sink_id = logger.add(lambda message: logs.append(str(message)), level="INFO")

    try:
        execute_python({"code": "print(123)"})
    finally:
        logger.remove(sink_id)

    text = "".join(logs)
    assert "Python tool execution started" in text
    assert "Python tool execution completed with returncode=0" in text
