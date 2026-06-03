"""
bc-backed compute tool for model tool-use loops.

This module exposes a ``compute`` tool that forwards a single expression to
``bc -l`` and returns the calculator output.
Found that a known syntax (bc) is better than a custom approximate "math" syntax,
and still simple enough for dumb model.
"""

from __future__ import annotations

import copy
import re
import subprocess

from .generator_with_tool import Tool

_BC_TIMEOUT_SECONDS = 5
_PI_PATTERN = re.compile(r"\bpi\b")

COMPUTE_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "compute",
        "description": (
            "Evaluate calculator input using bc -l. Use this for exact numeric "
            "calculations. The expression argument is sent directly to the "
            "calculator, so do not include the tool name or words like compute "
            "inside the expression. Use bc syntax, not Python syntax: use ^ for "
            "powers, scale=N for decimal precision, and functions such as "
            "sqrt(x), l(x), and e(x). pi constant is 4*a(1)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": ("Calculator input in bc syntax."),
                }
            },
            "required": ["expression"],
        },
    },
}


def compute_tool() -> Tool:
    """Return the ready-to-register ``compute`` tool."""
    return Tool(schema=copy.deepcopy(COMPUTE_TOOL_SCHEMA), execute=execute_compute)


def execute_compute(arguments: dict[str, object]) -> str:
    """Execute the compute tool with a ``{"expression": "..."}`` argument."""
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ValueError("expression must be a string")
    if not expression.strip():
        raise ValueError("expression must not be empty")

    try:
        completed = subprocess.run(
            ["bc", "-l"],
            input=f"{_preprocess_expression(expression)}\n",
            text=True,
            capture_output=True,
            timeout=_BC_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as error:
        raise ValueError("bc executable was not found") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("bc execution timed out") from error

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        message = stderr or f"bc exited with status {completed.returncode}"
        raise ValueError(f"bc error: {message}")
    if stderr:
        raise ValueError(f"bc error: {stderr}")
    return stdout


def _preprocess_expression(expression: str) -> str:
    return _PI_PATTERN.sub("4*a(1)", expression).replace("π", "4*a(1)")
