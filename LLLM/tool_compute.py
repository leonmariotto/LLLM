"""
Calculator tool for model tool-use loops.

This module exposes a small ``compute`` tool for arithmetic, not a Python
interpreter.  It accepts one expression such as ``2 + 3 * 4``, ``sqrt(81)``,
``sin(pi / 2)``, or ``mean([2, 4, 6])`` and returns the computed result as a
string.

Intentionally unsupported Python syntax includes imports, assignments,
attributes, indexing, comprehensions, comparisons, and statements.  Keeping the
language small makes the tool easier for small models to use and gives them
short, predictable error messages when they call it incorrectly.
"""

from __future__ import annotations

import ast
import copy
import math
import statistics
from collections.abc import Callable, Sequence

from .generator_with_tool import Tool

Number = int | float
Value = Number | list[Number] | tuple[Number, ...]

COMPUTE_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "compute",
        "description": (
            "Evaluate one calculator-style math expression. Use this for "
            "arithmetic, percentages, powers, square roots, logs, trig, sums, "
            "averages, min/max, and exact numeric calculations. This is not a "
            "Python interpreter: send only an expression. Use ** for powers. "
            "Use this tool whenever you have calculation to do."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": ("A single math expression."),
                }
            },
            "required": ["expression"],
        },
    },
}

_CONSTANTS: dict[str, Number] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}


def _one_arg(function: Callable[[float], Number]) -> Callable[[Value], Value]:
    def wrapped(value: Value) -> Value:
        return function(_require_number(value))

    return wrapped


def _round(*args: Value) -> Value:
    if len(args) not in (1, 2):
        raise ValueError("round expects 1 or 2 arguments")
    number = _require_number(args[0])
    if len(args) == 1:
        return round(number)
    digits = _require_number(args[1])
    if not isinstance(digits, int):
        raise ValueError("round digits must be an integer")
    return round(number, digits)


def _log(*args: Value) -> Value:
    if len(args) not in (1, 2):
        raise ValueError("log expects 1 or 2 arguments")
    number = _require_number(args[0])
    if len(args) == 1:
        return math.log(number)
    return math.log(number, _require_number(args[1]))


def _sum(*args: Value) -> Value:
    return sum(_collect_numbers("sum", args))


def _min(*args: Value) -> Value:
    numbers = _collect_numbers("min", args)
    return min(numbers)


def _max(*args: Value) -> Value:
    numbers = _collect_numbers("max", args)
    return max(numbers)


def _mean(*args: Value) -> Value:
    return statistics.mean(_collect_numbers("mean", args))


_FUNCTIONS: dict[str, Callable[..., Value]] = {
    "sqrt": _one_arg(math.sqrt),
    "sin": _one_arg(math.sin),
    "cos": _one_arg(math.cos),
    "tan": _one_arg(math.tan),
    "asin": _one_arg(math.asin),
    "acos": _one_arg(math.acos),
    "atan": _one_arg(math.atan),
    "log": _log,
    "log10": _one_arg(math.log10),
    "exp": _one_arg(math.exp),
    "floor": _one_arg(math.floor),
    "ceil": _one_arg(math.ceil),
    "round": _round,
    "abs": _one_arg(abs),
    "sum": _sum,
    "min": _min,
    "max": _max,
    "mean": _mean,
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
        value = _evaluate_expression(expression)
    except ArithmeticError as error:
        raise ValueError(f"math error: {error}") from error
    if not isinstance(value, int | float):
        raise ValueError("expression must return a number")
    if not math.isfinite(value):
        raise ValueError("result is not finite")
    return _format_number(value)


def _evaluate_expression(expression: str) -> Value:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("expression must be valid calculator syntax") from error
    return _evaluate_node(parsed.body)


def _evaluate_node(node: ast.AST) -> Value:
    if isinstance(node, ast.Constant):
        return _evaluate_constant(node.value)
    if isinstance(node, ast.Name):
        return _evaluate_name(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _require_number(_evaluate_node(node.operand))
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError("unsupported unary operator")
        return operator(operand)
    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"unsupported operator {_operator_text(node.op)!r}")
        left = _require_number(_evaluate_node(node.left))
        right = _require_number(_evaluate_node(node.right))
        return operator(left, right)
    if isinstance(node, ast.Call):
        return _evaluate_call(node)
    if isinstance(node, ast.List):
        return [_require_number(_evaluate_node(element)) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_require_number(_evaluate_node(element)) for element in node.elts)
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _evaluate_constant(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("only numeric literals are supported")
    return value


def _evaluate_name(name: str) -> Number:
    value = _CONSTANTS.get(name)
    if value is None:
        raise ValueError(f"unknown name {name!r}")
    return value


def _evaluate_call(node: ast.Call) -> Value:
    if not isinstance(node.func, ast.Name):
        raise ValueError("function must be a simple name")
    if node.keywords:
        raise ValueError("keyword arguments are not supported")
    function = _FUNCTIONS.get(node.func.id)
    if function is None:
        raise ValueError(f"unknown function {node.func.id!r}")
    args = [_evaluate_node(argument) for argument in node.args]
    return function(*args)


def _require_number(value: Value) -> Number:
    if isinstance(value, int | float):
        return value
    raise ValueError("expected a number")


def _collect_numbers(name: str, args: Sequence[Value]) -> list[Number]:
    if len(args) == 1 and isinstance(args[0], list | tuple):
        values = list(args[0])
    else:
        values = list(args)
    if not values:
        raise ValueError(f"{name} expects at least one number")
    numbers = [_require_number(value) for value in values]
    return numbers


def _format_number(value: Number) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.15g}"


def _operator_text(operator: ast.operator) -> str:
    if isinstance(operator, ast.BitXor):
        return "^"
    return type(operator).__name__
