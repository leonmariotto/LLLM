import pytest

from ..LLLM.generator_with_tool import Tool
from ..LLLM.tool_compute import compute_tool, execute_compute


def compute(expression: str) -> str:
    return execute_compute({"expression": expression})


def test_compute_tool_returns_registered_tool() -> None:
    tool = compute_tool()

    assert isinstance(tool, Tool)
    assert tool.schema["type"] == "function"
    function = tool.schema["function"]
    assert isinstance(function, dict)
    assert function["name"] == "compute"
    assert tool.execute({"expression": "2 + 2"}) == "4"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("-5 + 2", "-3"),
        ("2 ** 10", "1024"),
        ("sqrt(81)", "9"),
        ("sin(pi / 2)", "1"),
        ("sum([1, 2, 3])", "6"),
        ("mean([2, 4, 6])", "4"),
        ("max([1, 5, 3])", "5"),
        ("round(1 / 3, 2)", "0.33"),
    ],
)
def test_execute_compute_evaluates_supported_expressions(
    expression: str,
    expected: str,
) -> None:
    assert compute(expression) == expected


def test_execute_compute_formats_numbers_readably() -> None:
    assert compute("6 / 2") == "3"
    assert compute("1 / 8") == "0.125"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"expression": 12},
        {"expression": ""},
        {"expression": "   "},
    ],
)
def test_execute_compute_validates_expression_argument(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="expression"):
        execute_compute(arguments)


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        ("unknown(2)", "unknown function"),
        ("x + 1", "unknown name"),
        ("math.sqrt(4)", "function must be a simple name"),
        ("[1, 2][0]", "unsupported syntax"),
        ("[x for x in [1, 2]]", "unsupported syntax"),
        ("2 if 1 else 3", "unsupported syntax"),
        ("2 < 3", "unsupported syntax"),
        ("import math", "valid calculator syntax"),
        ("2^10", r"unsupported operator '\^'"),
    ],
)
def test_execute_compute_rejects_unsupported_syntax(
    expression: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compute(expression)


@pytest.mark.parametrize("expression", ["1 / 0", "sqrt(-1)", "exp(1000)"])
def test_execute_compute_reports_math_failures(expression: str) -> None:
    with pytest.raises(ValueError):
        compute(expression)
