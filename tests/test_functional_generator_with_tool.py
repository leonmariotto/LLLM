from pathlib import Path

import pytest

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.generator_with_tool import GeneratorWithTool, Tool
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.tool_compute import compute_tool, execute_compute
from ..LLLM.tool_wikisearch import wikisearch_tool


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_generator_with_tool() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.mark.slow
def test_functional_qwen3_calls_hello_tool_and_uses_response(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []

    def hello(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        return "Hello Tool !"

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [
            Tool(
                schema={
                    "type": "function",
                    "function": {
                        "name": "hello",
                        "description": "Return the greeting required to answer.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
                execute=hello,
            )
        ],
        max_tool_rounds=2,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "You must call the hello function exactly once. Reply first "
                    "only with a tool call in the required <tool_call></tool_call> "
                    "XML format. Do not write JSON without the XML tags and do "
                    "not invent the result. After the tool returns, answer with "
                    "its response exactly and no additional text."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert calls == [{}]
    assert "Hello Tool !" in response


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_compute_tool(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []
    results: list[str] = []
    base_tool = compute_tool()

    def record_compute(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        result = execute_compute(arguments)
        results.append(result)
        return result

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [Tool(schema=base_tool.schema, execute=record_compute)],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "Use the compute tool exactly once to calculate 137 * 29. "
                    "The compute tool expression must use bc syntax. "
                    "First reply only with a valid <tool_call></tool_call> block. "
                    "After the tool response, answer with the numeric result and "
                    "no extra explanation."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert calls
    assert results == ["3973"]
    assert "3973" in response


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wikisearch_tool(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = wikisearch_tool()

    def record_wikisearch(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        return (
            "1. Frobnicate\n"
            "URL: https://en.wikipedia.org/wiki/Frobnicate\n"
            "Snippet: The controlled wikisearch tool response says frobnicate."
        )

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [Tool(schema=base_tool.schema, execute=record_wikisearch)],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "Use the wikisearch tool exactly once to search for "
                    "'frobnicate'. First reply only with a valid "
                    "<tool_call></tool_call> block. After the tool response, "
                    "answer with the URL from the result and the word frobnicate."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert calls
    assert calls[0].get("action") == "search"
    query = calls[0].get("query")
    assert isinstance(query, str)
    assert "frobnicate" in query.lower()
    assert "https://en.wikipedia.org/wiki/Frobnicate" in response
    assert "frobnicate" in response


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wikisearch_open_tool(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = wikisearch_tool()

    def record_wikisearch(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        return (
            "URL: https://en.wikipedia.org/wiki/CAC_40\n"
            "Title: CAC 40\n\n"
            "The controlled wiki page extract says calisson."
        )

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [Tool(schema=base_tool.schema, execute=record_wikisearch)],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "Use the wikisearch tool exactly once to open "
                    "https://en.wikipedia.org/wiki/CAC_40. First reply only "
                    "with a valid <tool_call></tool_call> block. After the "
                    "tool response, answer with the page title and the word "
                    "calisson."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert calls
    assert calls[0].get("action") == "open"
    assert calls[0].get("url") == "https://en.wikipedia.org/wiki/CAC_40"
    assert "CAC 40" in response
    assert "calisson" in response


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_compute_tool_autoload(
    qwen3_generator_with_tool: Generator,
) -> None:

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [compute_tool()],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "I want to measure the surface of an orb that have 22.2 "
                    "centimeter of diameter. What's the surface of the orb in "
                    "square centimeter ? Use bc syntax when calling the compute "
                    "tool."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert response is not None


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wikisearch_autoload(
    qwen3_generator_with_tool: Generator,
) -> None:

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [wikisearch_tool()],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "What's the CAC40 latest market cap ? I believe that this "
                    "information is present in wikipedia."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.0,
    )

    assert response is not None
