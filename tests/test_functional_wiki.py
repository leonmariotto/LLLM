from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.generator_with_tool import GeneratorWithTool, Tool
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.tool_wiki import wiki_tool

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
def test_functional_qwen3_with_thinking_calls_wiki_tool(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = wiki_tool()

    def record_wiki(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        return (
            "1. Frobnicate\n"
            "URL: https://en.wikipedia.org/wiki/Frobnicate\n"
            "Snippet: The controlled wiki tool response says frobnicate."
        )

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "Use the wiki tool exactly once to search for "
                    "'frobnicate'. First reply only with a valid "
                    "<tool_call></tool_call> block. After the tool response, "
                    "answer with the URL from the result and the word frobnicate."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    assert calls
    assert calls[0].get("action") == "search"
    query = calls[0].get("query")
    assert isinstance(query, str)
    assert "frobnicate" in query.lower()
    assert "https://en.wikipedia.org/wiki/Frobnicate" in response
    assert "frobnicate" in response


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wiki_open_tool(
    qwen3_generator_with_tool: Generator,
) -> None:
    calls: list[dict[str, object]] = []
    base_tool = wiki_tool()

    def record_wiki(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        return (
            "URL: https://en.wikipedia.org/wiki/CAC_40\n"
            "Title: CAC 40\n\n"
            "The controlled wiki page extract says calisson."
        )

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
        max_tool_rounds=3,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": (
                    "Use the wiki tool exactly once to open "
                    "https://en.wikipedia.org/wiki/CAC_40. First reply only "
                    "with a valid <tool_call></tool_call> block. After the "
                    "tool response, answer with the page title and the word "
                    "calisson."
                ),
            }
        ],
        max_generated_token=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    assert calls
    assert calls[0].get("action") == "open"
    assert calls[0].get("url") == "https://en.wikipedia.org/wiki/CAC_40"
    assert "CAC 40" in response
    assert "calisson" in response

@pytest.mark.slow
@pytest.mark.parametrize(
    ("prompt", "expected_in_response"),
    [
        pytest.param(
            (
                "What's the CAC40 latest market cap ? I believe that this "
                "information is present in wikipedia. Keep trying to use "
                "wiki until you got the response."
            ),
            "370,437,433,957.70",
            id="cac40-market-cap",
        ),
        pytest.param(
            (
                "According to Wikipedia, which city hosted the 2024 Summer "
                "Olympics? Keep trying to use wiki until you got the "
                "response."
            ),
            "Paris",
            id="2024-summer-olympics-host-city",
        ),
    ],
)
def test_functional_qwen3_with_thinking_calls_wiki_autoload(
    qwen3_generator_with_tool: Generator,
    prompt: str,
    expected_in_response: str,
) -> None:

    tool_generator = GeneratorWithTool(
        qwen3_generator_with_tool,
        [wiki_tool()],
        max_tool_rounds=8,
    )

    response = tool_generator.generate(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_generated_token=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    assert response is not None
    assert expected_in_response in response
