from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.agent import Agent
from ..LLLM.agent_llm import LlmClient
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.tool_common import Tool
from ..LLLM.tool_wiki import wiki_tool

QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


def wiki_agent(
    generator: Generator,
    tools: list[Tool],
    *,
    max_step: int = 3,
) -> Agent:
    return Agent(
        LlmClient(
            generator,
            max_generated_token=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
        tools,
        max_step=max_step,
    )


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wiki_tool(
    qwen3_generator: Generator,
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

    agent = wiki_agent(
        qwen3_generator,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
    )

    response = agent.run(
        "Use the wiki tool exactly once to search for "
        "'frobnicate'. First reply only with a valid "
        "<tool_call></tool_call> block. After the tool response, "
        "answer with the URL from the result and the word frobnicate."
    )

    assert calls
    assert calls[0].get("action") == "search"
    query = calls[0].get("query")
    assert isinstance(query, str)
    assert "frobnicate" in query.lower()
    assert isinstance(response.output, str)
    assert "https://en.wikipedia.org/wiki/Frobnicate" in response.output
    assert "frobnicate" in response.output


@pytest.mark.slow
def test_functional_qwen3_with_thinking_calls_wiki_open_tool(
    qwen3_generator: Generator,
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

    agent = wiki_agent(
        qwen3_generator,
        [Tool(schema=base_tool.schema, execute=record_wiki)],
    )

    response = agent.run(
        "Use the wiki tool exactly once to open "
        "https://en.wikipedia.org/wiki/CAC_40. First reply only "
        "with a valid <tool_call></tool_call> block. After the "
        "tool response, answer with the page title and the word "
        "calisson. "
        "The tool name is name=wiki and in parameters action=open."
    )

    assert calls
    assert calls[0].get("action") == "open"
    assert calls[0].get("url") == "https://en.wikipedia.org/wiki/CAC_40"
    assert isinstance(response.output, str)
    assert "CAC 40" in response.output
    assert "calisson" in response.output


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
            None,
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
    qwen3_generator: Generator,
    prompt: str,
    expected_in_response: str | None,
) -> None:
    agent = wiki_agent(
        qwen3_generator,
        [wiki_tool()],
        max_step=8,
    )

    response = agent.run(prompt)

    assert isinstance(response.output, str)
    if expected_in_response is not None:
        assert expected_in_response in response.output
