from pathlib import Path

import pytest

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.generator_with_tool import GeneratorWithTool, Tool
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


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
        max_generated_token=80,
        temperature=0.0,
    )

    assert calls == [{}]
    assert "Hello Tool !" in response
