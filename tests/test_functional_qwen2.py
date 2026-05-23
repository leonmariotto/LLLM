from pathlib import Path

import pytest

from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen2 import Qwen2Model, Qwen2Tokenizer


QWEN2_5_CODER_05B_INSTRUCT_REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


@pytest.mark.slow
def test_functional_qwen2_5_coder_generates_addition_function() -> None:
    ir = fetch_model_ir(QWEN2_5_CODER_05B_INSTRUCT_REPO_ID)
    cfg = Qwen2Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen2Tokenizer(str(path / "tokenizer.json"))
    model = Qwen2Model(cfg)
    model.load_ir_weights(ir)

    generator = Generator(
        model=model,
        tokenizer=tokenizer,
    )
    prompt_tokens = tokenizer.encode_instruct_prompt(
        "Write a Python function named add that takes two arguments and returns "
        "their sum. Return only the code.",
        enable_thinking=False,
    )

    generated_text = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=80,
        include_prompt=False,
    )
    print("Generated Qwen2 addition function:\n" + generated_text)

    assert generated_text.strip()
