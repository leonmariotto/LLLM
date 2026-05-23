from pathlib import Path

import pytest

from ..LLLM.generator import Generator
from ..LLLM.coder import Coder, CodeSelfConsistencyGenerator
from ..LLLM.fetch import fetch_model_ir
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

    generator_base = Generator(
        model=model,
        tokenizer=tokenizer,
    )
    generator = CodeSelfConsistencyGenerator(
        generator_base,
        encode_prompt=lambda prompt: tokenizer.encode_instruct_prompt(
            prompt,
            enable_thinking=False,
        ),
    )
    coder = Coder(generator)
    task = (
        #"Write a C program that print 42 on stdout."
        #"Write a C program that print it's own source code (quine)."
        "Write a C program that print the number of ac arguments."
    )

    result = coder.solve(task)

    assert result is not None

    print("Generated and selected Qwen2 program:\n" + result.selected_candidate.source)
