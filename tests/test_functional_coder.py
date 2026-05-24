from pathlib import Path

import pytest

from ..LLLM.generator import Generator
from ..LLLM.coder import (
    CodeCandidate,
    Coder,
    CompileResult,
)
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.qwen2 import Qwen2Model, Qwen2Tokenizer
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer

QWEN2_5_CODER_05B_INSTRUCT_REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"

@pytest.fixture(scope="module")
def qwen2_5_coder_generator() -> Generator:
    ir = fetch_model_ir(QWEN2_5_CODER_05B_INSTRUCT_REPO_ID)
    cfg = Qwen2Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen2Tokenizer(str(path / "tokenizer.json"))
    model = Qwen2Model(cfg)
    model.load_ir_weights(ir)

    return Generator(
        model=model,
        tokenizer=tokenizer,
        cache_length=16384
    )

@pytest.fixture(scope="module")
def qwen3_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)

    return Generator(
        model=model,
        tokenizer=tokenizer,
        cache_length=16384
    )

@pytest.mark.slow
def test_functional_qwen2_5_coder_generates_addition_function(
    qwen2_5_coder_generator: Generator,
    qwen3_generator: Generator,
) -> None:
    coder = Coder(
        qwen2_5_coder_generator,
        qwen3_generator,
        code_top_p=0.90,
    )
    task = (
        #"Write a C program that print the number of ac arguments."
        #"Write a C program that print its first parameter."
        "Write a C program that print all its argv parameter in reverse order."
    )

    result = coder.solve(task)

    assert result is not None

    print("Generated and selected Qwen2 program:\n" + result.selected_candidate.source)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("candidate_a_source", "candidate_b_source", "expected_winner_side"),
    [
        (
            """
#include <stdio.h>

int main(void) {
    printf("hello\\n");
    return 0;
}
""",
            """
#include <stdio.h>

int main(void) {
    printf("wrong message\\n");
    return 0;
}
""",
            "A",
        ),
        (
            """
#include <stdio.h>

int main(void) {
    printf("wrong message\\n");
    return 0;
}
""",
            """
#include <stdio.h>

int main(void) {
    printf("hello\\n");
    return 0;
}
""",
            "B",
        ),
    ],
)
def test_functional_qwen2_5_coder_judge_only_picks_matching_output(
    qwen2_5_coder_generator: Generator,
    qwen3_generator: Generator,
    candidate_a_source: str,
    candidate_b_source: str,
    expected_winner_side: str,
) -> None:
    task = 'Write a C program that prints exactly "hello" followed by a newline.'
    candidate_a = CodeCandidate(0, "judge-only", "raw", candidate_a_source)
    candidate_b = CodeCandidate(1, "judge-only", "raw", candidate_b_source)
    compile_results = (
        CompileResult(0, True, ("judge-only",), 0, "", ""),
        CompileResult(1, True, ("judge-only",), 0, "", ""),
    )
    coder = Coder(qwen2_5_coder_generator, qwen3_generator)

    judge_results = coder.judge_successful_candidate_tournament(
        task,
        (candidate_a, candidate_b),
        compile_results,
    )

    assert len(judge_results) == 1
    judge_result = judge_results[0]
    print(
        "\nJudge-only Qwen3 output "
        f"(expected {expected_winner_side}):\n{judge_result.raw_output}"
    )
    assert judge_result.winner_candidate_index == (
        candidate_a.index if expected_winner_side == "A" else candidate_b.index
    )
