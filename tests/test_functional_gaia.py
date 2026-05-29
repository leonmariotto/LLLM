"""
Run GAIA dataset tests.
Warning: a valid HF token must be set in environemnt var HF_TOKEN.
"""

import math
from pathlib import Path
from typing import cast

import pytest

from ..LLLM.eval_gaia import GaiaTask, evaluate_gaia_agent
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_06b_gaia_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.mark.slow
def test_functional_qwen3_06b_runs_real_gaia_validation_smoke(
    qwen3_06b_gaia_generator: Generator,
) -> None:
    tokenizer = cast(Qwen3Tokenizer, qwen3_06b_gaia_generator.tokenizer)

    def agent(task: GaiaTask) -> str:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else "\nNo attached file is available for this task."
        )
        prompt = (
            "Answer this GAIA benchmark question. Return only the final answer "
            "using this exact format: FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}"
        )
        prompt_tokens = tokenizer.encode_instruct_prompt(
            prompt,
            enable_thinking=False,
        )
        return qwen3_06b_gaia_generator.generate_from_tokens(
            prompt_tokens,
            max_generated_token=256,
            temperature=0.0,
            include_prompt=False,
        ).strip()

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        level=1,
        limit=1,
    )

    assert evaluation.total_tasks == 1
    assert evaluation.scored_tasks == 1
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0

    result = evaluation.results[0]
    assert result.task_id
    assert result.question
    assert result.prediction
    assert result.error is None
