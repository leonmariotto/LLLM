"""
Run GAIA dataset tests.
Warning: a valid HF token must be set in environemnt var HF_TOKEN.
"""

import math
from pathlib import Path
from typing import cast

from loguru import logger

import pytest

pytestmark = pytest.mark.slow

from ..LLLM.eval_gaia import GaiaTask, evaluate_gaia_agent
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.generator_with_tool import GeneratorWithTool
from ..LLLM.tool_compute import compute_tool


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

@pytest.fixture(scope="module")
def qwen3_06b_gaia_generator_with_compute() -> GeneratorWithTool:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    qwen3_generator = Generator(model=model, tokenizer=tokenizer, cache_length=16384)
    tool_generator = GeneratorWithTool(
            qwen3_generator,
            [compute_tool()]
    )
    return tool_generator


@pytest.mark.slow
def test_functional_qwen3_06b_no_harness_gaia_validation(
    qwen3_06b_gaia_generator: Generator,
) -> None:
    tokenizer = cast(Qwen3Tokenizer, qwen3_06b_gaia_generator.tokenizer)

    def agent(task: GaiaTask) -> str:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else "\nNo attached file is available for this task."
        )
        #TODO: GAIA output format should be part of gaia contract and derived
        #from a pydantic type.
        # TODO add informations like: is_solvable: bool and unsolvability_reason:str
        prompt = (
            "Answer this GAIA benchmark question. Return only the final answer "
            "using this exact format: FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}"
        )
        prompt_tokens = tokenizer.encode_instruct_prompt( prompt,
            enable_thinking=True,
        )
        return qwen3_06b_gaia_generator.generate_from_tokens(
            prompt_tokens,
            max_generated_token=4096,
            temperature=0.0,
            include_prompt=False,
        ).strip()

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        level=1,
        limit=20,
    )
    logger.info("evaluation={}", evaluation)

    #assert evaluation.total_tasks == 1
    #assert evaluation.scored_tasks == 1
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0

    result = evaluation.results[0]
    assert result.task_id
    assert result.question
    assert result.prediction
    assert result.error is None

@pytest.mark.slow
def test_functional_qwen3_06b_with_compute_gaia_validation(
    qwen3_06b_gaia_generator_with_compute: GeneratorWithTool,
) -> None:
    tokenizer = cast(Qwen3Tokenizer, qwen3_06b_gaia_generator_with_compute.tokenizer)

    def agent(task: GaiaTask) -> str:
        attachment_note = (
            f"\nAttached file path: {task.file_path}"
            if task.file_path is not None
            else "\nNo attached file is available for this task."
        )
        #TODO: GAIA output format should be part of gaia contract and derived
        #from a pydantic type.
        # TODO add informations like: is_solvable: bool and unsolvability_reason:str
        prompt = (
            "Answer this GAIA benchmark question. Return only the final answer "
            "using this exact format: FINAL ANSWER: <answer>\n\n"
            f"Question: {task.question}"
            f"{attachment_note}"
        )
        prompt_tokens = tokenizer.encode_instruct_prompt(
            prompt,
            enable_thinking=True,
        )
        return qwen3_06b_gaia_generator_with_compute.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "Answer this GAIA benchmark question. Use the compute tool when "
                        "arithmetic or exact calculation is needed. Return only the final "
                        "answer using this exact format: FINAL ANSWER: <answer>\n\n"
                        f"Question: {task.question}"
                        f"{attachment_note}"
                    ),
                }
            ],
            max_generated_token=4096,
            temperature=0.0,
        ).strip()

    evaluation = evaluate_gaia_agent(
        agent,
        split="validation",
        level=1,
        limit=20,
    )
    logger.info("evaluation={}", evaluation)

    #assert evaluation.total_tasks == 1
    #assert evaluation.scored_tasks == 1
    assert evaluation.overall_accuracy is not None
    assert math.isfinite(evaluation.overall_accuracy)
    assert 0.0 <= evaluation.overall_accuracy <= 1.0

    result = evaluation.results[0]
    assert result.task_id
    assert result.question
    assert result.prediction
    assert result.error is None
