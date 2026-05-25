from pathlib import Path

import pytest

from ..LLLM.generator import Generator
from ..LLLM.planner import Planner, PlannerGenerationOptions
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_planner_generator() -> Generator:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    return Generator(model=model, tokenizer=tokenizer, cache_length=16384)


@pytest.mark.slow
def test_functional_planner_generates_summary_and_plan(
    qwen3_planner_generator: Generator,
) -> None:
    planner = Planner(
        qwen3_planner_generator,
        options=PlannerGenerationOptions(max_generated_token=512),
    )
    request = "Design a small in-memory cache with item expiry."

    expansions = planner.generate_expansions(request, expansion_count=2)
    summary = planner.synthesize_summary(request, expansions)
    task_plan = planner.generate_task_plan(request, summary)

    assert len(expansions) == 2
    assert all(expansion.strip() for expansion in expansions)
    assert summary.strip()
    assert task_plan.strip()
