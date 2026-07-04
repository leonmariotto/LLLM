import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from ..LLLM.eval import (
    DatasetAdapter,
    boolq_adapter,
    boolq_prediction,
    evaluate_base_model_perplexity,
    evaluate_instructions_model,
    extract_last_number,
    gsm8k_adapter,
    squad_adapter,
    squad_score,
)
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer
from ..LLLM.utils import get_device

pytestmark = pytest.mark.slow

QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"
QWEN3_06B_BASE_REPO_ID = "Qwen/Qwen3-0.6B-base"


class StructuredQwen3Smoke(BaseModel):
    ok: bool
    values: list[str]
    s: str


def _qwen3_encode_instruction_prompt(tokenizer: Any, prompt: str) -> list[int]:
    return tokenizer.encode_instruct_prompt(prompt)


qwen3_boolq_adapter = DatasetAdapter(
    dataset_id=boolq_adapter.dataset_id,
    config=boolq_adapter.config,
    split=boolq_adapter.split,
    build_prompt=lambda row: (
        "Read the passage and answer the question with exactly yes or no.\n\n"
        f"Passage: {row['passage']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=boolq_adapter.extract_expected,
    extract_prediction=boolq_prediction,
    score=boolq_adapter.score,
    encode_prompt=_qwen3_encode_instruction_prompt,
)


qwen3_squad_adapter = DatasetAdapter(
    dataset_id=squad_adapter.dataset_id,
    config=squad_adapter.config,
    split=squad_adapter.split,
    build_prompt=lambda row: (
        "Answer the question using only the context below. "
        "Return the shortest exact answer span.\n\n"
        f"Context: {row['context']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=squad_adapter.extract_expected,
    extract_prediction=lambda text: text.strip(),
    score=squad_score,
    encode_prompt=_qwen3_encode_instruction_prompt,
)


qwen3_gsm8k_adapter = DatasetAdapter(
    dataset_id=gsm8k_adapter.dataset_id,
    config=gsm8k_adapter.config,
    split=gsm8k_adapter.split,
    build_prompt=lambda row: (
        "Solve the math problem. Return only the final numeric answer.\n\n"
        f"Problem: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=gsm8k_adapter.extract_expected,
    extract_prediction=extract_last_number,
    score=gsm8k_adapter.score,
    encode_prompt=_qwen3_encode_instruction_prompt,
)


@pytest.fixture(scope="module")
def qwen3_06b_model_and_tokenizer() -> tuple[Qwen3Model, Qwen3Tokenizer]:
    ir = fetch_model_ir(QWEN3_06B_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    assert cfg["rope_theta"] == 1000000.0
    assert cfg["attention_bias"] is False
    assert cfg["n_layers"] == 28
    assert cfg["n_heads"] == 16
    assert cfg["n_kv_groups"] == 8

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    return model, tokenizer


@pytest.mark.slow
def test_functional_qwen3_06b_generates_constrained_response_format(
    qwen3_06b_model_and_tokenizer: tuple[Qwen3Model, Qwen3Tokenizer],
) -> None:
    model, tokenizer = qwen3_06b_model_and_tokenizer
    generator = Generator(model=model, tokenizer=tokenizer, cache_length=16384)
    prompt_tokens = tokenizer.encode_instruct_prompt(
        f"Return a JSON object matching this schema: {StructuredQwen3Smoke.model_json_schema()}.",
        enable_thinking=False,
    )

    generated_text = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=1024,
        include_prompt=False,
        response_format=StructuredQwen3Smoke,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    parsed = StructuredQwen3Smoke.model_validate_json(generated_text)
    assert isinstance(parsed.ok, bool)


@pytest.mark.slow
def test_functional_qwen3_06b_constrained_response_format_allows_thinking(
    qwen3_06b_model_and_tokenizer: tuple[Qwen3Model, Qwen3Tokenizer],
) -> None:
    model, tokenizer = qwen3_06b_model_and_tokenizer
    generator = Generator(model=model, tokenizer=tokenizer, cache_length=16384)
    prompt_tokens = tokenizer.encode_instruct_prompt(
        "Think very briefly in a <think></think> block, then return only a JSON "
        "object matching this schema after the think block: "
        f"{StructuredQwen3Smoke.model_json_schema()}.",
        enable_thinking=True,
    )

    generated_text = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=2048,
        include_prompt=False,
        response_format=StructuredQwen3Smoke,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    assert generated_text.startswith("<think>")
    assert "</think>" in generated_text
    json_payload = generated_text.rsplit("</think>", maxsplit=1)[-1].strip()
    parsed = StructuredQwen3Smoke.model_validate_json(json_payload)
    assert isinstance(parsed.ok, bool)


@pytest.mark.slow
def test_functional_qwen3_06b_base_wikitext_perplexity() -> None:
    ir = fetch_model_ir(QWEN3_06B_BASE_REPO_ID)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    assert cfg["rope_theta"] == 1000000.0
    assert cfg["attention_bias"] is False
    assert cfg["n_layers"] == 28
    assert cfg["n_heads"] == 16
    assert cfg["n_kv_groups"] == 8

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=10,
        context_length=128,
    )

    assert math.isfinite(perplexity)
    assert perplexity < 200.0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("adapter", "max_generated_token"),
    [
        pytest.param(qwen3_boolq_adapter, 1024, id="boolq"),
        pytest.param(qwen3_squad_adapter, 2048, id="squad"),
        pytest.param(qwen3_gsm8k_adapter, 2048, id="gsm8k"),
    ],
)
def test_functional_qwen3_06b_runs_instruction_eval(
    qwen3_06b_model_and_tokenizer: tuple[Qwen3Model, Qwen3Tokenizer],
    adapter: DatasetAdapter[Any, Any],
    max_generated_token: int,
) -> None:
    model, tokenizer = qwen3_06b_model_and_tokenizer
    device = get_device()
    model.to(device)

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        limit=1,
        max_generated_token=max_generated_token,
    )

    assert math.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0
