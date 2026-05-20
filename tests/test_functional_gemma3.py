import math
from typing import Any, cast

import pytest
import torch
from transformers import AutoTokenizer

from ..LLLM.eval import (
    DatasetAdapter,
    boolq_adapter,
    boolq_prediction,
    evaluate_instructions_model,
    extract_last_number,
    gsm8k_adapter,
    squad_adapter,
    squad_score,
)
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.gemma3 import Gemma3Model, Gemma3Tokenizer


GEMMA3_1B_IT_REPO_ID = "google/gemma-3-1b-it"
GEMMA3_1B_IT_QAT_GGUF_REPO_ID = "lmstudio-community/gemma-3-1B-it-qat-GGUF"
GEMMA3_1B_IT_QAT_Q4_FILE = "gemma-3-1B-it-QAT-Q4_0.gguf"


def _gemma3_encode_instruction_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return cast(list[int], input_ids.squeeze(0).tolist())
    return cast(list[int], input_ids)


def _gemma3_eos_token(tokenizer: Any) -> int | None:
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn >= 0:
        return end_of_turn
    eos_token_id = tokenizer.eos_token_id
    return eos_token_id if isinstance(eos_token_id, int) else None


gemma3_boolq_adapter = DatasetAdapter(
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
    encode_prompt=_gemma3_encode_instruction_prompt,
    eos_token=_gemma3_eos_token,
)


gemma3_squad_adapter = DatasetAdapter(
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
    encode_prompt=_gemma3_encode_instruction_prompt,
    eos_token=_gemma3_eos_token,
)


gemma3_gsm8k_adapter = DatasetAdapter(
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
    encode_prompt=_gemma3_encode_instruction_prompt,
    eos_token=_gemma3_eos_token,
)


@pytest.fixture(scope="module")
def gemma3_1b_it_model_and_tokenizer() -> tuple[Gemma3Model, Any]:
    ir = fetch_model_ir(GEMMA3_1B_IT_REPO_ID)
    cfg = Gemma3Model.config_from_ir(ir)
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        ir.metadata["path"], local_files_only=True
    )
    model = Gemma3Model(cfg)
    model.load_ir_weights(ir)
    del ir
    return model, tokenizer


@pytest.mark.slow
def test_functional_gemma3_1b_it_loads_and_runs_real_hf_checkpoint(
    gemma3_1b_it_model_and_tokenizer: tuple[Gemma3Model, Any],
) -> None:
    model, tokenizer = gemma3_1b_it_model_and_tokenizer
    encoded = tokenizer(
        "What is 2 + 2? Answer with one number.",
        return_tensors="pt",
    )
    input_ids = cast(torch.Tensor, encoded["input_ids"])

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], model.out_head.out_features)
    assert torch.isfinite(logits).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("adapter", "max_generated_token"),
    [
        pytest.param(gemma3_boolq_adapter, 4, id="boolq"),
        pytest.param(gemma3_squad_adapter, 16, id="squad"),
        pytest.param(gemma3_gsm8k_adapter, 64, id="gsm8k"),
    ],
)
def test_functional_gemma3_1b_it_runs_instruction_eval(
    gemma3_1b_it_model_and_tokenizer: tuple[Gemma3Model, Any],
    adapter: DatasetAdapter[Any, Any],
    max_generated_token: int,
) -> None:
    model, tokenizer = gemma3_1b_it_model_and_tokenizer

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        limit=5,
        max_generated_token=max_generated_token,
        context_size=2048,
    )

    assert math.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.slow
def test_functional_gemma3_qat_gguf_q4_quantized_loads_tokenizer_and_runs() -> None:
    ir = fetch_model_ir(
        GEMMA3_1B_IT_QAT_GGUF_REPO_ID,
        gguf_filename=GEMMA3_1B_IT_QAT_Q4_FILE,
        weight_mode="quantized",
    )
    cfg = Gemma3Model.config_from_ir(ir)

    assert cfg["emb_dim"] == 1152
    assert cfg["n_layers"] == 26
    assert cfg["n_heads"] == 4
    assert cfg["n_kv_groups"] == 1

    tokenizer = Gemma3Tokenizer(str(ir.metadata["path"]))
    input_ids = torch.tensor(
        [tokenizer.encode_instruct_prompt("What is 2 + 2? Answer with one number.")],
        dtype=torch.long,
    )
    model = Gemma3Model(cfg, weight_mode="quantized")
    model.load_ir_weights(ir)
    del ir

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], model.out_head.out_features)
    assert torch.isfinite(logits).all()

@pytest.mark.slow
@pytest.mark.parametrize(
    ("adapter", "max_generated_token"),
    [
        pytest.param(gemma3_boolq_adapter, 4, id="boolq"),
        pytest.param(gemma3_squad_adapter, 16, id="squad"),
        pytest.param(gemma3_gsm8k_adapter, 64, id="gsm8k"),
    ],
)
def test_functional_gemma3_qat_gguf_q4_quantized_it_runs_instruction_eval(
    gemma3_1b_it_model_and_tokenizer: tuple[Gemma3Model, Any],
    adapter: DatasetAdapter[Any, Any],
    max_generated_token: int,
) -> None:
    model, tokenizer = gemma3_1b_it_model_and_tokenizer

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        limit=5,
        max_generated_token=max_generated_token,
        context_size=2048,
    )

    assert math.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0

