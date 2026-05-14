from pathlib import Path
import math

import pytest
import tiktoken

from ..LLLM.eval import (
    DatasetAdapter,
    boolq_adapter,
    evaluate_base_model_perplexity,
    evaluate_instructions_model,
    gsm8k_adapter,
    squad_adapter,
)
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.gpt import GPT2Tokenizer, GPTModel, gpt_config_from_fetched

PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"

def _load_local_gpt2(tmp_path: Path) -> tuple[GPTModel, GPT2Tokenizer]:
    fetched = fetch_hf_model(
        str(PREFETCHED_GPT2_PATH),
    )
    tokenizer = GPT2Tokenizer()
    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    return model, tokenizer


def _load_remote_gpt2_instruct(
    tmp_path: Path,
) -> tuple[GPTModel, GPT2Tokenizer]:
    GPT2_INSTRUCT_REPO_ID = "Sanjarbek1024/gpt2-instruct"
    fetched = fetch_hf_model(
        GPT2_INSTRUCT_REPO_ID,
    )
    tokenizer = GPT2Tokenizer(
            extra_special_tokens={
                "<|user|>": 50257,
                "<|assistant|>": 50258,
        }
    )
    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    return model, tokenizer


@pytest.mark.parametrize(
    "adapter",
    [
        pytest.param(gsm8k_adapter, id="gsm8k"),
        pytest.param(boolq_adapter, id="boolq"),
        pytest.param(squad_adapter, id="squad"),
    ],
)
def test_functional_gpt_eval_runs_dataset_adapter_against_local_model(
    adapter: DatasetAdapter[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _configure_hf_cache(tmp_path, monkeypatch)
    model, tokenizer = _load_local_gpt2(tmp_path)
    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        limit=1,
        max_generated_token=1,
        context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
    )

    assert 0.0 <= accuracy <= 1.0


def test_functional_gpt_eval_runs_wikitext_perplexity_against_local_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _configure_hf_cache(tmp_path, monkeypatch)
    model, tokenizer = _load_local_gpt2(tmp_path)

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=10,
        context_size=64,
    )

    assert math.isfinite(perplexity)
    assert perplexity > 0.0


@pytest.mark.slow
def test_functional_gpt_eval_runs_boolq_against_remote_gpt2_instruct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # # _configure_hf_cache(tmp_path, monkeypatch)
    model, tokenizer = _load_remote_gpt2_instruct(tmp_path)

    boolq_adapter.build_prompt=lambda row: (
        "<|user|>Read the passage and answer the question with only yes or no.\n\n"
        f"Passage: {row['passage']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:\n<|assistant|>\n"
    )
    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=boolq_adapter,
        limit=2,
        max_generated_token=3,
        context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
    )

    assert 0.0 <= accuracy <= 1.0
