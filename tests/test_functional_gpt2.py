from pathlib import Path
import math

import pytest

from ..LLLM.eval import (
    DatasetAdapter,
    boolq_adapter,
    evaluate_base_model_perplexity,
    evaluate_instructions_model,
    gsm8k_adapter,
    squad_adapter,
)
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.generator import Generator
from ..LLLM.gpt2 import GPT2Tokenizer, GPT2Model, gpt2_config_from_fetched

PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"

def _load_local_gpt2(tmp_path: Path) -> tuple[GPT2Model, GPT2Tokenizer]:
    fetched = fetch_hf_model(
        str(PREFETCHED_GPT2_PATH),
    )
    tokenizer = GPT2Tokenizer()
    model = GPT2Model(gpt2_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    return model, tokenizer


def _load_remote_gpt2_instruct(
    tmp_path: Path,
) -> tuple[GPT2Model, GPT2Tokenizer]:
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
    model = GPT2Model(gpt2_config_from_fetched(fetched.config))
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

def _generate_20_tokens_from_fetched_model(repo_id: str) -> str:
    fetched = fetch_hf_model(
        repo_id,
    )
    tokenizer = GPT2Tokenizer()
    model = GPT2Model(gpt2_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    model.eval()

    prompt = "Every effort moves the project forward."
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
    )
    generated_text = generator.generate(prompt, max_generated_token=20)
    print("Generated text : [" + generated_text + "]\n")

    assert generated_text.startswith(prompt)
    assert len(generated_text) > len(prompt)
    return str(fetched.path)

@pytest.mark.slow
def test_functional_gpt2_fetch_load_generate_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    _generate_20_tokens_from_fetched_model(
        "openai-community/gpt2",
    )


def test_functional_gpt2_load_local_snapshot_generate_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_20_tokens_from_fetched_model(
        str(PREFETCHED_GPT2_PATH),
    )
