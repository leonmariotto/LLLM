from pathlib import Path
import math

import pytest

from ..LLLM.eval import (
    evaluate_base_model_perplexity,
)
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.gpt2 import GeneratorGPT2, GPT2Tokenizer, GPT2Model

PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"

def _load_local_gpt2(tmp_path: Path) -> tuple[GPT2Model, GPT2Tokenizer]:
    ir = fetch_model_ir(
        str(PREFETCHED_GPT2_PATH),
    )
    tokenizer = GPT2Tokenizer()
    model = GPT2Model(GPT2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    return model, tokenizer


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


def _generate_20_tokens_from_fetched_model(repo_id: str) -> str:
    ir = fetch_model_ir(
        repo_id,
    )
    tokenizer = GPT2Tokenizer()
    model = GPT2Model(GPT2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    model.eval()

    prompt = "Every effort moves the project forward."
    generator = GeneratorGPT2(
        model=model,
        tokenizer=tokenizer,
        context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
    )
    generated_text = generator.generate(prompt, max_generated_token=20)
    print("Generated text : [" + generated_text + "]\n")

    assert generated_text.startswith(prompt)
    assert len(generated_text) > len(prompt)
    return str(ir.metadata.get("path", repo_id))

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
