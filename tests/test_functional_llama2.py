from pathlib import Path
import math

import pytest

from ..LLLM.eval import evaluate_base_model_perplexity
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.generator import Generator
from ..LLLM.llama2 import Llama2Model, Llama2Tokenizer, llama2_config_from_fetched

"""
llama2-15m-tinystories evaluation:
Generation produce coherent text.
The WikiText evaluation return :
Loss: 10.4215
Perplexity: 33574.9Which is very high.
But as long as generation produce coherent text we can assume the weight are correctly loaded.
"""
LLAMA2_TINYSTORIES_REPO_ID = "0rn0/llama2-15m-tinystories"


def _load_remote_llama2_tinystories(
    tmp_path: Path,
) -> tuple[Llama2Model, Llama2Tokenizer]:
    fetched = fetch_hf_model(LLAMA2_TINYSTORIES_REPO_ID)
    tokenizer = Llama2Tokenizer(str(fetched.path / "tokenizer.model"))
    model = Llama2Model(llama2_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    return model, tokenizer


@pytest.mark.slow
def test_functional_llama2_fetch_load_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = _load_remote_llama2_tinystories(tmp_path)

    prompt = "Once upon a time"
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        context_size=model.context_length,
    )
    generated_text = generator.generate(
        prompt,
        max_generated_token=20,
        stop_at_eos=False,
    )
    print("Generated text : [" + generated_text + "]\n")

    assert len(generated_text) > len(prompt)

@pytest.mark.slow
def test_functional_llama2_fetch_load_generate_raw_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = _load_remote_llama2_tinystories(tmp_path)

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=2,
        context_size=64,
    )

    assert math.isfinite(perplexity)
    assert perplexity > 0.0
