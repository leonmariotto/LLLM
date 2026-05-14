from pathlib import Path

import pytest

from ..LLLM.fetch import fetch_hf_model
from ..LLLM.generator import Generator
from ..LLLM.gpt import GPT2Tokenizer, GPTModel, gpt_config_from_fetched


PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"


def _generate_20_tokens_from_fetched_model(repo_id: str) -> str:
    fetched = fetch_hf_model(
        repo_id,
    )
    tokenizer = GPT2Tokenizer()
    model = GPTModel(gpt_config_from_fetched(fetched.config))
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
