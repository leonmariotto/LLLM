from pathlib import Path
from typing import Any, cast

import pytest
import torch

from ..LLLM.fetch import fetch_hf_model
from ..LLLM.gpt import GPT2Tokenizer, GPTModel, gpt_config_from_fetched
from ..LLLM.utils import generate_text_simple


PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"


def _configure_hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "hf-xet-cache"))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")


def _generate_20_tokens_from_fetched_model(repo_id: str, cache_dir: Path) -> str:
    fetched = fetch_hf_model(
        repo_id,
        cache_dir=cache_dir,
    )
    tokenizer = GPT2Tokenizer()
    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    model.eval()

    prompt = "Every effort moves the project forward."
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    with torch.no_grad():
        generated_ids = generate_text_simple(
            model=model,
            idx=input_ids,
            max_new_tokens=20,
            context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
        )

    generated_tokens = cast(list[int], cast(Any, generated_ids.squeeze(0)).tolist())
    generated_text = tokenizer.decode(generated_tokens)
    print("Generated text : [" + generated_text + "]\n")

    assert generated_ids.shape == (1, len(prompt_tokens) + 20)
    assert generated_text.startswith(prompt)
    assert len(generated_text) > len(prompt)
    return str(fetched.path)


# This test will download the GPT2 model. Commented-out for now
# TODO runtime option to activate it or not ?
# def test_functional_gpt2_fetch_load_generate_and_decode(
#     tmp_path: Path,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     _configure_hf_cache(tmp_path, monkeypatch)
# 
#     _generate_20_tokens_from_fetched_model(
#         "openai-community/gpt2",
#         tmp_path / "hf-cache",
#     )


def test_functional_gpt2_load_local_snapshot_generate_and_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_hf_cache(tmp_path, monkeypatch)

    _generate_20_tokens_from_fetched_model(
        str(PREFETCHED_GPT2_PATH),
        tmp_path / "unused-cache",
    )
