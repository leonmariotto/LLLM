from pathlib import Path

import pytest

from ..LLLM.eval import evaluate_instructions_model, gsm8k_adapter
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.gpt import GPT2Tokenizer, GPTModel, gpt_config_from_fetched


PREFETCHED_GPT2_PATH = Path(__file__).parent / "prefetched_models" / "gpt2"


def _configure_hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf_home = tmp_path / "hf-home"
    hf_hub_cache = hf_home / "hub"
    hf_xet_cache = hf_home / "xet"
    hf_datasets_cache = hf_home / "datasets"
    hf_modules_cache = hf_home / "modules"

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_hub_cache))
    monkeypatch.setenv("HF_XET_CACHE", str(hf_xet_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_datasets_cache))
    monkeypatch.setenv("HF_MODULES_CACHE", str(hf_modules_cache))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    import datasets.config
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HOME", str(hf_home))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hf_hub_cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_XET_CACHE", str(hf_xet_cache))
    monkeypatch.setattr(
        huggingface_hub.constants,
        "default_cache_path",
        str(hf_hub_cache),
    )
    monkeypatch.setattr(
        datasets.config,
        "HF_DATASETS_CACHE",
        str(hf_datasets_cache),
    )
    monkeypatch.setattr(
        datasets.config,
        "HF_MODULES_CACHE",
        str(hf_modules_cache),
    )
    monkeypatch.setattr(
        datasets.config,
        "DOWNLOADED_DATASETS_PATH",
        str(hf_datasets_cache / "downloads"),
    )


def test_functional_gpt_eval_runs_gsm8k_against_local_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_hf_cache(tmp_path, monkeypatch)

    fetched = fetch_hf_model(
        str(PREFETCHED_GPT2_PATH),
        cache_dir=tmp_path / "unused-cache",
    )
    tokenizer = GPT2Tokenizer()
    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=gsm8k_adapter,
        limit=1,
        max_generated_token=1,
        context_size=model.pos_emb.num_embeddings if model.pos_emb else 1024,
    )

    assert 0.0 <= accuracy <= 1.0
