from typing import Any, cast

import pytest
import torch
from transformers import AutoTokenizer

from ..LLLM.fetch import fetch_hf_model
from ..LLLM.gemma3 import Gemma3Model, gemma3_config_from_fetched


GEMMA3_1B_IT_REPO_ID = "google/gemma-3-1b-it"


@pytest.mark.slow
def test_functional_gemma3_1b_it_loads_and_runs_real_hf_checkpoint() -> None:
    fetched = fetch_hf_model(GEMMA3_1B_IT_REPO_ID)
    cfg = gemma3_config_from_fetched(fetched.config)
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        fetched.path, local_files_only=True
    )
    model = Gemma3Model(cfg)
    model.load_fetched_model(fetched)
    del fetched

    encoded = tokenizer(
        "What is 2 + 2? Answer with one number.",
        return_tensors="pt",
    )
    input_ids = cast(torch.Tensor, encoded["input_ids"])

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], cfg["vocab_size"])
    assert torch.isfinite(logits).all()
