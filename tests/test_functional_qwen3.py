import math
from pathlib import Path

import pytest

from ..LLLM.eval import evaluate_base_model_perplexity
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.qwen3 import Qwen3Model, Qwen3Tokenizer


QWEN3_06B_REPO_ID = "Qwen/Qwen3-0.6B"


@pytest.mark.slow
def test_functional_qwen3_06b_wikitext_perplexity() -> None:
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

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=10,
        context_size=128,
    )

    assert math.isfinite(perplexity)
    assert perplexity < 200.0
