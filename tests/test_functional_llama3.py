import math

import pytest
import torch
from transformers import AutoModel

from ..LLLM.eval import evaluate_base_model_perplexity
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.llama3 import Llama3Model, Llama3Tokenizer, llama3_config_from_fetched


LLAMA3_TINY_INSTRUCT_REPO_ID = "AlignmentResearch/Llama-3.3-Tiny-Instruct"
SMOLLM2_135M_REPO_ID = "HuggingFaceTB/SmolLM2-135M"


@pytest.mark.slow
def test_functional_llama3_compatibility_with_reference_implementation() -> None:
    fetched = fetch_hf_model(LLAMA3_TINY_INSTRUCT_REPO_ID)
    cfg = llama3_config_from_fetched(fetched.config)

    assert cfg["context_length"] == 131072
    assert cfg["rope_theta"] == 500000.0
    assert cfg["freq_config"] == {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
    }

    tokenizer = Llama3Tokenizer(str(fetched.path / "tokenizer.model"))
    model = Llama3Model(cfg)
    model.load_fetched_model(fetched)
    reference_model = AutoModel.from_pretrained(
        fetched.path,
        local_files_only=True,
        dtype=torch.float32,
    )
    reference_model.eval()

    prompt = "Hello, how are you?"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        reference_hidden = reference_model(input_ids=input_ids).last_hidden_state
        reference_logits = reference_hidden @ reference_model.embed_tokens.weight.T

    assert logits.shape == reference_logits.shape
    torch.testing.assert_close(logits, reference_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_functional_llama3_smol_lm2_wikitext_perplexity() -> None:
    fetched = fetch_hf_model(SMOLLM2_135M_REPO_ID)
    cfg = llama3_config_from_fetched(fetched.config)

    assert cfg["rope_theta"] == 100000.0
    assert cfg["rope_interleaved"] is False

    tokenizer = Llama3Tokenizer(str(fetched.path / "tokenizer.json"))
    model = Llama3Model(cfg)
    model.load_fetched_model(fetched)

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=10,
        context_size=128,
    )

    # Expected observed result: around 64 perplexity on this bounded WikiText
    # slice. A finite value below 100 indicates the pretrained weights, tokenizer,
    # and RoPE layout loaded coherently rather than producing random logits or a
    # trivial overconfident score.
    assert math.isfinite(perplexity)
    assert perplexity < 100.0
