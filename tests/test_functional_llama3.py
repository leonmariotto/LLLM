import pytest
import torch
from transformers import AutoModel

from ..LLLM.fetch import fetch_hf_model
from ..LLLM.llama3 import Llama3Model, Llama3Tokenizer, llama3_config_from_fetched


LLAMA3_TINY_INSTRUCT_REPO_ID = "AlignmentResearch/Llama-3.3-Tiny-Instruct"


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
