import math
import gc

import pytest
import torch
from transformers import AutoModel

from ..LLLM.eval import (
    DatasetAdapter,
    boolq_adapter,
    boolq_prediction,
    evaluate_base_model_perplexity,
    evaluate_instructions_model,
)
from ..LLLM.fetch import fetch_hf_model
from ..LLLM.llama3 import Llama3Model, Llama3Tokenizer, llama3_config_from_fetched


LLAMA3_TINY_INSTRUCT_REPO_ID = "AlignmentResearch/Llama-3.3-Tiny-Instruct"
SMOLLM2_135M_REPO_ID = "HuggingFaceTB/SmolLM2-135M"
LLAMA32_1B_INSTRUCT_GGUF_REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"
LLAMA32_1B_INSTRUCT_Q4_K_M_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


llama3_boolq_adapter = DatasetAdapter(
    dataset_id=boolq_adapter.dataset_id,
    config=boolq_adapter.config,
    split=boolq_adapter.split,
    build_prompt=boolq_adapter.build_prompt,
    extract_expected=boolq_adapter.extract_expected,
    extract_prediction=boolq_prediction,
    score=boolq_adapter.score,
    encode_prompt=lambda tokenizer, prompt: (
        tokenizer.encode_instruct_prompt(prompt)
        if isinstance(tokenizer, Llama3Tokenizer)
        else tokenizer.encode(prompt)
    ),
    eos_token=lambda tokenizer: (
        tokenizer.special["<|eot_id|>"]
        if isinstance(tokenizer, Llama3Tokenizer)
        else None
    ),
)


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


@pytest.mark.slow
def test_functional_llama3_gguf_q4_k_m_runs_instruction_eval() -> None:
    fetched = fetch_hf_model(
        LLAMA32_1B_INSTRUCT_GGUF_REPO_ID,
        gguf_filename=LLAMA32_1B_INSTRUCT_Q4_K_M_FILE,
    )
    cfg = llama3_config_from_fetched(fetched.config)

    assert cfg["emb_dim"] == 2048
    assert cfg["n_layers"] == 16
    assert cfg["n_kv_groups"] == 8
    assert cfg["freq_config"] == {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_len": 8192,
    }

    tokenizer = Llama3Tokenizer.from_gguf(str(fetched.path))
    model = Llama3Model(cfg)
    model.load_fetched_model(fetched)
    del fetched

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=llama3_boolq_adapter,
        limit=5,
        max_generated_token=3,
        context_size=1024,
    )

    assert math.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.slow
def test_functional_llama3_gguf_q4_k_m_quantized_runs_instruction_eval() -> None:
    fetched = fetch_hf_model(
        LLAMA32_1B_INSTRUCT_GGUF_REPO_ID,
        gguf_filename=LLAMA32_1B_INSTRUCT_Q4_K_M_FILE,
        weight_mode="quantized",
    )
    cfg = llama3_config_from_fetched(fetched.config)

    tokenizer = Llama3Tokenizer.from_gguf(str(fetched.path))
    model = Llama3Model(cfg, weight_mode="quantized")
    model.load_quantized_fetched_model(fetched)
    del fetched

    accuracy = evaluate_instructions_model(
        model=model,
        tokenizer=tokenizer,
        adapter=llama3_boolq_adapter,
        limit=2,
        max_generated_token=2,
        context_size=1024,
    )

    assert math.isfinite(accuracy)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.slow
def test_functional_llama3_gguf_q4_k_m_quantized_matches_eager_dequantized() -> None:
    dense_fetched = fetch_hf_model(
        LLAMA32_1B_INSTRUCT_GGUF_REPO_ID,
        gguf_filename=LLAMA32_1B_INSTRUCT_Q4_K_M_FILE,
    )
    cfg = llama3_config_from_fetched(dense_fetched.config)
    tokenizer = Llama3Tokenizer.from_gguf(str(dense_fetched.path))
    input_ids = torch.tensor(
        [tokenizer.encode_instruct_prompt("Answer yes or no: is water wet?")],
        dtype=torch.long,
    )

    dense_model = Llama3Model(cfg)
    dense_model.load_fetched_model(dense_fetched)
    with torch.no_grad():
        dense_logits = dense_model(input_ids)
    del dense_model
    del dense_fetched
    gc.collect()

    quantized_fetched = fetch_hf_model(
        LLAMA32_1B_INSTRUCT_GGUF_REPO_ID,
        gguf_filename=LLAMA32_1B_INSTRUCT_Q4_K_M_FILE,
        weight_mode="quantized",
    )
    quantized_model = Llama3Model(cfg, weight_mode="quantized")
    quantized_model.load_quantized_fetched_model(quantized_fetched)
    with torch.no_grad():
        quantized_logits = quantized_model(input_ids)

    assert quantized_logits.shape == dense_logits.shape
    torch.testing.assert_close(quantized_logits, dense_logits, rtol=1e-5, atol=1e-5)
