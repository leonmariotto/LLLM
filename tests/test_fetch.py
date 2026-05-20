from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import torch

from ..LLLM.fetch import (
    fetch_model_ir,
    load_model_ir,
)
from ..LLLM.gpt2 import GPT2Model
from ..LLLM.gpt2 import GPT2TransformerBlock
from ..LLLM.hf_loader import model_ir_from_hf


_safetensors_torch = cast(Any, import_module("safetensors.torch"))
_save_file = cast(
    Callable[[dict[str, torch.Tensor], Path], None],
    _safetensors_torch.save_file,
)


def _tiny_hf_gpt2_config() -> dict[str, int | str | float]:
    return {
        "model_type": "gpt2",
        "vocab_size": 5,
        "n_positions": 4,
        "n_embd": 4,
        "n_head": 1,
        "n_layer": 1,
        "resid_pdrop": 0.0,
    }


def _tiny_gpt2_weights() -> dict[str, torch.Tensor]:
    emb_dim = 4
    vocab_size = 5
    context_length = 4
    return {
        "transformer.wte.weight": torch.arange(
            vocab_size * emb_dim, dtype=torch.float32
        ).reshape(vocab_size, emb_dim),
        "transformer.wpe.weight": torch.arange(
            context_length * emb_dim, dtype=torch.float32
        ).reshape(context_length, emb_dim),
        "transformer.h.0.attn.c_attn.weight": torch.arange(
            emb_dim * emb_dim * 3, dtype=torch.float32
        ).reshape(emb_dim, emb_dim * 3),
        "transformer.h.0.attn.c_attn.bias": torch.arange(
            emb_dim * 3, dtype=torch.float32
        ),
        "transformer.h.0.attn.c_proj.weight": torch.arange(
            emb_dim * emb_dim, dtype=torch.float32
        ).reshape(emb_dim, emb_dim),
        "transformer.h.0.attn.c_proj.bias": torch.arange(
            emb_dim, dtype=torch.float32
        ),
        "transformer.h.0.ln_1.weight": torch.full((emb_dim,), 1.1),
        "transformer.h.0.ln_1.bias": torch.full((emb_dim,), 1.2),
        "transformer.h.0.ln_2.weight": torch.full((emb_dim,), 2.1),
        "transformer.h.0.ln_2.bias": torch.full((emb_dim,), 2.2),
        "transformer.h.0.mlp.c_fc.weight": torch.arange(
            emb_dim * emb_dim * 4, dtype=torch.float32
        ).reshape(emb_dim, emb_dim * 4),
        "transformer.h.0.mlp.c_fc.bias": torch.arange(
            emb_dim * 4, dtype=torch.float32
        ),
        "transformer.h.0.mlp.c_proj.weight": torch.arange(
            emb_dim * 4 * emb_dim, dtype=torch.float32
        ).reshape(emb_dim * 4, emb_dim),
        "transformer.h.0.mlp.c_proj.bias": torch.arange(
            emb_dim, dtype=torch.float32
        ),
        "transformer.ln_f.weight": torch.full((emb_dim,), 3.1),
        "transformer.ln_f.bias": torch.full((emb_dim,), 3.2),
    }


def _unprefixed_gpt2_weights() -> dict[str, torch.Tensor]:
    prefix = "transformer."
    return {
        key.removeprefix(prefix): value
        for key, value in _tiny_gpt2_weights().items()
    }


def _write_tiny_hf_gpt2_snapshot(path: Path) -> None:
    (path / "config.json").write_text(
        (
            '{"model_type": "gpt2", "vocab_size": 5, "n_positions": 4, '
            '"n_embd": 4, "n_head": 1, "n_layer": 1}'
        ),
        encoding="utf-8",
    )
    _save_file(
        {"transformer.wte.weight": torch.ones(5, 4)},
        path / "model.safetensors",
    )


def test_load_model_ir_reads_hf_config_and_safetensors(tmp_path: Path) -> None:
    _write_tiny_hf_gpt2_snapshot(tmp_path)

    ir = load_model_ir(tmp_path)

    assert ir.architecture == "gpt2"
    assert ir.config.require_int("vocab_size") == 5
    assert set(ir.weights) == {"token_embedding.weight"}


def test_load_model_ir_rejects_quantized_safetensors(tmp_path: Path) -> None:
    _write_tiny_hf_gpt2_snapshot(tmp_path)

    with pytest.raises(NotImplementedError, match="GGUF"):
        load_model_ir(tmp_path, weight_mode="quantized")


def test_fetch_model_ir_loads_local_snapshot_path(tmp_path: Path) -> None:
    _write_tiny_hf_gpt2_snapshot(tmp_path)

    ir = fetch_model_ir(str(tmp_path))

    assert ir.architecture == "gpt2"
    assert ir.metadata["path"] == str(tmp_path)
    assert set(ir.weights) == {"token_embedding.weight"}


def test_gpt_model_loads_from_hf_gpt2_ir() -> None:
    weights = _tiny_gpt2_weights()
    ir = model_ir_from_hf(_tiny_hf_gpt2_config(), weights, architecture="gpt2")

    model = GPT2Model(GPT2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    block = cast(GPT2TransformerBlock, model.trf_blocks[0])

    torch.testing.assert_close(model.tok_emb.weight, weights["transformer.wte.weight"])
    torch.testing.assert_close(model.out_head.weight, weights["transformer.wte.weight"])
    q_weight, k_weight, v_weight = weights["transformer.h.0.attn.c_attn.weight"].chunk(
        3, dim=1
    )
    torch.testing.assert_close(block.att.W_query.weight, q_weight.T)
    torch.testing.assert_close(block.att.W_key.weight, k_weight.T)
    torch.testing.assert_close(block.att.W_value.weight, v_weight.T)
    torch.testing.assert_close(
        block.ff.fc1.weight,
        weights["transformer.h.0.mlp.c_fc.weight"].T,
    )
    torch.testing.assert_close(
        block.ff.fc2.weight,
        weights["transformer.h.0.mlp.c_proj.weight"].T,
    )


def test_gpt_model_loads_from_openai_gpt2_unprefixed_safetensors() -> None:
    weights = _unprefixed_gpt2_weights()
    ir = model_ir_from_hf(_tiny_hf_gpt2_config(), weights, architecture="gpt2")

    model = GPT2Model(GPT2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    block = cast(GPT2TransformerBlock, model.trf_blocks[0])

    torch.testing.assert_close(model.tok_emb.weight, weights["wte.weight"])
    torch.testing.assert_close(model.out_head.weight, weights["wte.weight"])
    q_weight, k_weight, v_weight = weights["h.0.attn.c_attn.weight"].chunk(
        3, dim=1
    )
    torch.testing.assert_close(block.att.W_query.weight, q_weight.T)
    torch.testing.assert_close(block.att.W_key.weight, k_weight.T)
    torch.testing.assert_close(block.att.W_value.weight, v_weight.T)
