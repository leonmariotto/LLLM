from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch import nn

from ..LLLM.fetch import (
    FetchedModel,
    load_cached_model,
)
from ..LLLM.gpt import GPTModel, gpt_config_from_fetched
from ..LLLM.transformer import TransformerBlock


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


def test_load_cached_model_reads_config_and_safetensors(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "gpt2"}')
    _save_file(
        {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        tmp_path / "a.safetensors",
    )
    _save_file({"c": torch.tensor([3.0])}, tmp_path / "b.safetensors")

    fetched = load_cached_model(tmp_path)

    assert fetched.model_type == "gpt2"
    assert set(fetched.weights) == {"a", "b", "c"}


def test_gpt_model_loads_from_fetched_gpt2_artifacts() -> None:
    weights = _tiny_gpt2_weights()
    fetched = FetchedModel(
        path=Path("/tmp/fake-snapshot"),
        config=_tiny_hf_gpt2_config(),
        weights=weights,
    )

    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    block = cast(TransformerBlock, model.trf_blocks[0])

    torch.testing.assert_close(model.tok_emb.weight, weights["transformer.wte.weight"])
    torch.testing.assert_close(model.out_head.weight, weights["transformer.wte.weight"])
    q_weight, k_weight, v_weight = weights["transformer.h.0.attn.c_attn.weight"].chunk(
        3, dim=1
    )
    torch.testing.assert_close(block.att.W_query.weight, q_weight.T)
    torch.testing.assert_close(block.att.W_key.weight, k_weight.T)
    torch.testing.assert_close(block.att.W_value.weight, v_weight.T)
    fc = cast(nn.Linear, block.ff.layers[0])
    proj = cast(nn.Linear, block.ff.layers[2])
    torch.testing.assert_close(
        fc.weight,
        weights["transformer.h.0.mlp.c_fc.weight"].T,
    )
    torch.testing.assert_close(
        proj.weight,
        weights["transformer.h.0.mlp.c_proj.weight"].T,
    )
