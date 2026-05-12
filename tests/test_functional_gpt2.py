from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch

from ..LLLM.fetch import fetch_hf_model
from ..LLLM.gpt import GPTModel, gpt_config_from_fetched


_safetensors_torch = cast(Any, import_module("safetensors.torch"))
_save_file = cast(
    Callable[[dict[str, torch.Tensor], Path], None],
    _safetensors_torch.save_file,
)


def test_functional_gpt2_fetch_load_and_evaluate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    snapshot_path = tmp_path / "tiny-gpt2"
    snapshot_path.mkdir()
    (snapshot_path / "config.json").write_text(
        """
        {
          "model_type": "gpt2",
          "vocab_size": 5,
          "n_positions": 4,
          "n_embd": 4,
          "n_head": 1,
          "n_layer": 1,
          "resid_pdrop": 0.0
        }
        """,
        encoding="utf-8",
    )
    _save_file(_tiny_gpt2_weights(), snapshot_path / "model.safetensors")

    def fake_snapshot_download(**_: Any) -> str:
        return str(snapshot_path)

    hf_hub = cast(Any, import_module("huggingface_hub"))
    monkeypatch.setattr(
        hf_hub,
        "snapshot_download",
        fake_snapshot_download,
    )

    fetched = fetch_hf_model("local/tiny-gpt2", local_files_only=True)
    model = GPTModel(gpt_config_from_fetched(fetched.config))
    model.load_fetched_model(fetched)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor([[0, 1, 2]], dtype=torch.long))

    assert logits.shape == (1, 3, 5)
    assert torch.isfinite(logits).all()


def _tiny_gpt2_weights() -> dict[str, torch.Tensor]:
    emb_dim = 4
    vocab_size = 5
    context_length = 4
    return {
        "transformer.wte.weight": torch.arange(
            vocab_size * emb_dim, dtype=torch.float32
        ).reshape(vocab_size, emb_dim)
        / 100,
        "transformer.wpe.weight": torch.arange(
            context_length * emb_dim, dtype=torch.float32
        ).reshape(context_length, emb_dim)
        / 100,
        "transformer.h.0.attn.c_attn.weight": torch.arange(
            emb_dim * emb_dim * 3, dtype=torch.float32
        ).reshape(emb_dim, emb_dim * 3)
        / 100,
        "transformer.h.0.attn.c_attn.bias": torch.zeros(emb_dim * 3),
        "transformer.h.0.attn.c_proj.weight": torch.eye(emb_dim),
        "transformer.h.0.attn.c_proj.bias": torch.zeros(emb_dim),
        "transformer.h.0.ln_1.weight": torch.ones(emb_dim),
        "transformer.h.0.ln_1.bias": torch.zeros(emb_dim),
        "transformer.h.0.ln_2.weight": torch.ones(emb_dim),
        "transformer.h.0.ln_2.bias": torch.zeros(emb_dim),
        "transformer.h.0.mlp.c_fc.weight": torch.arange(
            emb_dim * emb_dim * 4, dtype=torch.float32
        ).reshape(emb_dim, emb_dim * 4)
        / 100,
        "transformer.h.0.mlp.c_fc.bias": torch.zeros(emb_dim * 4),
        "transformer.h.0.mlp.c_proj.weight": torch.arange(
            emb_dim * 4 * emb_dim, dtype=torch.float32
        ).reshape(emb_dim * 4, emb_dim)
        / 100,
        "transformer.h.0.mlp.c_proj.bias": torch.zeros(emb_dim),
        "transformer.ln_f.weight": torch.ones(emb_dim),
        "transformer.ln_f.bias": torch.zeros(emb_dim),
    }
