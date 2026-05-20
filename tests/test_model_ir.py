from pathlib import Path

import gguf
import numpy as np
import torch

from ..LLLM.gemma3 import Gemma3Model
from ..LLLM.gguf import load_gguf_ir
from ..LLLM.hf_loader import model_ir_from_hf
from ..LLLM.model_ir import ModelConfigIR, ModelIR, assert_canonical_weight_names
from ..LLLM.quantization import QuantizedLinear, QuantizedWeight


def test_hf_loader_produces_canonical_gpt2_ir_names() -> None:
    config = {
        "model_type": "gpt2",
        "vocab_size": 5,
        "n_positions": 4,
        "n_embd": 4,
        "n_head": 1,
        "n_layer": 1,
    }
    weights = {
        "transformer.wte.weight": torch.randn(5, 4),
        "transformer.wpe.weight": torch.randn(4, 4),
        "transformer.h.0.attn.c_attn.weight": torch.randn(4, 12),
        "transformer.h.0.attn.c_attn.bias": torch.randn(12),
        "transformer.h.0.ln_1.weight": torch.randn(4),
        "transformer.ln_f.weight": torch.randn(4),
    }

    ir = model_ir_from_hf(config, weights)

    assert ir.architecture == "gpt2"
    assert "token_embedding.weight" in ir.weights
    assert "layers.0.attention.qkv_proj.weight" in ir.weights
    assert_canonical_weight_names(ir)


def test_gguf_loader_produces_canonical_ir_names(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_vocab_size(4)
    writer.add_token_list(["a", "b", "c", "d"])
    writer.add_context_length(8)
    writer.add_embedding_length(4)
    writer.add_feed_forward_length(12)
    writer.add_block_count(1)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_tensor("token_embd.weight", np.ones((4, 4), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.eye(4, dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    ir = load_gguf_ir(path)

    assert ir.architecture == "llama3"
    assert "token_embedding.weight" in ir.weights
    assert "layers.0.attention.q_proj.weight" in ir.weights
    assert "final_norm.weight" in ir.weights
    assert_canonical_weight_names(ir)


def test_gemma3_quantized_ir_loader_installs_quantized_linear() -> None:
    source = np.linspace(-1.0, 1.0, 5 * 32, dtype=np.float32).reshape(5, 32)
    quantized = gguf.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    ir = ModelIR(
        architecture="gemma3",
        config=ModelConfigIR(
            {
                "vocab_size": 5,
                "context_length": 4,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "sliding_window": 4,
                "num_hidden_layers": 0,
                "head_dim": 32,
                "rope_base": 1000000.0,
                "rope_local_base": 10000.0,
                "rope_interleaved": False,
                "layer_types": [],
                "rms_norm_eps": 1e-6,
                "query_pre_attn_scalar": 32,
                "final_logit_softcapping": None,
                "attn_logit_softcapping": None,
                "attention_bias": False,
            }
        ),
        weights={
            "token_embedding.weight": torch.randn(5, 32),
            "final_norm.weight": torch.zeros(32),
            "lm_head.weight": QuantizedWeight(
                name="lm_head.weight",
                tensor_type=gguf.GGMLQuantizationType.Q4_0,
                data=quantized,
                shape=source.shape,
                dtype=torch.float32,
            ),
        },
    )
    model = Gemma3Model(Gemma3Model.config_from_ir(ir), weight_mode="quantized")

    model.load_ir_weights(ir)
    logits = model(torch.tensor([[0, 1]], dtype=torch.long))

    assert isinstance(model.out_head, QuantizedLinear)
    assert logits.shape == (1, 2, 5)
    assert torch.isfinite(logits).all()
