from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.gguf_reader import ReaderField, ReaderTensor

from ..LLLM.gguf import (
    dequantize_gguf_tensor,
    load_gguf_ir,
    _unpermute_llama_attention_weight,
)
from ..LLLM.quantization import QuantizedLinear, QuantizedWeight


def test_gguf_reader_translates_llama_metadata_and_tensor_names(tmp_path: Path) -> None:
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
    writer.add_rope_freq_base(500000.0)
    writer.add_tensor(
        "token_embd.weight",
        np.arange(16, dtype=np.float32).reshape(4, 4),
    )
    writer.add_tensor("blk.0.attn_q.weight", np.eye(4, dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    ir = load_gguf_ir(path)

    assert ir.architecture == "llama3"
    assert ir.config.get("vocab_size") == 4
    assert ir.config.get("context_length") == 8
    assert ir.config.get("num_key_value_heads") == 1

    assert set(ir.weights) == {
        "token_embedding.weight",
        "layers.0.attention.q_proj.weight",
        "final_norm.weight",
        "lm_head.weight",
    }
    torch.testing.assert_close(
        ir.weights["token_embedding.weight"],
        torch.arange(16, dtype=torch.float16).reshape(4, 4),
    )


def test_dequantize_gguf_tensor_supports_quantized_blocks() -> None:
    source = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    quantized = gguf.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    tensor = ReaderTensor(
        name="blk.0.attn_q.weight",
        tensor_type=gguf.GGMLQuantizationType.Q4_0,
        shape=np.array([32, 1], dtype=np.uint32),
        n_elements=32,
        n_bytes=int(quantized.size),
        data_offset=0,
        data=quantized,
        field=ReaderField(0, "blk.0.attn_q.weight", [], [], []),
    )

    dense = dequantize_gguf_tensor(tensor)

    assert dense.shape == (1, 32)
    assert dense.dtype == torch.float32
    assert torch.isfinite(dense).all()


def test_quantized_linear_dequantizes_weight_in_forward() -> None:
    source = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    quantized = gguf.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    weight = QuantizedWeight(
        name="linear.weight",
        tensor_type=gguf.GGMLQuantizationType.Q4_0,
        data=quantized,
        shape=source.shape,
        dtype=torch.float32,
    )
    layer = QuantizedLinear(weight, in_features=32, out_features=1)
    x = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    output = layer(x)
    expected = torch.nn.functional.linear(x, weight.dequantize())

    torch.testing.assert_close(output, expected)


def test_quantized_gguf_reader_preserves_linear_quantized_weight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny-q4.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_vocab_size(4)
    writer.add_token_list(["a", "b", "c", "d"])
    writer.add_context_length(8)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(32)
    writer.add_block_count(1)
    writer.add_head_count(1)
    writer.add_head_count_kv(1)
    source = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    quantized = gguf.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor(
        "blk.0.ffn_down.weight",
        quantized,
        raw_shape=quantized.shape,
        raw_dtype=gguf.GGMLQuantizationType.Q4_0,
    )
    writer.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    ir = load_gguf_ir(path, weight_mode="quantized")

    assert isinstance(ir.weights["layers.0.feed_forward.down_proj.weight"], QuantizedWeight)
    assert isinstance(ir.weights["final_norm.weight"], torch.Tensor)


def test_unpermute_llama_attention_weight_restores_split_half_layout() -> None:
    hf_layout = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(8, 3)
    gguf_layout = hf_layout.reshape(2, 2, 2, 3).transpose(1, 2).reshape(8, 3)

    restored = _unpermute_llama_attention_weight(gguf_layout, n_heads=2, head_dim=4)

    torch.testing.assert_close(restored, hf_layout)
