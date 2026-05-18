from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.gguf_reader import ReaderField, ReaderTensor

from ..LLLM.gguf import (
    config_from_gguf_reader,
    dequantize_gguf_tensor,
    tensors_from_gguf_reader,
    _unpermute_llama_attention_weight,
)


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

    reader = gguf.GGUFReader(path)

    config = config_from_gguf_reader(reader)
    assert config["model_type"] == "llama"
    assert config["vocab_size"] == 4
    assert config["max_position_embeddings"] == 8
    assert config["num_key_value_heads"] == 1

    weights = tensors_from_gguf_reader(reader)
    assert set(weights) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    }
    torch.testing.assert_close(
        weights["model.embed_tokens.weight"],
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


def test_unpermute_llama_attention_weight_restores_split_half_layout() -> None:
    hf_layout = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(8, 3)
    gguf_layout = hf_layout.reshape(2, 2, 2, 3).transpose(1, 2).reshape(8, 3)

    restored = _unpermute_llama_attention_weight(gguf_layout, n_heads=2, head_dim=4)

    torch.testing.assert_close(restored, hf_layout)
