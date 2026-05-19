"""
GGUF loading helpers.

GGUF files are single binary files containing metadata, tokenizer data, and possibly
quantized tensors.
This module is the adapter layer that turns GGUF back into the same representation
used by the existing loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from gguf.gguf_reader import ReaderField, ReaderTensor

from loguru import logger

from .quantization import DenseOrQuantizedWeight, QuantizedWeight


GGUF_WEIGHT_ALIASES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "output.weight",
    "output_norm.weight": "model.norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


class GGUFLoadError(ValueError):
    """Raised when a GGUF file cannot be translated into this project's format."""


def find_gguf_file(path: str | Path, filename: str | None = None) -> Path:
    """Return the GGUF file selected by ``filename`` or by single-file detection."""
    source = Path(path).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".gguf":
            raise GGUFLoadError(f"{source} is not a .gguf file")
        return source

    if filename is not None:
        selected = source / filename
        if not selected.is_file():
            raise FileNotFoundError(f"missing GGUF file {selected}")
        return selected

    files = sorted(source.glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"no GGUF weights found in {source}")
    if len(files) > 1:
        names = ", ".join(file.name for file in files[:5])
        raise GGUFLoadError(
            f"multiple GGUF files found in {source}; pass filename explicitly "
            f"(examples: {names})"
        )
    return files[0]


def load_gguf(
    path: str | Path, filename: str | None = None
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Load a GGUF model as Hugging Face-like config plus dequantized tensors.

    This is intentionally eager for the first milestone: every GGUF tensor is
    dequantized into a normal Torch tensor so ``Llama3Model.load_fetched_model``
    can keep using its existing copy logic.
    """
    gguf_path = find_gguf_file(path, filename)
    logger.info("Reading GGUF file [%s]" % gguf_path)
    reader = GGUFReader(gguf_path)
    config = config_from_gguf_reader(reader)
    logger.info("Config extracted from GGUF file [%s]" % str(config))
    weights = tensors_from_gguf_reader(reader)
    logger.info("Weight dequantized")
    return config, weights


def load_gguf_quantized(
    path: str | Path, filename: str | None = None
) -> tuple[dict[str, Any], dict[str, DenseOrQuantizedWeight]]:
    """
    Load a GGUF model while preserving quantized linear tensors.

    Non-linear tensors such as embeddings and norms remain dense for this
    milestone. Linear tensors can be installed into ``QuantizedLinear`` and
    dequantized inside their forward pass.
    """
    gguf_path = find_gguf_file(path, filename)
    logger.info("Reading GGUF file [%s]" % gguf_path)
    reader = GGUFReader(gguf_path)
    config = config_from_gguf_reader(reader)
    logger.info("Config extracted from GGUF file [%s]" % str(config))
    weights = quantized_tensors_from_gguf_reader(reader)
    logger.info("Weight extracted (with quantization)")
    return config, weights


# TODO I should have an intermediate representation so that
# Any format -> Intermediate -> Any models.
# Because here this function translate gguf config into HF (config.json) config.
def config_from_gguf_reader(reader: GGUFReader) -> dict[str, Any]:
    """Translate GGUF Llama metadata into the keys ``llama3_config_from_fetched`` expects."""
    architecture = _string_field(reader, "general.architecture")
    if architecture != "llama":
        raise GGUFLoadError(f"unsupported GGUF architecture {architecture!r}")

    config: dict[str, Any] = {
        "model_type": "llama",
        "vocab_size": _vocab_size(reader),
        "max_position_embeddings": _int_field(reader, "llama.context_length"),
        "hidden_size": _int_field(reader, "llama.embedding_length"),
        "intermediate_size": _int_field(reader, "llama.feed_forward_length"),
        "num_hidden_layers": _int_field(reader, "llama.block_count"),
        "num_attention_heads": _int_field(reader, "llama.attention.head_count"),
        "num_key_value_heads": _int_field(
            reader,
            "llama.attention.head_count_kv",
            default=_int_field(reader, "llama.attention.head_count"),
        ),
        "rope_theta": _float_field(reader, "llama.rope.freq_base", default=10000.0),
        # llama.cpp and HF both store Llama 3 tensors in split-half RoPE layout.
        "rope_interleaved": False,
    }

    rope_type = _optional_string_field(reader, "llama.rope.scaling.type")
    if rope_type is not None and rope_type != "none":
        if rope_type != "llama3":
            raise GGUFLoadError(f"unsupported GGUF rope scaling type {rope_type!r}")
        config["rope_scaling"] = {
            "rope_type": "llama3",
            "factor": _float_field(reader, "llama.rope.scaling.factor"),
            "low_freq_factor": _float_field(
                reader, "llama.rope.scaling.low_freq_factor"
            ),
            "high_freq_factor": _float_field(
                reader, "llama.rope.scaling.high_freq_factor"
            ),
            "original_max_position_embeddings": _int_field(
                reader, "llama.rope.scaling.original_context_length"
            ),
        }
    elif inferred_rope_scaling := _infer_llama3_rope_scaling(reader, config):
        config["rope_scaling"] = inferred_rope_scaling

    return config


def tensors_from_gguf_reader(reader: GGUFReader) -> dict[str, torch.Tensor]:
    """Dequantize all supported GGUF tensors and expose Hugging Face-style names."""
    n_heads = _int_field(reader, "llama.attention.head_count")
    n_kv_heads = _int_field(reader, "llama.attention.head_count_kv", default=n_heads)
    head_dim = _int_field(
        reader,
        "llama.attention.key_length",
        default=_int_field(reader, "llama.embedding_length") // n_heads,
    )

    tensors: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        name = _map_tensor_name(tensor.name)
        if name is None:
            continue
        if name in tensors:
            raise GGUFLoadError(f"duplicate mapped tensor name {name!r}")
        value = dequantize_gguf_tensor(tensor, dtype=torch.float16)
        if tensor.name.endswith(".attn_q.weight"):
            value = _unpermute_llama_attention_weight(value, n_heads, head_dim)
        elif tensor.name.endswith(".attn_k.weight"):
            value = _unpermute_llama_attention_weight(value, n_kv_heads, head_dim)
        tensors[name] = value
    return tensors


def quantized_tensors_from_gguf_reader(
    reader: GGUFReader,
) -> dict[str, DenseOrQuantizedWeight]:
    """Expose GGUF tensors, preserving quantized linear weights when possible."""
    n_heads = _int_field(reader, "llama.attention.head_count")
    n_kv_heads = _int_field(reader, "llama.attention.head_count_kv", default=n_heads)
    head_dim = _int_field(
        reader,
        "llama.attention.key_length",
        default=_int_field(reader, "llama.embedding_length") // n_heads,
    )

    tensors: dict[str, DenseOrQuantizedWeight] = {}
    for tensor in reader.tensors:
        name = _map_tensor_name(tensor.name)
        if name is None:
            continue
        if name in tensors:
            raise GGUFLoadError(f"duplicate mapped tensor name {name!r}")

        transform_heads: int | None = None
        if tensor.name.endswith(".attn_q.weight"):
            transform_heads = n_heads
        elif tensor.name.endswith(".attn_k.weight"):
            transform_heads = n_kv_heads

        if _is_forward_quantized_linear(name, tensor):
            data = cast(np.ndarray[Any, Any], np.array(tensor.data, copy=True))
            tensors[name] = QuantizedWeight(
                name=name,
                tensor_type=tensor.tensor_type,
                data=data,
                shape=tuple(int(dim) for dim in tensor.shape[::-1]),
                transform=(
                    "llama_attention_unpermute" if transform_heads is not None else None
                ),
                n_heads=transform_heads,
                head_dim=head_dim if transform_heads is not None else None,
            )
            continue

        value = dequantize_gguf_tensor(tensor, dtype=torch.float16)
        if transform_heads is not None:
            value = _unpermute_llama_attention_weight(value, transform_heads, head_dim)
        tensors[name] = value
    return tensors


def dequantize_gguf_tensor(
    tensor: ReaderTensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Return a dense Torch tensor for a GGUF tensor.

    K-quants such as Q4_K and Q6_K are stored as packed byte blocks.  The
    upstream ``gguf`` package owns the exact block layout, while this function
    performs the project-specific conversion into a writable Torch tensor.
    """
    if tensor.tensor_type == GGMLQuantizationType.BF16:
        # gguf.dequantize does not currently handle BF16, but some GGUF files use
        # it for small non-quantized tensors.  Preserve that case explicitly.
        array = tensor.data.view(np.uint16).astype(np.uint32) << 16
        array = array.view(np.float32)
    else:
        array = gguf.dequantize(tensor.data, tensor.tensor_type)

    dense_array = cast(np.ndarray[Any, Any], np.array(array, copy=True))
    torch_any = cast(Any, torch)
    from_numpy = cast(Callable[[Any], torch.Tensor], torch_any.from_numpy)
    torch_tensor = from_numpy(dense_array)
    if torch_tensor.is_floating_point():
        torch_tensor = torch_tensor.to(dtype=dtype)
    return torch_tensor


def _is_forward_quantized_linear(name: str, tensor: ReaderTensor) -> bool:
    if not _is_quantized_tensor(tensor):
        return False
    if name in {"output.weight", "lm_head.weight"}:
        return True
    if ".self_attn." in name:
        return True
    return ".mlp." in name


def _is_quantized_tensor(tensor: ReaderTensor) -> bool:
    return tensor.tensor_type not in {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    }


def tokenizer_json_from_gguf(path: str | Path) -> str | None:
    """Return embedded Hugging Face tokenizer JSON when the GGUF file contains it."""
    reader = GGUFReader(find_gguf_file(path))
    field = reader.fields.get("tokenizer.huggingface.json")
    if field is None:
        return None
    return cast(str, field.contents())


def tokenizer_mergeable_ranks_from_gguf(path: str | Path) -> dict[bytes, int]:
    """
    Build tiktoken mergeable ranks from GGUF's GPT-2/Llama-BPE tokenizer fields.

    GGUF stores token text using the same printable-byte alphabet used by
    GPT-2's ``vocab.json``.  tiktoken wants the original byte strings, so this
    function reverses that byte-to-unicode transform and keeps only normal BPE
    tokens.  Llama special tokens are supplied separately by ``Llama3Tokenizer``.
    """
    reader = GGUFReader(find_gguf_file(path))
    tokenizer_model = _string_field(reader, "tokenizer.ggml.model")
    if tokenizer_model != "gpt2":
        raise GGUFLoadError(
            f"unsupported GGUF tokenizer model {tokenizer_model!r}; expected 'gpt2'"
        )

    tokens = cast(list[str], _field(reader, "tokenizer.ggml.tokens").contents())
    token_types = cast(
        list[int], _field(reader, "tokenizer.ggml.token_type").contents()
    )
    byte_decoder = _data_gym_byte_decoder()

    ranks: dict[bytes, int] = {}
    for token_id, (token, token_type) in enumerate(zip(tokens, token_types)):
        # In GGUF TokenType, 1 is NORMAL.  Special/control tokens are handled by
        # the explicit Llama 3 special-token map rather than mergeable ranks.
        if token_type != 1:
            continue
        try:
            ranks[bytes(byte_decoder[char] for char in token)] = token_id
        except KeyError:
            # Some tokenizer entries are special strings that are not encoded in
            # GPT-2's byte alphabet even when their token_type is loose.  They
            # are not valid mergeable BPE tokens, so skip them.
            continue
    return ranks


def _map_tensor_name(name: str) -> str | None:
    exact = GGUF_WEIGHT_ALIASES.get(name)
    if exact is not None:
        return exact

    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk":
        layer_idx = parts[1]
        local_name = ".".join(parts[2:])
        hf_suffix = GGUF_WEIGHT_ALIASES.get(local_name)
        if hf_suffix is None:
            return None
        return f"model.layers.{layer_idx}.{hf_suffix}"

    return None


def _unpermute_llama_attention_weight(
    weight: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Convert llama.cpp's GGUF Q/K layout back to this model's split-half RoPE layout.

    llama.cpp stores Llama query/key projection rows with each RoPE pair adjacent
    because its kernels rotate pairs directly.  This project uses the Hugging
    Face-style split-half layout in ``apply_rope(..., use_interleaved=False)``:
    all first-half coordinates first, then all second-half coordinates.  The
    transform below is the inverse of llama.cpp's conversion-time permutation.
    """
    if weight.ndim != 2:
        raise GGUFLoadError(f"expected 2D attention weight, got {tuple(weight.shape)}")
    expected_rows = n_heads * head_dim
    if weight.shape[0] != expected_rows:
        raise GGUFLoadError(
            f"attention weight has {weight.shape[0]} rows, expected {expected_rows}"
        )
    if head_dim % 2 != 0:
        raise GGUFLoadError(f"attention head_dim must be even, got {head_dim}")

    return (
        weight.reshape(n_heads, head_dim // 2, 2, weight.shape[1])
        .transpose(1, 2)
        .reshape(weight.shape)
    )


def _vocab_size(reader: GGUFReader) -> int:
    if "llama.vocab_size" in reader.fields:
        return _int_field(reader, "llama.vocab_size")
    if "tokenizer.ggml.tokens" in reader.fields:
        return len(cast(list[str], reader.fields["tokenizer.ggml.tokens"].contents()))
    raise GGUFLoadError("GGUF metadata does not contain a vocab size")


def _infer_llama3_rope_scaling(
    reader: GGUFReader, config: dict[str, Any]
) -> dict[str, float | int | str] | None:
    """
    Infer Llama 3.1/3.2 RoPE scaling when GGUF only stores rope_freqs.weight.

    Some llama.cpp conversions omit the explicit ``llama.rope.scaling.*``
    metadata because llama.cpp can consume the precomputed ``rope_freqs.weight``
    tensor directly.  This project computes RoPE from config instead, so for the
    known Llama 3.1/3.2 long-context family we restore the published scaling
    constants used by Hugging Face configs.
    """
    name = _optional_string_field(reader, "general.name") or ""
    basename = _optional_string_field(reader, "general.basename") or ""
    model_label = f"{name} {basename}".lower()
    if (
        "llama 3.1" not in model_label
        and "llama-3.1" not in model_label
        and "llama 3.2" not in model_label
        and "llama-3.2" not in model_label
    ):
        return None
    if int(config["max_position_embeddings"]) < 131072:
        return None
    if not any(tensor.name == "rope_freqs.weight" for tensor in reader.tensors):
        return None
    return {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }


def _field(reader: GGUFReader, name: str) -> ReaderField:
    field = reader.fields.get(name)
    if field is None:
        raise GGUFLoadError(f"missing GGUF metadata field {name!r}")
    return field


def _int_field(reader: GGUFReader, name: str, *, default: int | None = None) -> int:
    field = reader.fields.get(name)
    if field is None:
        if default is not None:
            return default
        raise GGUFLoadError(f"missing GGUF metadata field {name!r}")
    return int(field.contents())


def _float_field(
    reader: GGUFReader, name: str, *, default: float | None = None
) -> float:
    field = reader.fields.get(name)
    if field is None:
        if default is not None:
            return default
        raise GGUFLoadError(f"missing GGUF metadata field {name!r}")
    return float(field.contents())


def _string_field(reader: GGUFReader, name: str) -> str:
    return cast(str, _field(reader, name).contents())


def _optional_string_field(reader: GGUFReader, name: str) -> str | None:
    field = reader.fields.get(name)
    if field is None:
        return None
    return cast(str, field.contents())


def _data_gym_byte_decoder() -> dict[str, int]:
    """Return the inverse of GPT-2's printable-byte encoding."""
    printable = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
    byte_decoder = {chr(b): b for b in printable}
    n = 0
    for byte in range(2**8):
        if byte not in printable:
            byte_decoder[chr(2**8 + n)] = byte
            n += 1
    return byte_decoder
