"""
Stable internal model representation.

``ModelIR`` is the boundary between source formats and model implementations.
Hugging Face repositories, GGUF files, and future artifact formats must be
translated into this representation before a model class consumes them.

The IR deliberately uses semantic names instead of source names:

* global tensors: ``token_embedding.weight``, ``position_embedding.weight``,
  ``final_norm.weight``, ``final_norm.bias``, ``lm_head.weight``
* decoder attention: ``layers.N.attention.q_proj.weight``,
  ``k_proj.weight``, ``v_proj.weight``, ``o_proj.weight`` and optional biases
* decoder attention norms: ``layers.N.attention.q_norm.weight`` and
  ``k_norm.weight`` when the architecture has per-head norms
* decoder block norms: ``layers.N.input_norm.weight``,
  ``post_attention_norm.weight``, ``pre_ffn_norm.weight``,
  ``post_ffn_norm.weight``
* decoder feed-forward: ``layers.N.feed_forward.gate_proj.weight``,
  ``up_proj.weight``, ``down_proj.weight``
* GPT-2 packed projections: ``layers.N.attention.qkv_proj.weight`` and
  ``qkv_proj.bias``.  The values preserve GPT-2 Conv1D layout
  ``[input_dim, 3 * output_dim]`` so the GPT-2 loader can split and transpose.

Config fields are normalized by meaning and unit.  Required common fields are
``vocab_size``, ``context_length``, ``hidden_size``, ``intermediate_size``,
``num_hidden_layers``, and ``num_attention_heads``.  Architectures add fields:

* ``gpt2``: ``dropout`` (probability), ``qkv_bias`` (bool),
  ``positional_encoding`` (``"gpt2"`` or ``"rope"``)
* ``llama2``: ``rope_theta`` (float)
* ``llama3``: ``num_key_value_heads``, ``rope_theta``, ``rope_interleaved``,
  optional ``rope_scaling``
* ``qwen2``: ``num_key_value_heads``, ``head_dim``, ``rope_theta``,
  ``rope_interleaved``, ``rms_norm_eps``, and ``attention_bias``
* ``qwen3``: ``num_key_value_heads``, ``head_dim``, ``rope_theta``,
  ``rope_interleaved``, ``rms_norm_eps``, ``attention_bias``, and Q/K norms
* ``gemma3``: ``num_key_value_heads``, ``sliding_window``, ``head_dim``,
  ``rope_base``, ``rope_local_base``, ``rope_interleaved``, ``layer_types``,
  ``rms_norm_eps``, ``query_pre_attn_scalar``, optional logit softcapping
  fields, and ``attention_bias``

Dense weights are plain ``torch.Tensor`` instances.  Quantized weights are
``QuantizedWeight`` containers that keep packed source data plus explicit layout
transform metadata.  Model classes may install them into ``QuantizedLinear`` but
must not infer source-specific transforms from tensor names.

Format loaders own source complexity: nested HF configs, HF aliases, tied
embeddings, GGUF metadata names, GGUF tensor roles, tokenizer payloads, and
quantized tensor preservation.  Model loaders own only validation and copying
from this canonical representation into modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch

from .quantization import QuantizedWeight


ArchitectureId = Literal["gpt2", "llama2", "llama3", "qwen2", "qwen3", "gemma3"]
IRWeight: TypeAlias = torch.Tensor | QuantizedWeight
ModelWeightsIR: TypeAlias = dict[str, IRWeight]


@dataclass(frozen=True)
class ModelConfigIR:
    """Normalized architecture config consumed by model constructors."""

    fields: dict[str, Any]

    def require_int(self, name: str) -> int:
        value = self.fields.get(name)
        if not isinstance(value, int):
            raise ValueError(f"IR config field {name!r} must be an int")
        return value

    def require_float(self, name: str) -> float:
        value = self.fields.get(name)
        if isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise ValueError(f"IR config field {name!r} must be a float")
        return value

    def optional_float(self, name: str) -> float | None:
        value = self.fields.get(name)
        if value is None:
            return None
        if isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise ValueError(f"IR config field {name!r} must be a float or None")
        return value

    def require_bool(self, name: str) -> bool:
        value = self.fields.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"IR config field {name!r} must be a bool")
        return value

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class ModelIR:
    """Source-independent model description and canonical tensors."""

    architecture: ArchitectureId
    config: ModelConfigIR
    weights: ModelWeightsIR
    metadata: dict[str, Any] = field(default_factory=lambda: {})


SOURCE_PREFIX_FRAGMENTS = (
    "model.layers",
    "model.language_model",
    "self_attn",
    "mlp.",
    "blk.",
    "token_embd",
)


def assert_canonical_weight_names(ir: ModelIR) -> None:
    """Raise if an IR leaks source-format tensor names."""

    leaked = [
        name
        for name in ir.weights
        if any(fragment in name for fragment in SOURCE_PREFIX_FRAGMENTS)
    ]
    if leaked:
        examples = ", ".join(sorted(leaked)[:5])
        raise ValueError(f"IR contains source-format tensor names: {examples}")
