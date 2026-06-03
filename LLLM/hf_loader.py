"""
Hugging Face artifact loader that produces ``ModelIR``.
HF format parser, provide only decode functions.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch

from .model_ir import ArchitectureId, ModelConfigIR, ModelIR, ModelWeightsIR
from .quantization import QuantizedWeight


def load_hf_model_ir(
    path: str | Path, *, architecture: ArchitectureId | None = None
) -> ModelIR:
    """
    Load ``config.json`` and safetensors from a HF-style snapshot as IR.

    Args:
        path: Snapshot directory containing ``config.json`` and safetensors.
        architecture: Optional architecture override when inference is ambiguous.

    Returns:
        Source-independent model IR.
    """

    snapshot_path = Path(path).expanduser()
    config_path = snapshot_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {snapshot_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} did not contain a JSON object")
    config = cast(dict[str, Any], raw_config)
    weights = _load_safetensors(snapshot_path)
    return model_ir_from_hf(
        config, weights, path=snapshot_path, architecture=architecture
    )


def model_ir_from_hf(
    config: dict[str, Any],
    weights: dict[str, Any],
    *,
    path: str | Path | None = None,
    architecture: ArchitectureId | None = None,
) -> ModelIR:
    """
    Translate decoded HF config and state dict tensors into canonical IR.

    Args:
        config: Decoded Hugging Face config mapping.
        weights: State dict mapping tensor names to tensor values.
        path: Optional source path stored in IR metadata.
        architecture: Optional architecture override when inference is ambiguous.

    Returns:
        Source-independent model IR.
    """

    arch = architecture or _infer_architecture(config)
    normalized_config = _normalize_config(config, arch)
    if arch == "gpt2":
        ir_weights = _canonical_gpt2_weights(weights)
    elif arch in {"llama2", "llama3", "qwen2", "qwen3"}:
        ir_weights = canonical_llama_weights_from_hf(weights)
    elif arch == "gemma3":
        ir_weights = _canonical_gemma3_weights(weights)
    else:
        raise AssertionError(f"unhandled architecture {arch}")
    metadata: dict[str, Any] = {
        "source_format": "huggingface",
        "original_architecture": config.get("model_type"),
    }
    if path is not None:
        metadata["path"] = str(path)
    return ModelIR(
        architecture=arch,
        config=ModelConfigIR(normalized_config),
        weights=ir_weights,
        metadata=metadata,
    )


def _infer_architecture(config: dict[str, Any]) -> ArchitectureId:
    """
    Infer the supported model family from Hugging Face config fields.

    Args:
        config: Decoded Hugging Face config mapping.

    Returns:
        Architecture id used by model constructors.
    """

    model_type = config.get("model_type")
    text_config = config.get("text_config")
    if model_type in {"gemma3", "gemma3_text"} or isinstance(text_config, dict):
        return "gemma3"
    if model_type == "gpt2":
        return "gpt2"
    if model_type in {"qwen2", "qwen3"}:
        return cast(ArchitectureId, model_type)
    if model_type == "llama":
        rope_scaling = config.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_scaling = cast(dict[str, Any], rope_scaling)
            if rope_scaling.get("rope_type") == "llama3":
                return "llama3"
        if config.get("num_key_value_heads") != config.get("num_attention_heads"):
            return "llama3"
        return "llama2"
    if all(key in config for key in ("dim", "n_layers", "n_heads")):
        if config.get("n_kv_heads") != config.get("n_heads"):
            return "llama3"
        return "llama2"
    raise ValueError(
        f"cannot infer supported architecture from model_type={model_type!r}"
    )


def _normalize_config(config: dict[str, Any], arch: ArchitectureId) -> dict[str, Any]:
    """
    Convert Hugging Face config keys into normalized IR config keys.

    Args:
        config: Decoded Hugging Face config mapping.
        arch: Architecture id selected for the model.

    Returns:
        Normalized config mapping.
    """

    if arch == "gemma3":
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            config = cast(dict[str, Any], text_config)

    if arch == "gpt2":
        return {
            "vocab_size": _int(config, "vocab_size"),
            "context_length": _int(config, "n_ctx", fallback_key="n_positions"),
            "hidden_size": _int(config, "n_embd"),
            "num_attention_heads": _int(config, "n_head"),
            "num_hidden_layers": _int(config, "n_layer"),
            "dropout": float(config.get("resid_pdrop", 0.0)),
            "qkv_bias": True,
            "positional_encoding": "gpt2",
        }

    if arch in {"llama2", "llama3", "qwen2", "qwen3"}:
        normalized = {
            "vocab_size": _int(config, "vocab_size"),
            "context_length": _int(
                config, "max_position_embeddings", fallback_key="max_seq_len"
            ),
            "hidden_size": _int(config, "hidden_size", fallback_key="dim"),
            "intermediate_size": _hidden_dim(config),
            "num_attention_heads": _int(
                config, "num_attention_heads", fallback_key="n_heads"
            ),
            "num_hidden_layers": _int(
                config, "num_hidden_layers", fallback_key="n_layers"
            ),
            "rope_theta": _rope_theta(config, default=10000.0),
            "rope_interleaved": _bool(config, "rope_interleaved", default=False),
        }
        if arch in {"llama3", "qwen2", "qwen3"}:
            normalized["num_key_value_heads"] = _int(
                config, "num_key_value_heads", fallback_key="n_kv_heads"
            )
            if "rope_scaling" in config:
                normalized["rope_scaling"] = config["rope_scaling"]
        if arch in {"qwen2", "qwen3"}:
            normalized["head_dim"] = int(
                config.get(
                    "head_dim",
                    normalized["hidden_size"] // normalized["num_attention_heads"],
                )
            )
            normalized["rms_norm_eps"] = _float(config, "rms_norm_eps", default=1e-6)
            normalized["attention_bias"] = _bool(
                config, "attention_bias", default=(arch == "qwen2")
            )
        return normalized

    layer_types = _list_str(
        config,
        "layer_types",
        default=["sliding_attention"]
        * _int(config, "num_hidden_layers", fallback_key="n_layers"),
    )
    return {
        "vocab_size": _int(config, "vocab_size"),
        "context_length": _int(
            config, "max_position_embeddings", fallback_key="max_seq_len"
        ),
        "hidden_size": _int(config, "hidden_size", fallback_key="dim"),
        "intermediate_size": _hidden_dim(config),
        "num_attention_heads": _int(
            config, "num_attention_heads", fallback_key="n_heads"
        ),
        "num_key_value_heads": _int(
            config, "num_key_value_heads", fallback_key="n_heads"
        ),
        "sliding_window": _int(config, "sliding_window"),
        "num_hidden_layers": _int(config, "num_hidden_layers", fallback_key="n_layers"),
        "head_dim": _int(config, "head_dim"),
        "rope_base": _rope_base(config, "full_attention", 1000000.0),
        "rope_local_base": _rope_base(config, "sliding_attention", 10000.0),
        "rope_interleaved": _bool(config, "rope_interleaved", default=False),
        "layer_types": layer_types,
        "rms_norm_eps": _float(config, "rms_norm_eps", default=1e-6),
        "query_pre_attn_scalar": _int(
            config, "query_pre_attn_scalar", fallback_key="head_dim"
        ),
        "final_logit_softcapping": _optional_float(config, "final_logit_softcapping"),
        "attn_logit_softcapping": _optional_float(config, "attn_logit_softcapping"),
        "attention_bias": _bool(config, "attention_bias", default=False),
    }


def _canonical_gpt2_weights(weights: dict[str, Any]) -> ModelWeightsIR:
    """
    Map GPT-2 Hugging Face state dict names into canonical IR names.

    Args:
        weights: State dict mapping tensor names to tensor values.

    Returns:
        Canonical IR weight mapping.
    """

    mapped: ModelWeightsIR = {}
    for source, value in weights.items():
        if not isinstance(value, (torch.Tensor, QuantizedWeight)):
            continue
        name = source.removeprefix("transformer.")
        if name == "wte.weight":
            mapped["token_embedding.weight"] = value
        elif name == "wpe.weight":
            mapped["position_embedding.weight"] = value
        elif name == "ln_f.weight":
            mapped["final_norm.weight"] = value
        elif name == "ln_f.bias":
            mapped["final_norm.bias"] = value
        elif name == "lm_head.weight":
            mapped["lm_head.weight"] = value
        elif name.startswith("h."):
            parts = name.split(".")
            if len(parts) < 4:
                continue
            layer_idx = parts[1]
            tail = ".".join(parts[2:])
            aliases = {
                "attn.c_attn.weight": "attention.qkv_proj.weight",
                "attn.c_attn.bias": "attention.qkv_proj.bias",
                "attn.c_proj.weight": "attention.o_proj.weight",
                "attn.c_proj.bias": "attention.o_proj.bias",
                "ln_1.weight": "input_norm.weight",
                "ln_1.bias": "input_norm.bias",
                "ln_2.weight": "post_attention_norm.weight",
                "ln_2.bias": "post_attention_norm.bias",
                "mlp.c_fc.weight": "feed_forward.up_proj.weight",
                "mlp.c_fc.bias": "feed_forward.up_proj.bias",
                "mlp.c_proj.weight": "feed_forward.down_proj.weight",
                "mlp.c_proj.bias": "feed_forward.down_proj.bias",
            }
            canonical = aliases.get(tail)
            if canonical is not None:
                mapped[f"layers.{layer_idx}.{canonical}"] = value
    return mapped


def canonical_llama_weights_from_hf(weights: dict[str, Any]) -> ModelWeightsIR:
    """
    Map Llama-family Hugging Face state dict names into canonical IR names.

    Args:
        weights: State dict mapping tensor names to tensor values.

    Returns:
        Canonical IR weight mapping.
    """

    mapped: ModelWeightsIR = {}
    global_aliases = {
        "tok_embeddings.weight": "token_embedding.weight",
        "model.embed_tokens.weight": "token_embedding.weight",
        "norm.weight": "final_norm.weight",
        "model.norm.weight": "final_norm.weight",
        "output.weight": "lm_head.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_aliases = {
        "attention.wq": "attention.q_proj",
        "attention.wk": "attention.k_proj",
        "attention.wv": "attention.v_proj",
        "attention.wo": "attention.o_proj",
        "self_attn.q_proj": "attention.q_proj",
        "self_attn.k_proj": "attention.k_proj",
        "self_attn.v_proj": "attention.v_proj",
        "self_attn.o_proj": "attention.o_proj",
        "self_attn.q_norm": "attention.q_norm",
        "self_attn.k_norm": "attention.k_norm",
        "attention_norm": "input_norm",
        "input_layernorm": "input_norm",
        "ffn_norm": "post_attention_norm",
        "post_attention_layernorm": "post_attention_norm",
        "feed_forward.w1": "feed_forward.gate_proj",
        "feed_forward.w3": "feed_forward.up_proj",
        "feed_forward.w2": "feed_forward.down_proj",
        "mlp.gate_proj": "feed_forward.gate_proj",
        "mlp.up_proj": "feed_forward.up_proj",
        "mlp.down_proj": "feed_forward.down_proj",
    }
    for source, value in weights.items():
        if not isinstance(value, (torch.Tensor, QuantizedWeight)):
            continue
        if source in global_aliases:
            mapped[global_aliases[source]] = value
            continue
        layer_name = source.removeprefix("model.")
        if not layer_name.startswith("layers."):
            continue
        parts = layer_name.split(".")
        if len(parts) < 4:
            continue
        layer_idx = parts[1]
        suffix = parts[-1]
        role = ".".join(parts[2:-1])
        canonical_role = layer_aliases.get(role)
        if canonical_role is not None:
            mapped[f"layers.{layer_idx}.{canonical_role}.{suffix}"] = value
    if "lm_head.weight" not in mapped and "token_embedding.weight" in mapped:
        mapped["lm_head.weight"] = mapped["token_embedding.weight"]
    return mapped


def _canonical_gemma3_weights(weights: dict[str, Any]) -> ModelWeightsIR:
    """
    Map Gemma3 Hugging Face state dict names into canonical IR names.

    Args:
        weights: State dict mapping tensor names to tensor values.

    Returns:
        Canonical IR weight mapping.
    """

    normalized = {
        key.removeprefix("model.language_model."): value
        for key, value in weights.items()
    }
    mapped = canonical_llama_weights_from_hf(normalized)
    layer_aliases = {
        "self_attn.q_norm": "attention.q_norm",
        "self_attn.k_norm": "attention.k_norm",
        "input_layernorm": "input_norm",
        "post_attention_layernorm": "post_attention_norm",
        "pre_feedforward_layernorm": "pre_ffn_norm",
        "post_feedforward_layernorm": "post_ffn_norm",
    }
    for source, value in normalized.items():
        if not isinstance(value, (torch.Tensor, QuantizedWeight)):
            continue
        layer_name = source.removeprefix("model.")
        if not layer_name.startswith("layers."):
            continue
        parts = layer_name.split(".")
        if len(parts) < 4:
            continue
        layer_idx = parts[1]
        suffix = parts[-1]
        role = ".".join(parts[2:-1])
        canonical_role = layer_aliases.get(role)
        if canonical_role is not None:
            mapped[f"layers.{layer_idx}.{canonical_role}.{suffix}"] = value
    return mapped


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """
    Load weight using safetensor.

    Args:
        path: Snapshot directory containing one or more safetensors files.

    Returns:
        State dict loaded from all safetensors files in the directory.
    """
    safetensors_torch = cast(Any, import_module("safetensors.torch"))
    load_safetensors_file = cast(
        Callable[[Path], dict[str, torch.Tensor]],
        safetensors_torch.load_file,
    )
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found in {path}")
    weights: dict[str, torch.Tensor] = {}
    for file in files:
        tensors = load_safetensors_file(file)
        overlap = set(weights).intersection(tensors)
        if overlap:
            names = ", ".join(sorted(overlap)[:3])
            raise ValueError(
                f"duplicate tensor names across safetensors files: {names}"
            )
        weights.update(tensors)
    return weights


def _int(config: dict[str, Any], key: str, *, fallback_key: str | None = None) -> int:
    """
    Small helper used to set value in the produced IR.

    Args:
        config: Source or normalized model configuration mapping.
        key: Primary config key to read.
        fallback_key: Optional config key used when ``key`` is absent.

    Returns:
        Integer config value.
    """
    value = config.get(key)
    if value is None and fallback_key is not None:
        value = config.get(fallback_key)
    if not isinstance(value, int):
        raise ValueError(f"config value {key!r} must be an int")
    return value


def _float(config: dict[str, Any], key: str, *, default: float | None = None) -> float:
    """
    Small helper used to set value in the produced IR.

    Args:
        config: Source or normalized model configuration mapping.
        key: Config key to read.
        default: Value returned when the key is absent.

    Returns:
        Floating-point config value.
    """
    value = config.get(key)
    if value is None and default is not None:
        return default
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"config value {key!r} must be a float")
    return value


def _rope_theta(config: dict[str, Any], *, default: float) -> float:
    """Return RoPE theta from old ``rope_theta`` or new ``rope_parameters``."""
    value = config.get("rope_theta")
    if value is not None:
        if isinstance(value, int):
            return float(value)
        if isinstance(value, float):
            return value
        raise ValueError("config value 'rope_theta' must be a float")

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        rope_parameters = cast(dict[str, Any], rope_parameters)
        theta = rope_parameters.get("rope_theta")
        if isinstance(theta, int):
            return float(theta)
        if isinstance(theta, float):
            return theta

    return default


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    """
    Small helper used to set value in the produced IR.

    Args:
        config: Source or normalized model configuration mapping.
        key: Config key to read.

    Returns:
        Floating-point config value, or ``None`` when absent.
    """
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"config value {key!r} must be a float or None")
    return value


def _bool(config: dict[str, Any], key: str, *, default: bool) -> bool:
    """
    Small helper used to set value in the produced IR.

    Args:
        config: Source or normalized model configuration mapping.
        key: Config key to read.
        default: Value returned when the key is absent.

    Returns:
        Boolean config value.
    """
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"config value {key!r} must be a bool")
    return value


def _list_str(config: dict[str, Any], key: str, *, default: list[str]) -> list[str]:
    """
    Small helper used to set value in the produced IR.

    Args:
        config: Source or normalized model configuration mapping.
        key: Config key to read.
        default: Value returned when the key is absent.

    Returns:
        List of string config values.
    """
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"config value {key!r} must be a list[str]")
    value = cast(list[Any], value)
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"config value {key!r} must be a list[str]")
    return cast(list[str], value)


def _hidden_dim(config: dict[str, Any]) -> int:
    """
    Compute Llama intermediate size from Meta-style config fields when needed.

    Args:
        config: Source or normalized model configuration mapping.

    Returns:
        Feed-forward hidden dimension.
    """
    intermediate_size = config.get("intermediate_size")
    if isinstance(intermediate_size, int):
        return intermediate_size
    dim = _int(config, "dim")
    multiple_of = _int(config, "multiple_of")
    hidden_dim = int(2 * (4 * dim) / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def _rope_base(config: dict[str, Any], layer_type: str, default: float) -> float:
    """
    Resolve Gemma3 RoPE base for a local or global attention layer type.

    Args:
        config: Source or normalized model configuration mapping.
        layer_type: Gemma3 layer type key inside ``rope_parameters``.
        default: Value returned when no layer-specific base is present.

    Returns:
        RoPE base frequency.
    """
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        layer_parameters = cast(dict[str, Any], rope_parameters).get(layer_type)
        if isinstance(layer_parameters, dict):
            value = cast(dict[str, Any], layer_parameters).get("rope_theta")
            if isinstance(value, int):
                return float(value)
            if isinstance(value, float):
                return value
    return _float(config, "rope_theta", default=default)
