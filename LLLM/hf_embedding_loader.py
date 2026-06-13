"""
Hugging Face SentenceTransformer loader producing ``EmbeddingModelIR``.
Only decode function (only HF -> IR, no IR -> HF)
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch

from .embedding_model_ir import (
    EmbeddingArchitectureId,
    EmbeddingModelConfigIR,
    EmbeddingModelIR,
    EmbeddingModelWeightsIR,
)


def load_hf_embedding_model_ir(
    path: str | Path,
    *,
    architecture: EmbeddingArchitectureId | None = None,
) -> EmbeddingModelIR:
    """
    Load a Hugging Face SentenceTransformer snapshot as embedding IR.

    Args:
        path: Snapshot directory containing config, tokenizer, module metadata,
            pooling metadata, and safetensors.
        architecture: Optional architecture override.

    Returns:
        Source-independent embedding model IR.
    """

    snapshot_path = Path(path).expanduser()
    config = _read_json_object(snapshot_path / "config.json")
    modules = _read_json_value(snapshot_path / "modules.json")
    pooling = _read_json_object(snapshot_path / "1_Pooling" / "config.json")
    weights = _load_safetensors(snapshot_path)
    tokenizer = _load_tokenizer_metadata(snapshot_path)
    return embedding_model_ir_from_hf(
        config,
        weights,
        modules=modules,
        pooling=pooling,
        tokenizer=tokenizer,
        path=snapshot_path,
        architecture=architecture,
    )


def embedding_model_ir_from_hf(
    config: dict[str, Any],
    weights: dict[str, Any],
    *,
    modules: Any | None = None,
    pooling: dict[str, Any] | None = None,
    tokenizer: dict[str, Any] | None = None,
    path: str | Path | None = None,
    architecture: EmbeddingArchitectureId | None = None,
) -> EmbeddingModelIR:
    """
    Translate decoded HF/SentenceTransformer artifacts into embedding IR.

    Args:
        config: Hugging Face encoder config.
        weights: State dict mapping tensor names to tensors.
        modules: Decoded SentenceTransformer ``modules.json`` value.
        pooling: Decoded SentenceTransformer pooling config.
        tokenizer: Decoded tokenizer metadata and tokenizer JSON payload.
        path: Optional source path stored in IR metadata.
        architecture: Optional architecture override.

    Returns:
        Source-independent embedding model IR.
    """

    arch = architecture or _infer_embedding_architecture(config, modules)
    normalized_pooling = _normalize_pooling(modules, pooling or {})
    normalized_config = _normalize_config(config, normalized_pooling, arch)
    ir_weights = _canonical_bert_sentence_transformer_weights(weights)
    metadata: dict[str, Any] = {
        "source_format": "huggingface_sentence_transformer",
        "original_architecture": config.get("model_type"),
    }
    if path is not None:
        metadata["path"] = str(path)
    return EmbeddingModelIR(
        architecture=arch,
        config=EmbeddingModelConfigIR(normalized_config),
        weights=ir_weights,
        tokenizer={} if tokenizer is None else dict(tokenizer),
        pooling=dict(normalized_pooling),
        metadata=metadata,
    )


def _infer_embedding_architecture(
    config: dict[str, Any], modules: Any | None
) -> EmbeddingArchitectureId:
    if config.get("model_type") != "bert":
        raise ValueError(
            f"cannot infer supported embedding architecture from "
            f"model_type={config.get('model_type')!r}"
        )
    _validate_sentence_transformer_modules(modules)
    return "bert_sentence_transformer"


def _normalize_config(
    config: dict[str, Any],
    pooling: dict[str, Any],
    arch: EmbeddingArchitectureId,
) -> dict[str, Any]:
    if arch != "bert_sentence_transformer":
        raise AssertionError(f"unhandled embedding architecture {arch}")
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("only absolute BERT position embeddings are supported")
    hidden_act = str(config.get("hidden_act", "gelu"))
    if hidden_act != "gelu":
        raise ValueError(f"unsupported BERT hidden_act {hidden_act!r}")
    return {
        "vocab_size": _int(config, "vocab_size"),
        "context_length": _int(config, "max_position_embeddings"),
        "hidden_size": _int(config, "hidden_size"),
        "intermediate_size": _int(config, "intermediate_size"),
        "num_hidden_layers": _int(config, "num_hidden_layers"),
        "num_attention_heads": _int(config, "num_attention_heads"),
        "type_vocab_size": _int(config, "type_vocab_size"),
        "layer_norm_eps": _float(config, "layer_norm_eps", default=1e-12),
        "hidden_act": hidden_act,
        "pad_token_id": _int(config, "pad_token_id"),
        "pooling_mode": pooling["pooling_mode"],
        "normalize_embeddings": bool(pooling["normalize_embeddings"]),
    }


def _normalize_pooling(
    modules: Any | None,
    pooling: dict[str, Any],
) -> dict[str, Any]:
    _validate_sentence_transformer_modules(modules)
    if pooling.get("pooling_mode_mean_tokens") is not True:
        raise ValueError("only SentenceTransformer mean-token pooling is supported")
    unsupported = {
        "pooling_mode_cls_token": "CLS-token pooling",
        "pooling_mode_max_tokens": "max-token pooling",
        "pooling_mode_mean_sqrt_len_tokens": "mean-sqrt-length pooling",
    }
    enabled = [label for key, label in unsupported.items() if pooling.get(key) is True]
    if enabled:
        raise ValueError(
            f"unsupported SentenceTransformer pooling: {', '.join(enabled)}"
        )
    return {
        "pooling_mode": "mean_tokens",
        "normalize_embeddings": _has_module_type(
            modules, "sentence_transformers.models.Normalize"
        ),
        "word_embedding_dimension": _int(pooling, "word_embedding_dimension"),
    }


def _validate_sentence_transformer_modules(modules: Any | None) -> None:
    if modules is None:
        raise ValueError("missing SentenceTransformer modules metadata")
    if not isinstance(modules, list):
        raise ValueError("SentenceTransformer modules metadata must be a list")
    typed_modules = cast(list[object], modules)
    if not _has_module_type(typed_modules, "sentence_transformers.models.Transformer"):
        raise ValueError("missing SentenceTransformer Transformer module")
    if not _has_module_type(typed_modules, "sentence_transformers.models.Pooling"):
        raise ValueError("missing SentenceTransformer Pooling module")


def _has_module_type(modules: Any | None, module_type: str) -> bool:
    if not isinstance(modules, list):
        return False
    typed_modules = cast(list[object], modules)
    for module in typed_modules:
        if not isinstance(module, dict):
            continue
        typed_module = cast(dict[str, object], module)
        if typed_module.get("type") == module_type:
            return True
    return False


def _canonical_bert_sentence_transformer_weights(
    weights: dict[str, Any],
) -> EmbeddingModelWeightsIR:
    mapped: EmbeddingModelWeightsIR = {}
    global_aliases = {
        "embeddings.word_embeddings.weight": "embeddings.token.weight",
        "embeddings.position_embeddings.weight": "embeddings.position.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type.weight",
        "embeddings.LayerNorm.weight": "embeddings.norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.norm.bias",
    }
    layer_aliases = {
        "attention.self.query": "attention.q_proj",
        "attention.self.key": "attention.k_proj",
        "attention.self.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "attention.output.LayerNorm": "attention.output_norm",
        "intermediate.dense": "feed_forward.up_proj",
        "output.dense": "feed_forward.down_proj",
        "output.LayerNorm": "feed_forward.output_norm",
    }
    for source, value in weights.items():
        if not isinstance(value, torch.Tensor):
            continue
        if source in global_aliases:
            mapped[global_aliases[source]] = value
            continue
        if not source.startswith("encoder.layer."):
            continue
        parts = source.split(".")
        if len(parts) < 6:
            continue
        layer_idx = parts[2]
        suffix = parts[-1]
        role = ".".join(parts[3:-1])
        canonical_role = layer_aliases.get(role)
        if canonical_role is not None:
            mapped[f"layers.{layer_idx}.{canonical_role}.{suffix}"] = value
    return mapped


def _load_tokenizer_metadata(path: Path) -> dict[str, Any]:
    tokenizer: dict[str, Any] = {}
    tokenizer_json_path = path / "tokenizer.json"
    if tokenizer_json_path.is_file():
        tokenizer["tokenizer_json"] = tokenizer_json_path.read_text(encoding="utf-8")
    for filename, key in (
        ("tokenizer_config.json", "tokenizer_config"),
        ("special_tokens_map.json", "special_tokens_map"),
    ):
        file_path = path / filename
        if file_path.is_file():
            tokenizer[key] = _read_json_object(file_path)
    return tokenizer


def _read_json_value(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path.name} in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(dict[str, Any], value)


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
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


def _int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"config value {key!r} must be an int")
    return value


def _float(config: dict[str, Any], key: str, *, default: float | None = None) -> float:
    value = config.get(key)
    if value is None and default is not None:
        return default
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise ValueError(f"config value {key!r} must be a float")
    return value
