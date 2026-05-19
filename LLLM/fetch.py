"""Fetch Hugging Face model repos and load cached model artifacts."""

from __future__ import annotations

import logging

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch

from .quantization import WeightMode


@dataclass(frozen=True)
class FetchedModel:
    """A cached model snapshot plus its decoded config and loaded weights."""

    path: Path
    config: dict[str, Any]
    weights: dict[str, Any]

    @property
    def model_type(self) -> str:
        model_type = self.config.get("model_type")
        if not isinstance(model_type, str):
            raise ValueError("config.json does not contain a string model_type")
        return model_type


def fetch_hf_model(
    repo_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    gguf_filename: str | None = None,
    weight_mode: WeightMode = "dense",
) -> FetchedModel:
    """
    Download or reuse a Hugging Face model snapshot and load common artifacts.

    The cache is managed by huggingface_hub. Use default system cache dir.
    """
    local_path = Path(repo_id).expanduser()
    if local_path.is_dir():
        return load_cached_model(
            local_path, gguf_filename=gguf_filename, weight_mode=weight_mode
        )
    if local_path.is_file() and local_path.suffix.lower() == ".gguf":
        return load_cached_model(local_path, weight_mode=weight_mode)

    path = _download_snapshot(
        repo_id,
        revision=revision,
        local_files_only=local_files_only,
        gguf_filename=gguf_filename,
    )
    return load_cached_model(path, gguf_filename=gguf_filename, weight_mode=weight_mode)


def load_cached_model(
    path: str | Path,
    *,
    gguf_filename: str | None = None,
    weight_mode: WeightMode = "dense",
) -> FetchedModel:
    """Load config and weights from an already cached snapshot path."""
    snapshot_path = Path(path)
    if snapshot_path.is_file() and snapshot_path.suffix.lower() == ".gguf":
        from .gguf import load_gguf, load_gguf_quantized

        if weight_mode == "quantized":
            config, weights = load_gguf_quantized(snapshot_path)
        else:
            config, weights = load_gguf(snapshot_path)
        return FetchedModel(path=snapshot_path, config=config, weights=weights)

    config_path = snapshot_path / "config.json"
    gguf_files = sorted(snapshot_path.glob("*.gguf"))
    if gguf_filename is not None or (gguf_files and not config_path.is_file()):
        from .gguf import load_gguf, load_gguf_quantized

        if weight_mode == "quantized":
            config, weights = load_gguf_quantized(snapshot_path, gguf_filename)
        else:
            config, weights = load_gguf(snapshot_path, gguf_filename)
        return FetchedModel(path=snapshot_path, config=config, weights=weights)

    if weight_mode == "quantized":
        raise NotImplementedError("quantized loading is currently supported for GGUF")

    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {snapshot_path}")

    raw_config: object = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} did not contain a JSON object")
    config = cast(dict[str, Any], raw_config)

    weights = _load_safetensors(snapshot_path)
    return FetchedModel(path=snapshot_path, config=config, weights=weights)


def _download_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
    gguf_filename: str | None,
) -> Path:
    hf_hub = cast(Any, import_module("huggingface_hub"))
    download = cast(Callable[..., str], hf_hub.snapshot_download)
    logging.debug("repo_id=%s revision=%s", repo_id, revision)
    allow_patterns = [
        "config.json",
        "*.safetensors",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    if gguf_filename is not None:
        allow_patterns = [gguf_filename]
    else:
        allow_patterns.append("*.gguf")

    path = download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
    )
    return Path(path)


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
