"""Fetch Hugging Face model repos and load cached model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

import torch


@dataclass(frozen=True)
class FetchedModel:
    """A cached Hugging Face snapshot plus its decoded config and weights."""

    path: Path
    config: dict[str, Any]
    weights: dict[str, torch.Tensor]

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
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> FetchedModel:
    """
    Download or reuse a Hugging Face model snapshot and load common artifacts.

    The cache is managed by huggingface_hub. Passing the same repo_id, revision,
    and cache_dir reuses previously downloaded files.
    """
    local_path = Path(repo_id).expanduser()
    if local_path.is_dir():
        return load_cached_model(local_path)

    path = _download_snapshot(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return load_cached_model(path)


def load_cached_model(path: str | Path) -> FetchedModel:
    """Load config and safetensors weights from an already cached snapshot path."""
    snapshot_path = Path(path)
    config_path = snapshot_path / "config.json"
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
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Path:
    hf_hub = cast(Any, import_module("huggingface_hub"))
    download = cast(Callable[..., str], hf_hub.snapshot_download)
    path = download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
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
