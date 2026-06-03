"""
Fetch or locate model artifacts and call parser functions to load them into
ModelIR.
No format decode/encode here, only fetching.
Entrypoint is fetch_model_ir.
"""

from __future__ import annotations

import logging

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

from .model_ir import ModelIR
from .quantization import WeightMode


def load_model_ir(
    path: str | Path,
    *,
    gguf_filename: str | None = None,
    source_format: str | None = None,
    weight_mode: WeightMode = "dense",
) -> ModelIR:
    """
    Dispatch a local artifact path to the matching format loader
    and return IR.

    Args:
        path: Local model artifact path, snapshot directory, or GGUF file.
        gguf_filename: GGUF file to select when ``path`` is a directory.
        source_format: Explicit loader format override, such as ``"huggingface"``
            or ``"gguf"``.
        weight_mode: Whether GGUF linear weights are loaded densely or preserved
            as quantized weights.

    Returns:
        Source-independent model IR.
    """

    source_path = Path(path).expanduser()
    inferred_format = source_format
    if inferred_format is None:
        if source_path.is_file() and source_path.suffix.lower() == ".gguf":
            inferred_format = "gguf"
        elif gguf_filename is not None:
            inferred_format = "gguf"
        elif source_path.is_dir() and not (source_path / "config.json").is_file():
            if list(source_path.glob("*.gguf")):
                inferred_format = "gguf"
            else:
                inferred_format = "huggingface"
        else:
            inferred_format = "huggingface"

    if inferred_format in {"hf", "huggingface"}:
        if weight_mode == "quantized":
            raise NotImplementedError(
                "quantized loading is currently supported for GGUF"
            )
        from .hf_loader import load_hf_model_ir

        return load_hf_model_ir(source_path)
    if inferred_format == "gguf":
        from .gguf import load_gguf_ir

        return load_gguf_ir(source_path, gguf_filename, weight_mode=weight_mode)
    raise ValueError(f"unsupported source_format {source_format!r}")


def fetch_model_ir(
    repo_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    gguf_filename: str | None = None,
    weight_mode: WeightMode = "dense",
) -> ModelIR:
    """
    Fetch or locate a model repository and parse it into ``ModelIR``.

    Args:
        repo_id: Hugging Face repository id or local filesystem path.
        revision: Optional Hugging Face revision to download.
        local_files_only: Restrict downloads to the Hugging Face cache.
        gguf_filename: GGUF file to download or select from a local directory.
        weight_mode: Whether GGUF linear weights are loaded densely or preserved
            as quantized weights.

    Returns:
        Source-independent model IR.
    """

    local_path = Path(repo_id).expanduser()
    if local_path.exists():
        return load_model_ir(
            local_path, gguf_filename=gguf_filename, weight_mode=weight_mode
        )
    path = _download_snapshot(
        repo_id,
        revision=revision,
        local_files_only=local_files_only,
        gguf_filename=gguf_filename,
    )
    return load_model_ir(path, gguf_filename=gguf_filename, weight_mode=weight_mode)


def _download_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
    gguf_filename: str | None,
) -> Path:
    """
    Download only files needed by the supported loaders.

    Args:
        repo_id: Hugging Face repository id.
        revision: Optional repository revision to download.
        local_files_only: Restrict downloads to the Hugging Face cache.
        gguf_filename: GGUF file to download instead of HF-format artifacts.

    Returns:
        Local snapshot directory path.
    """

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
