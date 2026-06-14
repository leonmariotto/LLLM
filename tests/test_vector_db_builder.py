from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner
import pytest
import torch

from ..LLLM import vector_db as builder_module


class FakeEmbedder:
    def embed(self, text: str) -> torch.Tensor:
        return torch.tensor([float(len(text)), 0.0])

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack(
            [
                torch.tensor([float(index + 1), 0.0])
                for index, _ in enumerate(texts)
            ]
        )


def _patch_embedder(monkeypatch: pytest.MonkeyPatch, models: list[str]) -> None:
    def fake_build_embedder(model: str) -> FakeEmbedder:
        models.append(model)
        return FakeEmbedder()

    monkeypatch.setattr(builder_module, "_build_embedder", fake_build_embedder)


def test_vector_db_builder_cli_builds_json_from_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models: list[str] = []
    _patch_embedder(monkeypatch, models)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    out = tmp_path / "vectors.json"
    yaml_path = tmp_path / "manifest.yaml"
    first.write_text("abcdefghij", encoding="utf-8")
    second.write_text("klmnopqrst", encoding="utf-8")
    yaml_path.write_text(
        """
files:
  - file_path: first.txt
    chunk_size: 5
    chunk_overlap: 0
    metadata:
      - first
      - public
  - file_path: second.txt
    chunk_size: 4
    chunk_overlap: 1
    metadata:
      - second
""".lstrip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        builder_module.vector_db_builder_cli,
        ["--yaml", str(yaml_path), "--out", str(out), "--model", "custom-model"],
    )

    assert result.exit_code == 0, result.output
    assert models == ["custom-model"]
    records = cast(list[dict[str, Any]], json.loads(out.read_text(encoding="utf-8")))
    assert out.read_text(encoding="utf-8").startswith("[\n  {")
    assert [record["text"] for record in records] == [
        "abcde",
        "fghij",
        "klmn",
        "nopq",
        "qrst",
        "t",
    ]
    assert records[0]["metadata"] == ["first", "public"]
    assert records[2]["metadata"] == ["second"]
    assert "Exported 6 VectorDB records" in result.output


def test_vector_db_builder_cli_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models: list[str] = []
    _patch_embedder(monkeypatch, models)
    source = tmp_path / "source.txt"
    out = tmp_path / "vectors.json"
    yaml_path = tmp_path / "manifest.yaml"
    source.write_text("abcde", encoding="utf-8")
    yaml_path.write_text(
        """
files:
  - file_path: source.txt
    chunk_size: 5
    chunk_overlap: 0
    metadata:
      - source
""".lstrip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        builder_module.vector_db_builder_cli,
        ["--yaml", str(yaml_path), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert models == [builder_module.DEFAULT_EMBEDDING_MODEL]


def test_vector_db_builder_cli_rejects_missing_files_list(tmp_path: Path) -> None:
    yaml_path = tmp_path / "manifest.yaml"
    out = tmp_path / "vectors.json"
    yaml_path.write_text("documents: none\n", encoding="utf-8")

    result = CliRunner().invoke(
        builder_module.vector_db_builder_cli,
        ["--yaml", str(yaml_path), "--out", str(out)],
    )

    assert result.exit_code != 0
    assert "YAML must contain a 'files' list" in result.output


def test_vector_db_builder_cli_rejects_malformed_entry(tmp_path: Path) -> None:
    yaml_path = tmp_path / "manifest.yaml"
    out = tmp_path / "vectors.json"
    yaml_path.write_text(
        """
files:
  - file_path: source.txt
    chunk_size: 5
    chunk_overlap: 5
    metadata:
      - source
""".lstrip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        builder_module.vector_db_builder_cli,
        ["--yaml", str(yaml_path), "--out", str(out)],
    )

    assert result.exit_code != 0
    assert "chunk_overlap must be smaller than chunk_size" in result.output


def test_vector_db_builder_cli_rejects_missing_source_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models: list[str] = []
    _patch_embedder(monkeypatch, models)
    yaml_path = tmp_path / "manifest.yaml"
    out = tmp_path / "vectors.json"
    yaml_path.write_text(
        """
files:
  - file_path: missing.txt
    chunk_size: 5
    chunk_overlap: 0
    metadata:
      - missing
""".lstrip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        builder_module.vector_db_builder_cli,
        ["--yaml", str(yaml_path), "--out", str(out)],
    )

    assert result.exit_code != 0
    assert "failed to read source file" in result.output
