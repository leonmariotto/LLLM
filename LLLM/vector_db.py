"""VectorDB storage and command line builder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import click
from loguru import logger
import torch

from .fetch import fetch_embedding_model_ir
from .sentence_transformer import SentenceTransformerEmbedder
from .vector_search import SearchResult, TextEmbedder, chunk_text, vector_search
from .yaml_parser import YamlParser, YamlParserError


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorDBRecord(TypedDict):
    """
    A single record in VectorDB.
    """

    embedding: list[float]
    text: str
    metadata: list[str]


class VectorDB:
    """
    Small JSON-serializable vector database.
    Support filtering via metadata entries.

    2 init:
        db = VectorDB(embedder) # initiate an empty VectorDB, .add_text can be used
                                # to populate it.
        db = VectorDB.load(in_file, embedder) # load a VectorDB file.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        records: Sequence[VectorDBRecord] | None = None,
    ) -> None:
        self.embedder = embedder
        self.records = list(records or [])

    @classmethod
    def load(cls, path: str | Path, embedder: TextEmbedder) -> VectorDB:
        """
        Load database from a file.
        """
        load_path = Path(path)
        logger.info("Loading VectorDB: path={}", load_path)
        with load_path.open("r", encoding="utf-8") as vector_db_file:
            raw_json = cast(object, json.load(vector_db_file))
        if not isinstance(raw_json, list):
            raise ValueError("VectorDB file must contain a list of records")

        raw_records = cast(list[object], raw_json)
        records = [_validate_vector_db_record(record) for record in raw_records]
        logger.info("Loaded VectorDB: path={}, records={}", load_path, len(records))
        return cls(embedder, records)

    def add_text(
        self,
        text: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
        metadata: Sequence[str] | None = None,
    ) -> None:
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        logger.info(
            "Adding text to VectorDB: text_chars={}, chunks={}, metadata={}",
            len(text),
            len(chunks),
            list(metadata or []),
        )
        if not chunks:
            return

        chunk_embeddings = self.embedder.embed_batch(chunks)
        if chunk_embeddings.dim() != 2:
            raise ValueError(
                "embedder.embed_batch must return vectors with shape [items, dim]"
            )
        if chunk_embeddings.shape[0] != len(chunks):
            raise ValueError("embedder.embed_batch must return one vector per chunk")

        record_metadata = list(metadata or [])
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
            embedding_values = cast(
                list[float],
                embedding.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            self.records.append(
                {
                    "embedding": [float(value) for value in embedding_values],
                    "text": chunk,
                    "metadata": record_metadata.copy(),
                }
            )
        logger.info("VectorDB now contains {} records", len(self.records))

    def export(self, path: str | Path) -> None:
        """
        Save the database to a file.
        """
        export_path = Path(path)
        logger.info(
            "Exporting VectorDB: path={}, records={}",
            export_path,
            len(self.records),
        )
        with export_path.open("w", encoding="utf-8") as vector_db_file:
            json.dump(self.records, vector_db_file, indent=2)
            vector_db_file.write("\n")

    def search(
        self,
        query_str: str,
        metadata_filter: Sequence[str] | None = None,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        filter_values = list(metadata_filter or [])
        matching_records = [
            (index, record)
            for index, record in enumerate(self.records)
            if _record_matches_metadata_filter(record, filter_values)
        ]
        logger.info(
            "Searching VectorDB: query_chars={}, metadata_filter={}, candidates={}",
            len(query_str),
            filter_values,
            len(matching_records),
        )
        if not matching_records or top_k == 0:
            logger.info("VectorDB search returned 0 results")
            return []

        record_indices = [index for index, _ in matching_records]
        embeddings = torch.tensor(
            [record["embedding"] for _, record in matching_records],
            dtype=torch.float32,
        )
        sequences = [record["text"] for _, record in matching_records]
        results = vector_search(
            query_str, embeddings, sequences, self.embedder, top_k=top_k
        )
        remapped_results = [
            SearchResult(
                index=record_indices[result.index],
                score=result.score,
                sequence=result.sequence,
            )
            for result in results
        ]
        logger.info("VectorDB search returned {} results", len(remapped_results))
        return remapped_results


class VectorDBFileEntry(TypedDict):
    file_path: Path
    chunk_size: int
    chunk_overlap: int
    metadata: list[str]


def _build_embedder(model: str) -> TextEmbedder:
    ir = fetch_embedding_model_ir(model)
    # TODO LMA: should check the model family before calling .from_ir.
    return SentenceTransformerEmbedder.from_ir(ir)


def _load_file_entries(yaml_path: Path) -> list[VectorDBFileEntry]:
    parser = YamlParser()
    try:
        parser.parse(str(yaml_path))
    except YamlParserError as err:
        raise click.UsageError(f"failed to parse YAML file {yaml_path}") from err

    raw_files = cast(object, parser.data.get("files"))
    if not isinstance(raw_files, list):
        raise click.UsageError("YAML must contain a 'files' list")

    file_entries = cast(list[object], raw_files)
    base_dir = yaml_path.parent
    return [
        _validate_file_entry(raw_entry, index, base_dir)
        for index, raw_entry in enumerate(file_entries)
    ]


def _validate_file_entry(
    raw_entry: object,
    index: int,
    base_dir: Path,
) -> VectorDBFileEntry:
    if not isinstance(raw_entry, dict):
        raise click.UsageError(f"files[{index}] must be a dictionary")
    entry = cast(dict[str, Any], raw_entry)

    raw_file_path = cast(object, entry.get("file_path"))
    raw_chunk_size = cast(object, entry.get("chunk_size"))
    raw_chunk_overlap = cast(object, entry.get("chunk_overlap"))
    raw_metadata = cast(object, entry.get("metadata"))

    if not isinstance(raw_file_path, str) or raw_file_path == "":
        raise click.UsageError(f"files[{index}].file_path must be a non-empty string")
    chunk_size = _parse_yaml_int(raw_chunk_size, f"files[{index}].chunk_size")
    chunk_overlap = _parse_yaml_int(
        raw_chunk_overlap,
        f"files[{index}].chunk_overlap",
    )
    if chunk_size <= 0:
        raise click.UsageError(f"files[{index}].chunk_size must be positive")
    if chunk_overlap < 0:
        raise click.UsageError(f"files[{index}].chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise click.UsageError(
            f"files[{index}].chunk_overlap must be smaller than chunk_size"
        )
    if not isinstance(raw_metadata, list):
        raise click.UsageError(f"files[{index}].metadata must be a list of strings")
    metadata = cast(list[object], raw_metadata)
    if not all(isinstance(value, str) for value in metadata):
        raise click.UsageError(f"files[{index}].metadata must be a list of strings")

    file_path = Path(raw_file_path).expanduser()
    if not file_path.is_absolute():
        file_path = base_dir / file_path

    return {
        "file_path": file_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata": cast(list[str], metadata).copy(),
    }


def _parse_yaml_int(value: object, field_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as err:
            raise click.UsageError(f"{field_name} must be an integer") from err
    raise click.UsageError(f"{field_name} must be an integer")


def _validate_vector_db_record(record: object) -> VectorDBRecord:
    if not isinstance(record, dict):
        raise ValueError("VectorDB record must be a dictionary")

    record_dict = cast(dict[str, object], record)
    text = record_dict.get("text")
    embedding = record_dict.get("embedding")
    metadata = record_dict.get("metadata")
    if not isinstance(text, str):
        raise ValueError("VectorDB record text must be a string")
    if not isinstance(embedding, list):
        raise ValueError("VectorDB record embedding must be a list of numbers")
    embedding_values = cast(list[object], embedding)
    if not all(isinstance(value, int | float) for value in embedding_values):
        raise ValueError("VectorDB record embedding must be a list of numbers")
    if not isinstance(metadata, list):
        raise ValueError("VectorDB record metadata must be a list of strings")
    metadata_values = cast(list[object], metadata)
    if not all(isinstance(value, str) for value in metadata_values):
        raise ValueError("VectorDB record metadata must be a list of strings")

    return {
        "embedding": [float(value) for value in cast(list[int | float], embedding)],
        "text": text,
        "metadata": cast(list[str], metadata).copy(),
    }


def _record_matches_metadata_filter(
    record: VectorDBRecord,
    metadata_filter: Sequence[str],
) -> bool:
    metadata = record["metadata"]
    return all(filter_value in metadata for filter_value in metadata_filter)


@click.command(help="Build a VectorDB JSON file from a YAML manifest.")
@click.option(
    "--yaml",
    "yaml_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML manifest containing a 'files' list.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path where the VectorDB JSON file will be written.",
)
@click.option(
    "--model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Embedding model repo id or local path.",
)
def vector_db_builder_cli(
    yaml_path: Path,
    out_path: Path,
    model: str,
) -> None:
    logger.info(
        "Building VectorDB from YAML: yaml={}, out={}, model={}",
        yaml_path,
        out_path,
        model,
    )
    entries = _load_file_entries(yaml_path)
    logger.info("Loaded {} file entries from YAML", len(entries))

    embedder = _build_embedder(model)
    db = VectorDB(embedder)
    for entry in entries:
        source_path = entry["file_path"]
        logger.info("Adding file to VectorDB: path={}", source_path)
        try:
            text = source_path.read_text(encoding="utf-8")
        except OSError as err:
            raise click.UsageError(f"failed to read source file {source_path}") from err
        db.add_text(
            text,
            chunk_size=entry["chunk_size"],
            chunk_overlap=entry["chunk_overlap"],
            metadata=entry["metadata"],
        )

    db.export(out_path)
    logger.info("Exported VectorDB: path={}, records={}", out_path, len(db.records))
    click.echo(f"Exported {len(db.records)} VectorDB records to {out_path}")
