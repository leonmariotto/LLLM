"""
Vector Search module.

Small tensor utilities for cosine vector search.
Provide a pre-computed API for searching into existing embedding.
Provide an API that build an embedding from a text and search into it.
Provide VectorDB: a class to build, load, save and use simple vector database.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

from loguru import logger
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SearchResult:
    """One vector search hit."""

    index: int
    score: float
    sequence: str


class TextEmbedder(Protocol):
    """Minimal text embedding interface."""

    def embed(self, text: str) -> torch.Tensor: ...

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor: ...


def cosine_similarity(query: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between query vector(s) and candidate vectors.

    Args:
        query: Tensor with shape ``[dim]`` or ``[queries, dim]``.
        vectors: Tensor with shape ``[items, dim]``.

    Returns:
        Similarity tensor with shape ``[items]`` for a single query, or
        ``[queries, items]`` for batched queries.
    """

    query_was_vector = query.dim() == 1
    if query_was_vector:
        query = query.unsqueeze(0)
    if query.dim() != 2:
        raise ValueError("query must have shape [dim] or [queries, dim]")
    if vectors.dim() != 2:
        raise ValueError("vectors must have shape [items, dim]")
    if query.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            f"dimension mismatch: query dim {query.shape[-1]} != "
            f"vectors dim {vectors.shape[-1]}"
        )

    query_norm = F.normalize(query, p=2, dim=-1)
    vectors_norm = F.normalize(vectors, p=2, dim=-1)
    scores = query_norm @ vectors_norm.T
    if query_was_vector:
        return scores.squeeze(0)
    return scores


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if text == "":
        return []

    step = chunk_size - chunk_overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def _rank_embeddings(
    query: str,
    sequences: Sequence[str],
    embedding_vectors: torch.Tensor,
    embedder: TextEmbedder,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not sequences or top_k == 0:
        return []
    if embedding_vectors.dim() != 2:
        raise ValueError("embedding_vectors must have shape [items, dim]")
    if embedding_vectors.shape[0] != len(sequences):
        raise ValueError("embedding_vectors must align with sequences")

    query_embedding = embedder.embed(query)
    if query_embedding.dim() != 1:
        raise ValueError("embedder.embed must return a vector with shape [dim]")

    scores = cosine_similarity(query_embedding, embedding_vectors)
    limit = min(top_k, len(sequences))
    values, indices = torch.topk(scores, k=limit)
    return [
        SearchResult(
            index=int(index),
            score=float(score),
            sequence=sequences[int(index)],
        )
        for score, index in zip(values, indices, strict=True)
    ]


def vector_search(
    query: str,
    embedding_vectors: torch.Tensor,
    sequences: Sequence[str],
    embedder: TextEmbedder,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search a query string into an existing embedding batch.
    This function can be used to search into precomputed vectors.
    We can't recover text from embedding, so texts matching embedding must be
    included in the API.

    Args:
        query: Query text.
        embedding_vectors: Precomputed candidate embeddings with shape
            ``[items, dim]``.
        sequences: Candidate text sequences aligned with ``embedding_vectors``.
        embedder: Object providing ``embed`` for the query.
        top_k: Maximum number of matches to return.

    Returns:
        Results sorted by descending similarity.
    """

    return _rank_embeddings(query, sequences, embedding_vectors, embedder, top_k=top_k)


def vector_build_and_search(
    query: str,
    text: str,
    embedder: TextEmbedder,
    *,
    top_k: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> list[SearchResult]:
    """
    Chunk text, embed each chunk, and return top cosine-similarity matches.

    Args:
        query: Query text.
        text: Text to split into candidate chunks.
        embedder: Object providing ``embed`` and ``embed_batch``.
        top_k: Maximum number of matches to return.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Number of characters shared by adjacent chunks.

    Returns:
        Results sorted by descending similarity.
    """

    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    if not chunks or top_k == 0:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        return []

    chunk_embeddings = embedder.embed_batch(chunks)
    return _rank_embeddings(query, chunks, chunk_embeddings, embedder, top_k=top_k)


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
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
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
