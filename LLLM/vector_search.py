"""
Small tensor utilities for cosine vector search and high level vector
search API.
Provide a pre-computed API for searching into existing embedding.
Provide an API that build an embedding from a text and search into it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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
