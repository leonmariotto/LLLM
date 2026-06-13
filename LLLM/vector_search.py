"""
Small tensor utilities for cosine vector search and  high level vector
search API.
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
    """Minimal text embedding interface required by ``vector_search``."""

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


def vector_search(
    query: str,
    sequences: Sequence[str],
    embedder: TextEmbedder,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Embed a string query and return top cosine-similarity sequence matches.

    Args:
        query: Query text.
        sequences: Candidate text sequences to search.
        embedder: Object providing ``embed`` and ``embed_batch``.
        top_k: Maximum number of matches to return.

    Returns:
        Results sorted by descending similarity.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not sequences or top_k == 0:
        return []

    query_embedding = embedder.embed(query)
    candidate_embeddings = embedder.embed_batch(sequences)
    if query_embedding.dim() != 1:
        raise ValueError("embedder.embed must return a vector with shape [dim]")
    if candidate_embeddings.shape[0] != len(sequences):
        raise ValueError("embedder.embed_batch must return one vector per sequence")

    scores = cosine_similarity(query_embedding, candidate_embeddings)
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


def vector_search_into(
    query: str,
    embedding_vectors: torch.Tensor,
    sequences: Sequence[str],
    embedder: TextEmbedder,
) -> str:
    """
    Search a query string into an existing embedding batch and decode the best hit.
    This function can be used to search into pre-computed vectors.
    We can't recover text from embedding, so texts matching embedding must be
    included in the API.

    Args:
        query: Query text.
        embedding_vectors: Precomputed candidate embeddings with shape
            ``[items, dim]``.
        sequences: Candidate text sequences aligned with ``embedding_vectors``.
        embedder: Object providing ``embed`` for the query.

    Returns:
        The best matching candidate sequence.
    """

    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")
    if embedding_vectors.dim() != 2:
        raise ValueError("embedding_vectors must have shape [items, dim]")
    if embedding_vectors.shape[0] != len(sequences):
        raise ValueError("embedding_vectors must align with sequences")

    query_embedding = embedder.embed(query)
    scores = cosine_similarity(query_embedding, embedding_vectors)
    best_index = int(torch.argmax(scores))
    return sequences[best_index]
