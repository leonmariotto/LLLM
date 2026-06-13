import math
from collections.abc import Sequence

import pytest
import torch

from ..LLLM.vector_search import (
    SearchResult,
    cosine_similarity,
    vector_search,
    vector_search_into,
)


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def embed(self, text: str) -> torch.Tensor:
        return self.embeddings[text]

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.embeddings[text] for text in texts])


def test_cosine_similarity_scores_single_query_against_vectors() -> None:
    query = torch.tensor([3.0, 4.0])
    vectors = torch.tensor([[6.0, 8.0], [4.0, -3.0], [0.0, 5.0]])

    scores = cosine_similarity(query, vectors)

    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 0.8]))


def test_cosine_similarity_supports_batched_queries() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    vectors = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    scores = cosine_similarity(queries, vectors)

    expected = torch.tensor(
        [
            [1.0, 0.0, 2.0**-0.5],
            [0.0, 1.0, 2.0**-0.5],
        ]
    )
    torch.testing.assert_close(scores, expected)


def test_vector_search_returns_top_k_results_by_descending_score() -> None:
    embedder = FakeEmbedder(
        {
            "query": torch.tensor([1.0, 0.0]),
            "unrelated": torch.tensor([0.0, 1.0]),
            "exact": torch.tensor([1.0, 0.0]),
            "close": torch.tensor([1.0, 1.0]),
        }
    )
    sequences = ["unrelated", "exact", "close"]

    results = vector_search("query", sequences, embedder, top_k=2)

    assert results[0] == SearchResult(index=1, score=1.0, sequence="exact")
    assert results[1].index == 2
    assert results[1].sequence == "close"
    assert math.isclose(results[1].score, 2.0**-0.5, rel_tol=1e-6, abs_tol=1e-6)


def test_vector_search_handles_empty_sequences_and_zero_top_k() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})

    assert vector_search("query", [], embedder) == []
    assert vector_search("query", ["ignored"], embedder, top_k=0) == []


def test_vector_search_into_returns_best_decoded_sequence() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    sequences = ["unrelated", "exact", "close"]
    embedding_vectors = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    best_sequence = vector_search_into(
        "query",
        embedding_vectors,
        sequences,
        embedder,
    )

    assert best_sequence == "exact"


def test_vector_search_into_validates_precomputed_embeddings() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    with pytest.raises(ValueError, match="must not be empty"):
        vector_search_into("query", torch.empty((0, 2)), [], embedder)
    with pytest.raises(ValueError, match="shape"):
        vector_search_into("query", torch.ones(2), ["candidate"], embedder)
    with pytest.raises(ValueError, match="align"):
        vector_search_into("query", torch.eye(2), ["candidate"], embedder)


def test_vector_search_validates_inputs() -> None:
    embedder = FakeEmbedder({"query": torch.ones(2), "candidate": torch.ones(2)})
    with pytest.raises(ValueError, match="top_k"):
        vector_search("query", ["candidate"], embedder, top_k=-1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(torch.ones(3), torch.eye(2))
