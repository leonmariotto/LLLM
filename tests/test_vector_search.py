import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from ..LLLM.vector_search import (
    SearchResult,
    cosine_similarity,
    vector_build_and_search,
    vector_search,
)
from ..LLLM.vector_db import VectorDB


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def embed(self, text: str) -> torch.Tensor:
        return self.embeddings[text]

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.embeddings[text] for text in texts])


class DynamicFakeEmbedder:
    def embed(self, text: str) -> torch.Tensor:
        if text == "query":
            return torch.tensor([1.0, 0.0])
        return torch.tensor([0.0, 1.0])

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.stack(
            [
                torch.tensor([float(index + 1), 0.0])
                for index, _ in enumerate(texts)
            ]
        )


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


def test_vector_build_and_search_returns_top_k_chunk_results() -> None:
    embedder = FakeEmbedder(
        {
            "query": torch.tensor([1.0, 0.0]),
            "bad  ": torch.tensor([0.0, 1.0]),
            "exact": torch.tensor([1.0, 0.0]),
            "close": torch.tensor([1.0, 1.0]),
        }
    )
    text = "bad  exactclose"

    results = vector_build_and_search("query", text, embedder, top_k=2, chunk_size=5)

    assert results[0] == SearchResult(index=1, score=1.0, sequence="exact")
    assert results[1].index == 2
    assert results[1].sequence == "close"
    assert math.isclose(results[1].score, 2.0**-0.5, rel_tol=1e-6, abs_tol=1e-6)


def test_vector_build_and_search_handles_empty_text_and_zero_top_k() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})

    assert vector_build_and_search("query", "", embedder) == []
    assert vector_build_and_search("query", "ignored", embedder, top_k=0) == []


def test_vector_build_and_search_chunks_text_with_overlap() -> None:
    embedder = FakeEmbedder(
        {
            "query": torch.tensor([1.0, 0.0]),
            "abcd": torch.tensor([0.0, 1.0]),
            "cdef": torch.tensor([1.0, 0.0]),
            "ef": torch.tensor([1.0, 1.0]),
        }
    )

    results = vector_build_and_search(
        "query",
        "abcdef",
        embedder,
        chunk_size=4,
        chunk_overlap=2,
    )

    assert [result.sequence for result in results] == ["cdef", "ef", "abcd"]


def test_vector_search_returns_precomputed_top_k_results() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    sequences = ["unrelated", "exact", "close"]
    embedding_vectors = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    results = vector_search(
        "query",
        embedding_vectors,
        sequences,
        embedder,
        top_k=2,
    )

    assert results[0] == SearchResult(index=1, score=1.0, sequence="exact")
    assert results[1].index == 2
    assert results[1].sequence == "close"
    assert math.isclose(results[1].score, 2.0**-0.5, rel_tol=1e-6, abs_tol=1e-6)


def test_vector_search_handles_empty_sequences_and_zero_top_k() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})

    assert vector_search("query", torch.empty((0, 2)), [], embedder) == []
    assert vector_search("query", torch.ones((1, 2)), ["ignored"], embedder, top_k=0) == []


def test_vector_search_validates_precomputed_embeddings() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    with pytest.raises(ValueError, match="shape"):
        vector_search("query", torch.ones(2), ["candidate"], embedder)
    with pytest.raises(ValueError, match="align"):
        vector_search("query", torch.eye(2), ["candidate"], embedder)


def test_vector_search_validates_inputs() -> None:
    embedder = FakeEmbedder({"query": torch.ones(2), "candidate": torch.ones(2)})
    with pytest.raises(ValueError, match="top_k"):
        vector_search("query", torch.ones((1, 2)), ["candidate"], embedder, top_k=-1)
    with pytest.raises(ValueError, match="chunk_size"):
        vector_build_and_search("query", "candidate", embedder, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        vector_build_and_search("query", "candidate", embedder, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        vector_build_and_search(
            "query",
            "candidate",
            embedder,
            chunk_size=3,
            chunk_overlap=3,
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(torch.ones(3), torch.eye(2))


def test_vector_db_add_text_chunks_embeds_and_stores_metadata() -> None:
    db = VectorDB(DynamicFakeEmbedder())

    db.add_text("abcdefghijklmnopqrstuvwxyz012345678", chunk_size=25, metadata=["doc:1"])

    assert len(db.records) == 7
    assert db.records[0] == {
        "embedding": [1.0, 0.0],
        "text": "abcdefghijklmnopqrstuvwxy",
        "metadata": ["doc:1"],
    }
    assert db.records[1]["text"] == "fghijklmnopqrstuvwxyz0123"


def test_vector_db_search_returns_ranked_results_and_remaps_indices() -> None:
    db = VectorDB(
        FakeEmbedder({"query": torch.tensor([1.0, 0.0])}),
        records=[
            {"embedding": [0.0, 1.0], "text": "unrelated", "metadata": ["a"]},
            {"embedding": [1.0, 0.0], "text": "exact", "metadata": ["b"]},
            {"embedding": [1.0, 1.0], "text": "close", "metadata": ["a", "b"]},
        ],
    )

    results = db.search("query", top_k=2)

    assert results[0] == SearchResult(index=1, score=1.0, sequence="exact")
    assert results[1].index == 2
    assert results[1].sequence == "close"
    assert math.isclose(results[1].score, 2.0**-0.5, rel_tol=1e-6, abs_tol=1e-6)


def test_vector_db_search_filters_by_metadata_membership() -> None:
    db = VectorDB(
        FakeEmbedder({"query": torch.tensor([1.0, 0.0])}),
        records=[
            {"embedding": [1.0, 0.0], "text": "wrong metadata", "metadata": ["public"]},
            {"embedding": [0.8, 0.0], "text": "right metadata", "metadata": ["public", "manual"]},
        ],
    )

    results = db.search("query", ["manual"])

    assert results == [SearchResult(index=1, score=1.0, sequence="right metadata")]


def test_vector_db_search_handles_empty_matches_and_zero_top_k() -> None:
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    empty_db = VectorDB(embedder)
    db = VectorDB(
        embedder,
        records=[{"embedding": [1.0, 0.0], "text": "exact", "metadata": ["public"]}],
    )

    assert empty_db.search("query") == []
    assert db.search("query", ["missing"]) == []
    assert db.search("query", top_k=0) == []


def test_vector_db_exports_human_readable_json_and_loads(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    embedder = FakeEmbedder({"query": torch.tensor([1.0, 0.0])})
    db = VectorDB(
        embedder,
        records=[
            {"embedding": [1.0, 0.0], "text": "exact", "metadata": ["doc:1"]},
            {"embedding": [0.0, 1.0], "text": "unrelated", "metadata": ["doc:2"]},
        ],
    )

    db.export(path)
    loaded = VectorDB.load(path, embedder)

    assert path.read_text(encoding="utf-8").startswith("[\n  {")
    assert json.loads(path.read_text(encoding="utf-8")) == db.records
    assert loaded.records == db.records
    assert loaded.search("query", ["doc:1"], top_k=1) == [
        SearchResult(index=0, score=1.0, sequence="exact")
    ]


def test_vector_db_load_rejects_malformed_records(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(
        json.dumps([{"embedding": ["bad"], "text": "candidate", "metadata": []}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embedding"):
        VectorDB.load(path, DynamicFakeEmbedder())
