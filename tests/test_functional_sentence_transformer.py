import pytest

from ..LLLM.fetch import fetch_embedding_model_ir
from ..LLLM.sentence_transformer import SentenceTransformerEmbedder
from ..LLLM.vector_search import vector_search_into


pytestmark = pytest.mark.slow

SENTENCE_TRANSFORMER_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"


def test_functional_sentence_transformer_embedding_vector_search() -> None:
    ir = fetch_embedding_model_ir(SENTENCE_TRANSFORMER_REPO_ID)
    embedder = SentenceTransformerEmbedder.from_ir(ir)
    query = "How can I find documents with similar meaning?"
    candidates = [
        "Bake the cake until the top is golden brown.",
        "Semantic vector search ranks text by comparing embedding similarity.",
        "The weather forecast predicts heavy rain tomorrow.",
        "A guitar string vibrates when it is plucked.",
    ]

    candidate_embeddings = embedder.embed_batch(candidates)
    best_sequence = vector_search_into(
        query,
        candidate_embeddings,
        candidates,
        embedder,
    )

    print(f"Best vector search sequence: {best_sequence}")

    assert best_sequence == candidates[1]
