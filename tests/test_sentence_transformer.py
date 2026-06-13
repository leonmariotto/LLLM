from typing import Any, Callable, cast

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertConfig as TransformersBertConfig
from transformers import BertModel as TransformersBertModel

from ..LLLM.embedding_model_ir import assert_canonical_embedding_weight_names
from ..LLLM.fetch import fetch_embedding_model_ir
from ..LLLM.hf_embedding_loader import embedding_model_ir_from_hf
from ..LLLM.sentence_transformer import (
    SentenceTransformerEmbedder,
    SentenceTransformerModel,
    mean_pool,
)


def _tiny_hf_bert_config() -> dict[str, object]:
    return {
        "model_type": "bert",
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "hidden_size": 12,
        "intermediate_size": 24,
        "num_attention_heads": 3,
        "num_hidden_layers": 2,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
    }


def _sentence_transformer_modules() -> list[dict[str, object]]:
    return [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize",
        },
    ]


def _mean_pooling_config() -> dict[str, object]:
    return {
        "word_embedding_dimension": 12,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }


def test_embedding_hf_loader_produces_canonical_bert_names() -> None:
    weights = {
        "embeddings.word_embeddings.weight": torch.randn(32, 12),
        "embeddings.position_embeddings.weight": torch.randn(16, 12),
        "embeddings.token_type_embeddings.weight": torch.randn(2, 12),
        "embeddings.LayerNorm.weight": torch.randn(12),
        "encoder.layer.0.attention.self.query.weight": torch.randn(12, 12),
        "encoder.layer.0.attention.output.LayerNorm.bias": torch.randn(12),
        "encoder.layer.0.intermediate.dense.weight": torch.randn(24, 12),
    }

    ir = embedding_model_ir_from_hf(
        _tiny_hf_bert_config(),
        weights,
        modules=_sentence_transformer_modules(),
        pooling=_mean_pooling_config(),
    )

    assert ir.architecture == "bert_sentence_transformer"
    assert ir.config.require_str("pooling_mode") == "mean_tokens"
    assert ir.config.require_bool("normalize_embeddings") is True
    assert "embeddings.token.weight" in ir.weights
    assert "layers.0.attention.q_proj.weight" in ir.weights
    assert "layers.0.attention.output_norm.bias" in ir.weights
    assert "layers.0.feed_forward.up_proj.weight" in ir.weights
    assert_canonical_embedding_weight_names(ir)


def test_embedding_loader_rejects_unsupported_pooling() -> None:
    pooling = _mean_pooling_config()
    pooling["pooling_mode_max_tokens"] = True

    with pytest.raises(ValueError, match="unsupported SentenceTransformer pooling"):
        embedding_model_ir_from_hf(
            _tiny_hf_bert_config(),
            {},
            modules=_sentence_transformer_modules(),
            pooling=pooling,
        )


def test_mean_pool_ignores_padding_tokens() -> None:
    hidden_states = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 6.0], [100.0, 100.0]],
            [[4.0, 8.0], [10.0, 20.0], [30.0, 40.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = mean_pool(hidden_states, attention_mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0], [4.0, 8.0]]))


def test_similarity_is_cosine_similarity() -> None:
    a = torch.tensor([[3.0, 4.0]])
    b = torch.tensor([[6.0, 8.0], [4.0, -3.0]])

    similarity = SentenceTransformerEmbedder.similarity(a, b)

    torch.testing.assert_close(similarity, torch.tensor([[1.0, 0.0]]))


def test_tiny_bert_encoder_matches_transformers_reference_model() -> None:
    manual_seed = cast(Callable[[int], Any], cast(Any, torch).manual_seed)
    reference_config = cast(Any, TransformersBertConfig)(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=12,
        intermediate_size=24,
        num_attention_heads=3,
        num_hidden_layers=2,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
        pad_token_id=0,
        position_embedding_type="absolute",
    )
    manual_seed(123)
    reference = cast(Any, TransformersBertModel)(
        reference_config, add_pooling_layer=False
    ).eval()
    ir = embedding_model_ir_from_hf(
        reference_config.to_dict(),
        reference.state_dict(),
        modules=_sentence_transformer_modules(),
        pooling=_mean_pooling_config(),
    )
    model = SentenceTransformerModel(SentenceTransformerModel.config_from_ir(ir)).eval()
    model.load_ir_weights(ir)

    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    token_type_ids = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]])

    with torch.no_grad():
        actual = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        expected = reference(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.slow
def test_all_minilm_l6_v2_embeddings_match_transformers_reference() -> None:
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    ir = fetch_embedding_model_ir(repo_id)
    embedder = SentenceTransformerEmbedder.from_ir(ir)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    reference = AutoModel.from_pretrained(repo_id).eval()
    texts = ["A small embedding test.", "Vector search uses cosine similarity."]

    with torch.no_grad():
        actual = embedder.embed_batch(texts)
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        reference_hidden = reference(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(reference_hidden.dtype)
        expected = (reference_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        expected = F.normalize(expected, p=2, dim=-1)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
