"""
Stable internal representation for embedding models.

EmbeddingModelIR is the embedding-model sibling of ModelIR.  Source
formats such as Hugging Face SentenceTransformer snapshots are translated into
this representation before model implementations consume them.

Keep the mess ordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch


EmbeddingArchitectureId = Literal["bert_sentence_transformer"]
EmbeddingIRWeight: TypeAlias = torch.Tensor
EmbeddingModelWeightsIR: TypeAlias = dict[str, EmbeddingIRWeight]


@dataclass(frozen=True)
class EmbeddingModelConfigIR:
    """Normalized embedding model config consumed by model constructors."""

    fields: dict[str, Any]

    def require_int(self, name: str) -> int:
        value = self.fields.get(name)
        if not isinstance(value, int):
            raise ValueError(f"embedding IR config field {name!r} must be an int")
        return value

    def require_float(self, name: str) -> float:
        value = self.fields.get(name)
        if isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise ValueError(f"embedding IR config field {name!r} must be a float")
        return value

    def require_bool(self, name: str) -> bool:
        value = self.fields.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"embedding IR config field {name!r} must be a bool")
        return value

    def require_str(self, name: str) -> str:
        value = self.fields.get(name)
        if not isinstance(value, str):
            raise ValueError(f"embedding IR config field {name!r} must be a str")
        return value

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True)
class EmbeddingModelIR:
    """Source-independent embedding model description and tensors."""

    architecture: EmbeddingArchitectureId
    config: EmbeddingModelConfigIR
    weights: EmbeddingModelWeightsIR
    tokenizer: dict[str, Any] = field(default_factory=lambda: {})
    pooling: dict[str, Any] = field(default_factory=lambda: {})
    metadata: dict[str, Any] = field(default_factory=lambda: {})


SOURCE_PREFIX_FRAGMENTS = (
    "encoder.layer",
    "embeddings.word_embeddings",
    "embeddings.position_embeddings",
    "embeddings.token_type_embeddings",
    "attention.self.",
    "attention.output.",
)


def assert_canonical_embedding_weight_names(ir: EmbeddingModelIR) -> None:
    """Raise if an embedding IR leaks source-format tensor names."""

    leaked = [
        name
        for name in ir.weights
        if any(fragment in name for fragment in SOURCE_PREFIX_FRAGMENTS)
    ]
    if leaked:
        examples = ", ".join(sorted(leaked)[:5])
        raise ValueError(
            f"embedding IR contains source-format tensor names: {examples}"
        )
