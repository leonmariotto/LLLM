"""
Custom SentenceTransformer embedding inference.
Provide SentenceTransformerEmbedder, a high-level API to produce vector
embedding from a batch of string.
Include hidden_state production, pools and normalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import math
from typing import Any, Protocol, TypedDict, cast

import torch
from torch import nn
import torch.nn.functional as F

from .embedding_model_ir import EmbeddingModelIR, EmbeddingModelWeightsIR
from .norm import LayerNorm


class SentenceTransformerConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    hidden_dim: int
    n_heads: int
    n_layers: int
    type_vocab_size: int
    layer_norm_eps: float
    pad_token_id: int
    normalize_embeddings: bool


class TextTokenizer(Protocol):
    def encode_batch(self, inputs: list[str]) -> list[Any]: ...

    def enable_truncation(self, max_length: int) -> None: ...

    def enable_padding(self, *, pad_id: int, pad_type_id: int = 0) -> None: ...


class BertEmbeddings(nn.Module):
    """BERT token, position, and token-type embeddings."""

    def __init__(self, cfg: SentenceTransformerConfig) -> None:
        super().__init__()
        self.context_length = cfg["context_length"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.type_emb = nn.Embedding(cfg["type_vocab_size"], cfg["emb_dim"])
        self.norm = LayerNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"input sequence length {seq_len} exceeds context length "
                f"{self.context_length}"
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = (
            self.tok_emb(input_ids)
            + self.pos_emb(position_ids)
            + self.type_emb(token_type_ids)
        )
        return self.norm(x)


class BertSelfAttention(nn.Module):
    """Bidirectional BERT self-attention with a key padding mask."""

    def __init__(self, cfg: SentenceTransformerConfig) -> None:
        super().__init__()
        emb_dim = cfg["emb_dim"]
        n_heads = cfg["n_heads"]
        if emb_dim % n_heads != 0:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.num_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.W_query = nn.Linear(emb_dim, emb_dim)
        self.W_key = nn.Linear(emb_dim, emb_dim)
        self.W_value = nn.Linear(emb_dim, emb_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, num_tokens, emb_dim = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = queries @ keys.transpose(2, 3)
        scores = scores / math.sqrt(self.head_dim)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ values
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, num_tokens, emb_dim)
        )
        return context


class BertTransformerBlock(nn.Module):
    """BERT encoder block used by all-MiniLM-L6-v2."""

    def __init__(self, cfg: SentenceTransformerConfig) -> None:
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.att = BertSelfAttention(cfg)
        self.att_output = nn.Linear(emb_dim, emb_dim)
        self.att_norm = LayerNorm(emb_dim, eps=cfg["layer_norm_eps"])
        self.ff_up = nn.Linear(emb_dim, cfg["hidden_dim"])
        self.ff_down = nn.Linear(cfg["hidden_dim"], emb_dim)
        self.ff_norm = LayerNorm(emb_dim, eps=cfg["layer_norm_eps"])

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        att = self.att(x, attention_mask)
        x = self.att_norm(x + self.att_output(att))
        ff = self.ff_down(F.gelu(self.ff_up(x)))
        return self.ff_norm(x + ff)


class SentenceTransformerModel(nn.Module):
    """BERT-style encoder consumed by SentenceTransformer pooling."""

    def __init__(self, cfg: SentenceTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddings(cfg)
        self.blocks = nn.ModuleList(
            [BertTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

    @staticmethod
    def config_from_ir(ir: EmbeddingModelIR) -> SentenceTransformerConfig:
        if ir.architecture != "bert_sentence_transformer":
            raise ValueError(
                f"expected bert_sentence_transformer IR, got {ir.architecture!r}"
            )
        hidden_act = ir.config.require_str("hidden_act")
        if hidden_act != "gelu":
            raise ValueError(f"unsupported BERT hidden_act {hidden_act!r}")
        pooling_mode = ir.config.require_str("pooling_mode")
        if pooling_mode != "mean_tokens":
            raise ValueError(f"unsupported pooling mode {pooling_mode!r}")
        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "hidden_dim": ir.config.require_int("intermediate_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "type_vocab_size": ir.config.require_int("type_vocab_size"),
            "layer_norm_eps": ir.config.require_float("layer_norm_eps"),
            "pad_token_id": ir.config.require_int("pad_token_id"),
            "normalize_embeddings": ir.config.require_bool("normalize_embeddings"),
        }

    def load_ir_weights(self, ir: EmbeddingModelIR) -> None:
        if ir.architecture != "bert_sentence_transformer":
            raise ValueError(
                f"expected bert_sentence_transformer IR, got {ir.architecture!r}"
            )
        weights = ir.weights
        with torch.no_grad():
            self._copy_param(
                self.embeddings.tok_emb.weight,
                self._weight(weights, "embeddings.token.weight"),
            )
            self._copy_param(
                self.embeddings.pos_emb.weight,
                self._weight(weights, "embeddings.position.weight"),
            )
            self._copy_param(
                self.embeddings.type_emb.weight,
                self._weight(weights, "embeddings.token_type.weight"),
            )
            self._copy_param(
                self.embeddings.norm.scale,
                self._weight(weights, "embeddings.norm.weight"),
            )
            self._copy_param(
                self.embeddings.norm.shift,
                self._weight(weights, "embeddings.norm.bias"),
            )
            for layer_idx, module in enumerate(self.blocks):
                block = cast(BertTransformerBlock, module)
                prefix = f"layers.{layer_idx}"
                self._copy_param(
                    block.att.W_query.weight,
                    self._weight(weights, f"{prefix}.attention.q_proj.weight"),
                )
                self._copy_param(
                    block.att.W_query.bias,
                    self._weight(weights, f"{prefix}.attention.q_proj.bias"),
                )
                self._copy_param(
                    block.att.W_key.weight,
                    self._weight(weights, f"{prefix}.attention.k_proj.weight"),
                )
                self._copy_param(
                    block.att.W_key.bias,
                    self._weight(weights, f"{prefix}.attention.k_proj.bias"),
                )
                self._copy_param(
                    block.att.W_value.weight,
                    self._weight(weights, f"{prefix}.attention.v_proj.weight"),
                )
                self._copy_param(
                    block.att.W_value.bias,
                    self._weight(weights, f"{prefix}.attention.v_proj.bias"),
                )
                self._copy_param(
                    block.att_output.weight,
                    self._weight(weights, f"{prefix}.attention.o_proj.weight"),
                )
                self._copy_param(
                    block.att_output.bias,
                    self._weight(weights, f"{prefix}.attention.o_proj.bias"),
                )
                self._copy_param(
                    block.att_norm.scale,
                    self._weight(weights, f"{prefix}.attention.output_norm.weight"),
                )
                self._copy_param(
                    block.att_norm.shift,
                    self._weight(weights, f"{prefix}.attention.output_norm.bias"),
                )
                self._copy_param(
                    block.ff_up.weight,
                    self._weight(weights, f"{prefix}.feed_forward.up_proj.weight"),
                )
                self._copy_param(
                    block.ff_up.bias,
                    self._weight(weights, f"{prefix}.feed_forward.up_proj.bias"),
                )
                self._copy_param(
                    block.ff_down.weight,
                    self._weight(weights, f"{prefix}.feed_forward.down_proj.weight"),
                )
                self._copy_param(
                    block.ff_down.bias,
                    self._weight(weights, f"{prefix}.feed_forward.down_proj.bias"),
                )
                self._copy_param(
                    block.ff_norm.scale,
                    self._weight(weights, f"{prefix}.feed_forward.output_norm.weight"),
                )
                self._copy_param(
                    block.ff_norm.shift,
                    self._weight(weights, f"{prefix}.feed_forward.output_norm.bias"),
                )
        self.eval()

    @staticmethod
    def _copy_param(param: nn.Parameter | torch.Tensor, value: torch.Tensor) -> None:
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for parameter: expected {tuple(param.shape)}, "
                f"got {tuple(value.shape)}"
            )
        param.copy_(value)

    @staticmethod
    def _weight(weights: EmbeddingModelWeightsIR, name: str) -> torch.Tensor:
        if name not in weights:
            raise KeyError(f"missing embedding IR weight {name!r}")
        return weights[name]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)
        for module in self.blocks:
            block = cast(BertTransformerBlock, module)
            x = block(x, attention_mask)
        return x


class SentenceTransformerEmbedder:
    """High-level embedding API around ``SentenceTransformerModel``."""

    def __init__(
        self,
        model: SentenceTransformerModel,
        tokenizer: TextTokenizer,
        *,
        context_length: int,
        pad_token_id: int,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.normalize_embeddings = normalize_embeddings
        self.tokenizer.enable_truncation(max_length=context_length)
        self.tokenizer.enable_padding(pad_id=pad_token_id, pad_type_id=0)

    @classmethod
    def from_ir(cls, ir: EmbeddingModelIR) -> "SentenceTransformerEmbedder":
        tokenizer_json = ir.tokenizer.get("tokenizer_json")
        if not isinstance(tokenizer_json, str):
            raise ValueError("embedding IR is missing tokenizer_json metadata")
        tokenizers = import_module("tokenizers")
        tokenizer_cls = getattr(tokenizers, "Tokenizer")
        tokenizer = cast(TextTokenizer, tokenizer_cls.from_str(tokenizer_json))
        model = SentenceTransformerModel(SentenceTransformerModel.config_from_ir(ir))
        model.load_ir_weights(ir)
        return cls(
            model,
            tokenizer,
            context_length=ir.config.require_int("context_length"),
            pad_token_id=ir.config.require_int("pad_token_id"),
            normalize_embeddings=ir.config.require_bool("normalize_embeddings"),
        )

    def embed(self, text: str) -> torch.Tensor:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(
                (0, self.model.cfg["emb_dim"]),
                dtype=next(self.model.parameters()).dtype,
                device=next(self.model.parameters()).device,
            )
        input_ids, attention_mask, token_type_ids = self._encode_batch(texts)
        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        token_type_ids = token_type_ids.to(model_device)
        with torch.no_grad():
            hidden_states = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            embeddings = mean_pool(hidden_states, attention_mask)
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _encode_batch(
        self, texts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer.encode_batch(list(texts))
        input_ids = [list(item.ids) for item in encoded]
        attention_mask = [list(item.attention_mask) for item in encoded]
        token_type_ids = [list(item.type_ids) for item in encoded]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
        )

    # TODO LMA: this should be moved in a common embedding module
    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        return a_norm @ b_norm.T

    # TODO LMA: this should be moved in a common embedding module


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean-pool token hidden states over non-padding tokens.

    Args:
        hidden_states: Tensor with shape ``[batch, tokens, hidden]``.
        attention_mask: Tensor with shape ``[batch, tokens]`` where non-zero
            entries indicate tokens included in the pool.

    Returns:
        Pooled embeddings with shape ``[batch, hidden]``.
    """

    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(torch.finfo(hidden_states.dtype).eps)
    return summed / counts
