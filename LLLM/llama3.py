"""
LLama3
Support 3.1 and 3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, cast, Any
import os
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from .llama2 import Llama2FeedForward
from .norm import RMSNorm
from .rope import precompute_rope_cache, apply_rope, RopeFrequencyConfig
from .kv_cache import KVCache

if TYPE_CHECKING:
    from .fetch import FetchedModel


class Llama3Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_kv_groups: int  # GQA
    n_layers: int
    hidden_dim: int
    rope_theta: float
    freq_config: RopeFrequencyConfig | None
    dtype: torch.dtype


def llama3_config_from_fetched(config: dict[str, Any]) -> Llama3Config:
    """Translate a Hugging Face Llama config into LLLM Llama2Config."""

    def _float_config(
        config: dict[str, Any],
        key: str,
        *,
        fallback_key: str | None = None,
        default: float | None = None,
    ) -> float:
        value = config.get(key)
        if value is None and fallback_key is not None:
            value = config.get(fallback_key)
        if value is None and default is not None:
            return default
        if not isinstance(value, float):
            raise ValueError(f"config value {key!r} must be an float")
        return value

    def _int_config(
        config: dict[str, Any], key: str, *, fallback_key: str | None = None
    ) -> int:
        value = config.get(key)
        if value is None and fallback_key is not None:
            value = config.get(fallback_key)
        if not isinstance(value, int):
            raise ValueError(f"config value {key!r} must be an int")
        return value

    def _hidden_dim_config(config: dict[str, Any]) -> int:
        intermediate_size = config.get("intermediate_size")
        if isinstance(intermediate_size, int):
            return intermediate_size

        dim = _int_config(config, "dim")
        multiple_of = _int_config(config, "multiple_of")
        hidden_dim = int(2 * (4 * dim) / 3)
        return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    return {
        "vocab_size": _int_config(config, "vocab_size"),
        "context_length": _int_config(
            config, "max_position_embeddings", fallback_key="max_seq_len"
        ),
        "emb_dim": _int_config(config, "hidden_size", fallback_key="dim"),
        "n_heads": _int_config(config, "num_attention_heads", fallback_key="n_heads"),
        "n_kv_groups": _int_config(
            config, "num_key_value_heads", fallback_key="n_heads"
        ),
        "n_layers": _int_config(config, "num_hidden_layers", fallback_key="n_layers"),
        "hidden_dim": _hidden_dim_config(config),
        "rope_theta": _float_config(config, "rope_theta", default=10000.0),
        "freq_config": None,
        "dtype": torch.float32,
    }


class Llama3GroupedQueryAttention(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Attention mechanism involve 3 trainable matrix : query, key, values.

    Implement Causal mask and dropout. Mask is computed on the fly in the forward pass.
    Default dropout is 0 which result in identity matrix.

    Multi-headed attention: d_out is splited in num_head parts. Each head
    produce a part of d_out (head_dim, calculated at init), and at the end
    context_vec is reshaped to the correct size.
    So things can be parallel.

    Implement RoPe. Enabled by default.

    Qrouped Query Attention: reduce the number of query group that attend to the
    KV pair. This reduce the size of parameters, without reducing the model
    performance (as much). Each query group needs to be repeated to match the number
    of heads. Note that if num_kv_groups == num_heads we're back to MHA, so this code
    is compatible with MHA too.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        dropout: float = 0.0,  # No dropout by default.
        qkv_bias: bool = False,  # No bias.
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        - d_in: embedding size (size of embedded vector, 1 embedded vector per token)
        - d_out context vector size.
        - context_lenght: correspond to the number token used to compute a context
        vector. In the case of a DataSet/DataLoader setup, it will correspond
        to the window_size.
        - droput: for training purpose, it is possible to hide randomly some attention
        weight before computing the context vector. dropout value is the probability
        for a weight to be zeroed.
        - num_head: number of head.
        - num_kv_groups: for GQA, this is the number of KV matrice, that will be
        distributed to heads.
        """
        super().__init__()

        assert num_heads != 0, "num_head shall not be 0"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups."
        )

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.context_length = context_length
        self.num_kv_groups = num_kv_groups
        self.kv_group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype
        )
        self.out_proj = nn.Linear(
            d_out, d_out, bias=qkv_bias
        )  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        forward method is called by the nn.Module __call__ method.
        x is expected to be a batch of tensor of d_in size.
        (number of batch, number of token, embedding size)
        return a tensor of d_out size.
        Note that mask is always computed on-the-fly here.

        cos and sin are pre-computed rope sequence.
        pos: contains the (context-wide relative) starting index of the sequence.
        """
        b, num_tokens, d_in = x.shape

        assert self.d_in == d_in, "invalid d_in (embedding size)"

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # About tensor.view and tensor.transpose methods:
        #   Tensor view method reshape a tensor, without moving elements in memory.
        #   Whereas transpose change how dimensions are indexed.
        #   We implicitly split the matrix by adding a `num_heads` dimension
        #   Unroll last dim:
        #       (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose:
        #   (b, num_tokens, num_heads, head_dim) ->
        #       (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        #   (b, num_tokens, num_kv_groups, head_dim) ->
        #       (b, num_kv_groups, num_tokens, head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)

        # Get the cache length for the current layer.
        past_tokens = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            past_tokens = kv_cache.layer_seq_len(layer_idx)

        # Add RoPE after Q/K projection and head reshaping and before computing
        # attention scores.
        # We pass to apply_rope only sin/cos for the current position. Compute each index
        # position from a pos offset.
        # If pos is None we infer from cache length.
        if pos is None:
            start_pos = past_tokens
        else:
            start_pos = pos
        position_ids = torch.arange(
            start_pos,
            start_pos + num_tokens,
            device=x.device,
        )
        # TODO rope can support positional information exceeding context lenght, remove the
        # following when in place.
        assert int(position_ids[-1]) < self.context_length, (
            "RoPE position exceeds precomputed context length"
        )
        current_cos = cos[position_ids]
        current_sin = sin[position_ids]
        queries = apply_rope(queries, current_cos, current_sin)
        keys_new = apply_rope(keys_new, current_cos, current_sin)

        if kv_cache is None:
            keys, values = keys_new, values_new
        else:
            assert layer_idx is not None
            keys, values = kv_cache.update(layer_idx, keys_new, values_new)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        keys = keys.repeat_interleave(self.kv_group_size, dim=1)
        values = values.repeat_interleave(self.kv_group_size, dim=1)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # `queries` has shape (batch, num_heads, num_tokens, head_dim).
        # So shape[-2] is the query-token dimension.
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        key_positions = torch.arange(num_tokens_K, device=x.device)
        query_positions = torch.arange(
            num_tokens_K - num_tokens_Q,
            num_tokens_K,
            device=x.device,
        )
        mask_bool = key_positions[None, :] > query_positions[:, None]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class Llama3TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.att = Llama3GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.ff = Llama2FeedForward(cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"])

        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        if kv_cache is None:
            x = self.att(x, cos, sin, pos)  # Shape [batch_size, num_tokens, emb_size]
        else:
            x = self.att(x, cos, sin, pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class Llama3Model(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.Sequential(
            *[Llama3TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        cos, sin = precompute_rope_cache(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            base=cfg["rope_theta"],
            seq_len=cfg["context_length"],
            freq_config=cfg["freq_config"],
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        in_idx: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
    ):
        # batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        for layer_idx, module in enumerate(self.trf_blocks):
            block = cast(Llama3TransformerBlock, module)
            x = block(
                x,
                self.rope_cos,
                self.rope_sin,
                pos=pos,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def load_fetched_model(self, fetched: FetchedModel) -> None:
        """Copy Llama tensors from a fetched safetensors checkpoint."""
        with torch.no_grad():
            embedding_weight = self._optional_weight(
                fetched.weights, "tok_embeddings.weight"
            )
            if embedding_weight is None:
                embedding_weight = self._weight(fetched.weights, "output.weight")
            self._copy_param(self.tok_emb.weight, embedding_weight)

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Llama3TransformerBlock, module)
                prefix = f"layers.{layer_idx}"

                self._copy_param(
                    block.att.W_query.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wq.weight"),
                )
                self._copy_param(
                    block.att.W_key.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wk.weight"),
                )
                self._copy_param(
                    block.att.W_value.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wv.weight"),
                )
                self._copy_param(
                    block.att.out_proj.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wo.weight"),
                )

                self._copy_param(
                    block.norm1.weight,
                    self._weight(fetched.weights, f"{prefix}.attention_norm.weight"),
                )
                self._copy_param(
                    block.norm2.weight,
                    self._weight(fetched.weights, f"{prefix}.ffn_norm.weight"),
                )

                self._copy_param(
                    block.ff.fc1.weight,
                    self._weight(fetched.weights, f"{prefix}.feed_forward.w1.weight"),
                )
                self._copy_param(
                    block.ff.fc2.weight,
                    self._weight(fetched.weights, f"{prefix}.feed_forward.w3.weight"),
                )
                self._copy_param(
                    block.ff.fc3.weight,
                    self._weight(fetched.weights, f"{prefix}.feed_forward.w2.weight"),
                )

            self._copy_param(
                self.final_norm.weight, self._weight(fetched.weights, "norm.weight")
            )
            self._copy_param(
                self.out_head.weight, self._weight(fetched.weights, "output.weight")
            )

        self.eval()

    @staticmethod
    def _copy_param(
        param: nn.Parameter | torch.Tensor | None, value: torch.Tensor
    ) -> None:
        if param is None:
            raise ValueError("cannot copy into missing parameter")
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for parameter: expected {tuple(param.shape)}, "
                f"got {tuple(value.shape)}"
            )
        param.copy_(value)

    @staticmethod
    def _optional_weight(
        weights: dict[str, torch.Tensor], name: str
    ) -> torch.Tensor | None:
        if name in weights:
            return weights[name]
        return None

    @staticmethod
    def _weight(weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name in weights:
            return weights[name]
        raise KeyError(f"missing Llama weight {name!r}")


class Llama3Tokenizer:
    """
    Thin wrapper around tiktoken that keeps track of Llama-3 special IDs.
    Need to be init with a tokenizer file.
    """

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if 128002 + i not in self.special.values()
            }
        )

        self.tiktok = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
            r"|\p{N}{1,3}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+"
            r"|\s+(?!\S)"
            r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) + self.tiktok.encode(
            text
        )
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, tok: list[int]) -> str:
        return self.tiktok.decode(tok)
