"""
Gemma3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, cast, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .rope import precompute_rope_cache, apply_rope
from .kv_cache import KVCache

if TYPE_CHECKING:
    from .fetch import FetchedModel


class Gemma3Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_kv_groups: int  # GQA
    sliding_window: int  # SWA
    n_layers: int
    hidden_dim: int
    head_dim: int
    rope_base: float
    rope_local_base: float
    rope_interleaved: bool
    layer_types: list[str]
    dtype: torch.dtype


def gemma3_config_from_fetched(config: dict[str, Any]) -> Gemma3Config:
    """Translate a Hugging Face Llama config into LLLM Gemma3Config."""

    def _float_config(
        config: dict[str, Any],
        key: str,
        *,
        fallback_key: str | None = None,
        default: float | None = None,
        allow_int: bool = False,
    ) -> float:
        value = config.get(key)
        if value is None and fallback_key is not None:
            value = config.get(fallback_key)
        if value is None and default is not None:
            return default
        if allow_int and isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise ValueError(f"config value {key!r} must be an float")
        return value

    def _hidden_dim_config(config: dict[str, Any]) -> int:
        intermediate_size = config.get("intermediate_size")
        if isinstance(intermediate_size, int):
            return intermediate_size

        dim = _int_config(config, "dim")
        multiple_of = _int_config(config, "multiple_of")
        hidden_dim = int(2 * (4 * dim) / 3)
        return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    def _int_config(
        config: dict[str, Any], key: str, *, fallback_key: str | None = None
    ) -> int:
        value = config.get(key)
        if value is None and fallback_key is not None:
            value = config.get(fallback_key)
        if not isinstance(value, int):
            raise ValueError(f"config value {key!r} must be an int")
        return value

    def _layer_types_config(config: dict[str, Any], key: str) -> list[str]:
        value = config.get(key)
        if not isinstance(value, list):
            raise ValueError(f"config value {key!r} must be an int")
        return cast(list[str], value)

    def _bool_config(config: dict[str, Any], key: str, *, default: bool) -> bool:
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"config value {key!r} must be a bool")
        return value

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
        "sliding_window": _int_config(config, "sliding_window"),
        "n_layers": _int_config(config, "num_hidden_layers", fallback_key="n_layers"),
        "hidden_dim": _hidden_dim_config(config),
        "head_dim": _int_config(config, "head_dim"),
        "rope_base": _float_config(
            config, "rope_theta", default=1000000.0, allow_int=True
        ),
        "rope_local_base": _float_config(
            config, "rope_local_base_freq", default=10000.0, allow_int=True
        ),
        "rope_interleaved": _bool_config(config, "rope_interleaved", default=False),
        "layer_types": _layer_types_config(config, "layer_types"),
        "dtype": torch.float32,
    }


class Gemma3FeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dtype: Optional[torch.dtype]):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class Gemma3RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    RMSNorm uses only the root mean square, which improves computational efficiency
    (over LayerNorm which use mean and variance).
    See https://arxiv.org/abs/1910.07467.

    Custom version for Gemma3:
    - stores zero-centered weights and uses (1 + weight) during forward
    - compute norm in float32, then scale by (1 + w)
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)


class Gemma3GroupedQueryAttention(nn.Module):
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

    Sliding window: instead of using all previous token in attention computation, we
    use a sliding window to use only the X last token.
    query_pre_attn_scalar is the scaling factor used on the attention scores. It's part
    of Gemma3 models configuration.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        sliding_window: int,
        dropout: float = 0.0,  # No dropout by default.
        qkv_bias: bool = False,  # No bias.
        rope_interleaved: bool = False,
        query_pre_attn_scalar: Optional[int] = None,
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
        - sliding window: number of tokens to include in attention (SWA). TODO reword
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
        self.sliding_window = sliding_window
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.context_length = context_length
        self.num_kv_groups = num_kv_groups
        self.kv_group_size = num_heads // num_kv_groups
        self.rope_interleaved = rope_interleaved

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

        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=1e-6)

        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (self.head_dim) ** -0.5

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

        # Apply RMS normalization (Gemma3 specific).
        queries = self.q_norm(queries)
        keys_new = self.k_norm(keys_new)

        # Get the cache length for the current layer.
        past_tokens = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            past_tokens = kv_cache.layer_seq_len(layer_idx)

        # Add RoPE after Q/K projection and head reshaping and before computing
        # attention scores.
        # We pass to apply_rope only sin/cos for the current position. Compute
        # each index position from a pos offset.
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
        assert int(position_ids[-1]) < self.context_length, (
            "RoPE position exceeds precomputed context length"
        )
        current_cos = cos[position_ids]
        current_sin = sin[position_ids]
        queries = apply_rope(
            queries,
            current_cos,
            current_sin,
            use_interleaved=self.rope_interleaved,
        )
        keys_new = apply_rope(
            keys_new,
            current_cos,
            current_sin,
            use_interleaved=self.rope_interleaved,
        )

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
            device=x.device,  # TODO why ?
        )
        # The boolean comparison is applied element-wise after broadcasting dimension.
        # [1, K] broadcasts to [Q, K]
        # [Q, 1] broadcasts to [Q, K]
        # So < returns a boolean tensor of shape: [Q, K]
        causal_mask = key_positions[None, :] > query_positions[:, None]
        # TODO, it make no sens to do a sliding window mechanism without a sliding KVCache.
        # Need to implement a sliding KVCache here.
        sliding_window_mask = key_positions[None, :] < (
            query_positions[:, None] - self.sliding_window + 1
        )
        mask_bool = causal_mask | sliding_window_mask

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # Use pre-computed scaling factor here (Gemma3).
        # TODO ensure it is really equivalent to do :
        # queries = queries * self.scaling before.
        attn_weights = torch.softmax(attn_scores * self.scaling, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class Gemma3TransformerBlock(nn.Module):
    def __init__(self, cfg: Gemma3Config, attn_type: str):
        super().__init__()

        self.attn_type = attn_type

        self.att = Gemma3GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            sliding_window=cfg["sliding_window"],
            rope_interleaved=cfg["rope_interleaved"],
            dtype=cfg["dtype"],
        )
        self.ff = Gemma3FeedForward(cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"])

        self.input_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.post_attention_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            cos = cos_local
            sin = sin_local
        else:
            cos = cos_global
            sin = sin_global

        if kv_cache is None:
            x = self.att(x, cos, sin, pos)  # Shape [batch_size, num_tokens, emb_size]
        else:
            x = self.att(x, cos, sin, pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn

        return x


class Gemma3Model(nn.Module):
    rope_cos_global: torch.Tensor
    rope_sin_global: torch.Tensor
    rope_cos_local: torch.Tensor
    rope_sin_local: torch.Tensor

    def __init__(self, cfg: Gemma3Config):
        super().__init__()

        assert (
            cfg["layer_types"] is not None
            and len(cfg["layer_types"]) == cfg["n_layers"]
        )

        self.context_length = cfg["context_length"]
        self.embedded_dim = cfg["emb_dim"]
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.dtype = cfg["dtype"]

        self.trf_blocks = nn.ModuleList(
            Gemma3TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]
        )

        self.final_norm = Gemma3RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        cos_global, sin_global = precompute_rope_cache(
            head_dim=cfg["head_dim"],
            base=cfg["rope_base"],
            seq_len=cfg["context_length"],
        )
        self.register_buffer("rope_cos_global", cos_global, persistent=False)
        self.register_buffer("rope_sin_global", sin_global, persistent=False)
        cos_local, sin_local = precompute_rope_cache(
            head_dim=cfg["head_dim"],
            base=cfg["rope_local_base"],
            seq_len=cfg["context_length"],
        )
        self.register_buffer("rope_cos_local", cos_local, persistent=False)
        self.register_buffer("rope_sin_local", sin_local, persistent=False)

    def forward(
        self,
        in_idx: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
    ):
        # batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) * (self.embedded_dim**0.5)  # TODO WTF ?
        for layer_idx, module in enumerate(self.trf_blocks):
            block = cast(Gemma3TransformerBlock, module)
            x = block(
                x,
                self.rope_cos_global,
                self.rope_sin_global,
                self.rope_cos_local,
                self.rope_sin_local,
                pos=pos,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))
        return logits

    def load_fetched_model(self, fetched: FetchedModel) -> None:
        """Copy Llama tensors from a fetched safetensors checkpoint."""
        # TODO
        self.eval()


class Gemma3Tokenizer:
    """
    Thin wrapper around tiktoken that keeps track of Llama-3 special IDs.
    Need to be init with a tokenizer file.
    """

    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        from_file = cast(Callable[[str], Tokenizer], cast(Any, Tokenizer).from_file)
        self.tok = from_file(str(tok_file))
        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        encode = cast(Callable[[str], Any], cast(Any, self.tok).encode)
        encoded = encode(text)
        return cast(list[int], encoded.ids)

    def decode(self, ids: list[int]) -> str:
        decode = cast(Any, self.tok).decode
        return cast(str, decode(ids, skip_special_tokens=False))
