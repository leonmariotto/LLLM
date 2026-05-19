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
    rms_norm_eps: float
    query_pre_attn_scalar: int
    final_logit_softcapping: float | None
    attn_logit_softcapping: float | None
    attention_bias: bool
    dtype: torch.dtype


def gemma3_config_from_fetched(config: dict[str, Any]) -> Gemma3Config:
    """
    Translate Hugging Face Gemma3 config data into this project's text config.

    Input can be either a text-only ``Gemma3TextConfig`` dict or a multimodal
    ``Gemma3Config`` dict containing ``text_config``.  The returned config keeps
    only the language-model fields used here: GQA sizes, independent attention
    ``head_dim``, local/global RoPE bases, sliding/full layer types, Gemma RMSNorm
    epsilon, attention softcapping, and final-logit softcapping.
    """
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = cast(dict[str, Any], text_config)

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
        if value is None:
            return ["sliding_attention"] * _int_config(
                config, "num_hidden_layers", fallback_key="n_layers"
            )
        if not isinstance(value, list):
            raise ValueError(f"config value {key!r} must be a list")
        return cast(list[str], value)

    def _optional_float_config(config: dict[str, Any], key: str) -> float | None:
        value = config.get(key)
        if value is None:
            return None
        if isinstance(value, int):
            return float(value)
        if not isinstance(value, float):
            raise ValueError(f"config value {key!r} must be a float or None")
        return value

    def _rope_base_config(
        config: dict[str, Any], layer_type: str, default: float
    ) -> float:
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_parameters = cast(dict[str, Any], rope_parameters)
            layer_parameters = rope_parameters.get(layer_type)
            if isinstance(layer_parameters, dict):
                layer_parameters = cast(dict[str, Any], layer_parameters)
                value = layer_parameters.get("rope_theta")
                if isinstance(value, int):
                    return float(value)
                if isinstance(value, float):
                    return value
        return _float_config(config, "rope_theta", default=default, allow_int=True)

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
        "rope_base": _rope_base_config(config, "full_attention", 1000000.0),
        "rope_local_base": _rope_base_config(config, "sliding_attention", 10000.0),
        "rope_interleaved": _bool_config(config, "rope_interleaved", default=False),
        "layer_types": _layer_types_config(config, "layer_types"),
        "rms_norm_eps": _float_config(config, "rms_norm_eps", default=1e-6),
        "query_pre_attn_scalar": _int_config(
            config, "query_pre_attn_scalar", fallback_key="head_dim"
        ),
        "final_logit_softcapping": _optional_float_config(
            config, "final_logit_softcapping"
        ),
        "attn_logit_softcapping": _optional_float_config(
            config, "attn_logit_softcapping"
        ),
        "attention_bias": _bool_config(config, "attention_bias", default=False),
        "dtype": torch.float32,
    }


class Gemma3FeedForward(nn.Module):
    """
    Gemma3 gated MLP block.

    Input and output shape are ``[..., emb_dim]``. Internally the block computes
    ``down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))``.  This matches the HF
    Gemma3 MLP naming where ``fc1`` is ``gate_proj``, ``fc2`` is ``up_proj``, and
    ``fc3`` is ``down_proj``.
    """

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
        head_dim: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        sliding_window: int | None,
        dropout: float = 0.0,  # No dropout by default.
        qkv_bias: bool = False,  # No bias.
        rope_interleaved: bool = False,
        query_pre_attn_scalar: Optional[int] = None,
        attn_logit_softcapping: float | None = None,
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
        - sliding window: number of recent key/value tokens visible to each query.
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
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_groups * head_dim
        self.context_length = context_length
        self.num_kv_groups = num_kv_groups
        self.kv_group_size = num_heads // num_kv_groups
        self.rope_interleaved = rope_interleaved
        self.attn_logit_softcapping = attn_logit_softcapping

        self.W_query = nn.Linear(d_in, self.q_size, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.kv_size, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.kv_size, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(
            self.q_size, d_out, bias=qkv_bias, dtype=dtype
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
        Project tokens through Gemma3 grouped-query self-attention.

        ``x`` has shape ``[batch, tokens, emb_dim]`` and the returned tensor has
        shape ``[batch, tokens, d_out]``.  Unlike Llama in this repo, Gemma3's
        attention projection size is ``num_heads * head_dim``; ``head_dim`` is
        read from config and does not need to equal ``emb_dim // num_heads``.

        ``cos`` and ``sin`` are the precomputed RoPE cache selected by the
        transformer block: local RoPE for sliding-window layers, global RoPE for
        full-attention layers.  Query and key states are RMS-normalized before
        RoPE, then attended with GQA expansion, causal masking, optional sliding
        window masking, optional logit softcapping, and the Gemma3
        ``query_pre_attn_scalar`` scale.
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
            device=x.device,
        )
        # The boolean comparison is applied element-wise after broadcasting dimension.
        # [1, K] broadcasts to [Q, K]
        # [Q, 1] broadcasts to [Q, K]
        # So < returns a boolean tensor of shape: [Q, K]
        causal_mask = key_positions[None, :] > query_positions[:, None]
        if self.sliding_window is None:
            mask_bool = causal_mask
        else:
            # During cached generation the cache may contain more tokens than the
            # sliding window. The mask keeps attention local even before a
            # dedicated sliding KV cache is added.
            sliding_window_mask = key_positions[None, :] < (
                query_positions[:, None] - self.sliding_window + 1
            )
            mask_bool = causal_mask | sliding_window_mask

        attn_scores = attn_scores * self.scaling
        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores)
            attn_scores = attn_scores * self.attn_logit_softcapping

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.q_size)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class Gemma3TransformerBlock(nn.Module):
    """
    One Gemma3 decoder block.

    The block applies input RMSNorm, self-attention, post-attention RMSNorm, then
    a residual add.  It then applies pre-FFN RMSNorm, the gated MLP,
    post-FFN RMSNorm, and a second residual add.  This differs from the Llama
    blocks in this repo because Gemma3 has both pre and post RMSNorms around the
    attention and feed-forward sublayers.

    ``attn_type`` selects the RoPE cache and mask behavior: ``sliding_attention``
    uses the local RoPE base and sliding-window mask; ``full_attention`` uses the
    global RoPE base and only the causal mask.
    """

    def __init__(self, cfg: Gemma3Config, attn_type: str):
        super().__init__()

        self.attn_type = attn_type

        self.att = Gemma3GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            head_dim=cfg["head_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            sliding_window=(
                cfg["sliding_window"] if attn_type == "sliding_attention" else None
            ),
            qkv_bias=cfg["attention_bias"],
            rope_interleaved=cfg["rope_interleaved"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            attn_logit_softcapping=cfg["attn_logit_softcapping"],
            dtype=cfg["dtype"],
        )
        self.ff = Gemma3FeedForward(cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"])

        norm_eps = cfg["rms_norm_eps"]
        self.input_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(cfg["emb_dim"], eps=norm_eps)

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
        x = self.post_attention_layernorm(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn

        return x


class Gemma3Model(nn.Module):
    """
    Gemma3 text-only causal language model.

    Input is token IDs with shape ``[batch, tokens]``.  Output is logits with
    shape ``[batch, tokens, vocab_size]``.  The embedding output is multiplied by
    ``sqrt(hidden_size)`` to match HF Gemma3's scaled word embedding.  The model
    precomputes separate local and global RoPE caches because Gemma3 alternates
    sliding-window and full-attention layers.

    This class intentionally implements only the text decoder.  Multimodal
    Gemma3 checkpoints can still provide compatible text weights under
    ``model.language_model.*`` aliases, but image towers/projectors are outside
    this model.
    """

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
        self.final_logit_softcapping = cfg["final_logit_softcapping"]
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.dtype = cfg["dtype"]

        self.trf_blocks = nn.ModuleList(
            Gemma3TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]
        )

        self.final_norm = Gemma3RMSNorm(cfg["emb_dim"], eps=cfg["rms_norm_eps"])
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
        x = self.tok_emb(in_idx) * (self.embedded_dim**0.5)
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
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    def load_fetched_model(self, fetched: FetchedModel) -> None:
        """
        Copy Gemma3 text weights from a fetched Hugging Face checkpoint.

        Supports both text-only names such as ``model.layers.0.self_attn.q_proj``
        and multimodal names such as
        ``model.language_model.layers.0.self_attn.q_proj``.  Tied-output
        checkpoints may omit ``lm_head.weight``; in that case the token embedding
        weight is reused for ``out_head``.
        """
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight,
                self._weight(fetched.weights, "tok_embeddings.weight"),
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Gemma3TransformerBlock, module)
                prefix = f"layers.{layer_idx}"
                self._copy_param(
                    block.att.W_query.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wq.weight"),
                )
                self._copy_optional_param(
                    block.att.W_query.bias,
                    self._optional_weight(
                        fetched.weights, f"{prefix}.attention.wq.bias"
                    ),
                )
                self._copy_param(
                    block.att.W_key.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wk.weight"),
                )
                self._copy_optional_param(
                    block.att.W_key.bias,
                    self._optional_weight(
                        fetched.weights, f"{prefix}.attention.wk.bias"
                    ),
                )
                self._copy_param(
                    block.att.W_value.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wv.weight"),
                )
                self._copy_optional_param(
                    block.att.W_value.bias,
                    self._optional_weight(
                        fetched.weights, f"{prefix}.attention.wv.bias"
                    ),
                )
                self._copy_param(
                    block.att.out_proj.weight,
                    self._weight(fetched.weights, f"{prefix}.attention.wo.weight"),
                )
                self._copy_optional_param(
                    block.att.out_proj.bias,
                    self._optional_weight(
                        fetched.weights, f"{prefix}.attention.wo.bias"
                    ),
                )
                self._copy_param(
                    block.att.q_norm.scale,
                    self._weight(fetched.weights, f"{prefix}.attention.q_norm.weight"),
                )
                self._copy_param(
                    block.att.k_norm.scale,
                    self._weight(fetched.weights, f"{prefix}.attention.k_norm.weight"),
                )

                self._copy_param(
                    block.input_layernorm.scale,
                    self._weight(fetched.weights, f"{prefix}.input_layernorm.weight"),
                )
                self._copy_param(
                    block.post_attention_layernorm.scale,
                    self._weight(
                        fetched.weights,
                        f"{prefix}.post_attention_layernorm.weight",
                    ),
                )
                self._copy_param(
                    block.pre_feedforward_layernorm.scale,
                    self._weight(
                        fetched.weights,
                        f"{prefix}.pre_feedforward_layernorm.weight",
                    ),
                )
                self._copy_param(
                    block.post_feedforward_layernorm.scale,
                    self._weight(
                        fetched.weights,
                        f"{prefix}.post_feedforward_layernorm.weight",
                    ),
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
                self.final_norm.scale, self._weight(fetched.weights, "norm.weight")
            )
            self._copy_param(
                self.out_head.weight, self._lm_head_weight(fetched.weights)
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
    def _copy_optional_param(
        param: nn.Parameter | torch.Tensor | None, value: torch.Tensor | None
    ) -> None:
        if param is None:
            return
        if value is None:
            raise KeyError("missing Gemma3 bias weight")
        Gemma3Model._copy_param(param, value)

    @staticmethod
    def _lm_head_weight(weights: dict[str, Any]) -> torch.Tensor:
        for name in Gemma3Model._weight_names("output.weight"):
            value = weights.get(name)
            if isinstance(value, torch.Tensor):
                return value
        return Gemma3Model._weight(weights, "tok_embeddings.weight")

    @staticmethod
    def _optional_weight(weights: dict[str, Any], name: str) -> torch.Tensor | None:
        for candidate in Gemma3Model._weight_names(name):
            value = weights.get(candidate)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @staticmethod
    def _weight(weights: dict[str, Any], name: str) -> torch.Tensor:
        for candidate in Gemma3Model._weight_names(name):
            value = weights.get(candidate)
            if isinstance(value, torch.Tensor):
                return value
        names = ", ".join(Gemma3Model._weight_names(name))
        raise KeyError(f"missing Gemma3 weight {name!r}; tried {names}")

    @staticmethod
    def _weight_names(name: str) -> list[str]:
        names = [name]
        exact_aliases = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "norm.weight": "model.norm.weight",
        }
        if name in exact_aliases:
            names.append(exact_aliases[name])
        if name == "tok_embeddings.weight":
            names.append("model.language_model.embed_tokens.weight")
        elif name == "norm.weight":
            names.append("model.language_model.norm.weight")
        elif name == "output.weight":
            names.append("model.language_model.embed_tokens.weight")

        prefix = "layers."
        if name.startswith(prefix):
            parts = name.split(".")
            if len(parts) == 5:
                _, layer_idx, group, weight_name, suffix = parts
                if suffix in {"weight", "bias"}:
                    layer_aliases = {
                        ("attention", "wq"): "self_attn.q_proj",
                        ("attention", "wk"): "self_attn.k_proj",
                        ("attention", "wv"): "self_attn.v_proj",
                        ("attention", "wo"): "self_attn.o_proj",
                        ("attention", "q_norm"): "self_attn.q_norm",
                        ("attention", "k_norm"): "self_attn.k_norm",
                        ("feed_forward", "w1"): "mlp.gate_proj",
                        ("feed_forward", "w2"): "mlp.down_proj",
                        ("feed_forward", "w3"): "mlp.up_proj",
                    }
                    hf_name = layer_aliases.get((group, weight_name))
                    if hf_name is not None:
                        names.append(f"model.layers.{layer_idx}.{hf_name}.{suffix}")
                        names.append(
                            f"model.language_model.layers.{layer_idx}."
                            f"{hf_name}.{suffix}"
                        )
            elif len(parts) == 4:
                _, layer_idx, weight_name, suffix = parts
                if suffix == "weight":
                    layer_norm_aliases = {
                        "input_layernorm": "input_layernorm",
                        "post_attention_layernorm": "post_attention_layernorm",
                        "pre_feedforward_layernorm": "pre_feedforward_layernorm",
                        "post_feedforward_layernorm": "post_feedforward_layernorm",
                    }
                    hf_name = layer_norm_aliases.get(weight_name)
                    if hf_name is not None:
                        names.append(f"model.layers.{layer_idx}.{hf_name}.weight")
                        names.append(
                            f"model.language_model.layers.{layer_idx}.{hf_name}.weight"
                        )

        return names


class Gemma3Tokenizer:
    """
    Thin wrapper around a Hugging Face tokenizer JSON file.

    ``encode`` maps text to token IDs and ``decode`` maps token IDs back to text
    without skipping special tokens.  Gemma3 tokenizer assets are gated for the
    real model, so tests currently validate the model path independently of this
    wrapper.
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
