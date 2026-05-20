"""
Gemma3.

- GQA attention with sliding window mechanism.
- Support quantization and KV cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, cast, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .rope import precompute_rope_cache, apply_rope
from .kv_cache import KVCache
from .quantization import QuantizedLinear, QuantizedWeight, WeightMode

if TYPE_CHECKING:
    from .model_ir import ModelIR


class Gemma3Config(TypedDict):
    """
    Internal configuration, obtained from ModelIR intermediate
    representation with Gemmma3.config_from_ir method.
    """

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
        """
        Args:
            x: Hidden states with shape ``[..., emb_dim]``.

        Returns:
            Hidden states with shape ``[..., emb_dim]``.
        """
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
        """
        Args:
            x: Input tensor with shape ``[..., emb_dim]``.

        Returns:
            Normalized tensor with shape ``[..., emb_dim]``.
        """
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

        ``x`` has shape ``[batch, tokens, d_in]`` and the returned tensor has
        shape ``[batch, tokens, d_out]``.  Unlike Llama in this repo, Gemma3's
        attention projection size is ``num_heads * head_dim``; ``head_dim`` is
        read from config and does not need to equal ``emb_dim // num_heads``.

        ``cos`` and ``sin`` are the precomputed RoPE cache selected by the
        transformer block with shape ``[context_length, head_dim]``: local RoPE
        for sliding-window layers, global RoPE for full-attention layers.  Query
        and key states are RMS-normalized before RoPE, then attended with GQA
        expansion, causal masking, optional sliding window masking, optional
        logit softcapping, and the Gemma3 ``query_pre_attn_scalar`` scale.
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

        queries = queries.transpose(1, 2)
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

        # Expand grouped K/V heads to match query heads.
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

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.q_size)
        return self.out_proj(context_vec)  # optional projection


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
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, emb_dim]``.
            cos_global: Global RoPE cosine cache with shape
                ``[context_length, head_dim]``.
            sin_global: Global RoPE sine cache with shape
                ``[context_length, head_dim]``.
            cos_local: Local RoPE cosine cache with shape
                ``[context_length, head_dim]``.
            sin_local: Local RoPE sine cache with shape
                ``[context_length, head_dim]``.
            pos: Optional starting token position used for RoPE
                or cached decoding.
            kv_cache: Optional key/value cache used
                during autoregressive decoding.
            layer_idx: Layer index used to read or update the cache.

        Returns:
            Hidden states with shape ``[batch, tokens, emb_dim]``.
        """
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
        return shortcut + x_ffn


class Gemma3Model(nn.Module):
    """
    Gemma3 text-only causal language model.

    Input is token IDs with shape ``[batch, tokens]``.  Output is logits with
    shape ``[batch, tokens, vocab_size]``.  The embedding output is multiplied by
    ``sqrt(hidden_size)`` to match HF Gemma3's scaled word embedding.  The model
    precomputes separate local and global RoPE caches because Gemma3 alternates
    sliding-window and full-attention layers.

    This class intentionally implements only the text decoder. Multimodal image
    towers/projectors are outside this model and must be filtered by loaders.
    """

    rope_cos_global: torch.Tensor
    rope_sin_global: torch.Tensor
    rope_cos_local: torch.Tensor
    rope_sin_local: torch.Tensor

    def __init__(self, cfg: Gemma3Config, weight_mode: WeightMode = "dense"):
        super().__init__()

        assert (
            cfg["layer_types"] is not None
            and len(cfg["layer_types"]) == cfg["n_layers"]
        )

        self.context_length = cfg["context_length"]
        self.weight_mode = weight_mode
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

    @staticmethod
    def config_from_ir(ir: ModelIR) -> Gemma3Config:
        """
        Build a model configuration from normalized IR.

        Args:
            ir: Normalized model IR that supplies configuration
                and canonical weights.

        Returns:
            Configuration dictionary accepted by the model constructor.
        """
        if ir.architecture != "gemma3":
            raise ValueError(f"expected gemma3 IR, got {ir.architecture!r}")
        layer_types = ir.config.get("layer_types")
        if not isinstance(layer_types, list):
            raise ValueError("IR config field 'layer_types' must be list[str]")
        layer_types = cast(list[Any], layer_types)
        if not all(isinstance(item, str) for item in layer_types):
            raise ValueError("IR config field 'layer_types' must be list[str]")
        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_kv_groups": ir.config.require_int("num_key_value_heads"),
            "sliding_window": ir.config.require_int("sliding_window"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "hidden_dim": ir.config.require_int("intermediate_size"),
            "head_dim": ir.config.require_int("head_dim"),
            "rope_base": ir.config.require_float("rope_base"),
            "rope_local_base": ir.config.require_float("rope_local_base"),
            "rope_interleaved": bool(ir.config.get("rope_interleaved", False)),
            "layer_types": cast(list[str], layer_types),
            "rms_norm_eps": ir.config.require_float("rms_norm_eps"),
            "query_pre_attn_scalar": ir.config.require_int("query_pre_attn_scalar"),
            "final_logit_softcapping": ir.config.optional_float(
                "final_logit_softcapping"
            ),
            "attn_logit_softcapping": ir.config.optional_float(
                "attn_logit_softcapping"
            ),
            "attention_bias": bool(ir.config.get("attention_bias", False)),
            "dtype": torch.float32,
        }

    def forward(
        self,
        in_idx: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
    ):
        """
        Args:
            in_idx: Token ids with shape ``[batch, tokens]``.
            pos: Optional starting token position used for RoPE
                or cached decoding.
            kv_cache: Optional key/value cache used during
                autoregressive decoding.

        Returns:
            Logits with shape ``[batch, tokens, vocab_size]``.
        """
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

    def load_ir_weights(self, ir: ModelIR) -> None:
        """
        Copy Gemma3 tensors from canonical IR.

        Dense mode copies tensors into ``nn.Linear`` modules.  Quantized mode
        installs ``QuantizedLinear`` for linear weights represented as
        ``QuantizedWeight`` while dense tensors such as embeddings and norms are
        still copied normally.
        """
        if ir.architecture != "gemma3":
            raise ValueError(f"expected gemma3 IR, got {ir.architecture!r}")
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight,
                self._dense_weight(ir.weights, "token_embedding.weight"),
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Gemma3TransformerBlock, module)
                prefix = f"layers.{layer_idx}"
                self._load_linear_weight(
                    block.att,
                    "W_query",
                    self._weight(ir.weights, f"{prefix}.attention.q_proj.weight"),
                )
                self._copy_optional_param(
                    block.att.W_query.bias,
                    self._optional_weight(
                        ir.weights, f"{prefix}.attention.q_proj.bias"
                    ),
                )
                self._load_linear_weight(
                    block.att,
                    "W_key",
                    self._weight(ir.weights, f"{prefix}.attention.k_proj.weight"),
                )
                self._copy_optional_param(
                    block.att.W_key.bias,
                    self._optional_weight(
                        ir.weights, f"{prefix}.attention.k_proj.bias"
                    ),
                )
                self._load_linear_weight(
                    block.att,
                    "W_value",
                    self._weight(ir.weights, f"{prefix}.attention.v_proj.weight"),
                )
                self._copy_optional_param(
                    block.att.W_value.bias,
                    self._optional_weight(
                        ir.weights, f"{prefix}.attention.v_proj.bias"
                    ),
                )
                self._load_linear_weight(
                    block.att,
                    "out_proj",
                    self._weight(ir.weights, f"{prefix}.attention.o_proj.weight"),
                )
                self._copy_optional_param(
                    block.att.out_proj.bias,
                    self._optional_weight(
                        ir.weights, f"{prefix}.attention.o_proj.bias"
                    ),
                )
                self._copy_param(
                    block.att.q_norm.scale,
                    self._dense_weight(ir.weights, f"{prefix}.attention.q_norm.weight"),
                )
                self._copy_param(
                    block.att.k_norm.scale,
                    self._dense_weight(ir.weights, f"{prefix}.attention.k_norm.weight"),
                )

                self._copy_param(
                    block.input_layernorm.scale,
                    self._dense_weight(ir.weights, f"{prefix}.input_norm.weight"),
                )
                self._copy_param(
                    block.post_attention_layernorm.scale,
                    self._dense_weight(
                        ir.weights,
                        f"{prefix}.post_attention_norm.weight",
                    ),
                )
                self._copy_param(
                    block.pre_feedforward_layernorm.scale,
                    self._dense_weight(ir.weights, f"{prefix}.pre_ffn_norm.weight"),
                )
                self._copy_param(
                    block.post_feedforward_layernorm.scale,
                    self._dense_weight(ir.weights, f"{prefix}.post_ffn_norm.weight"),
                )

                self._load_linear_weight(
                    block.ff,
                    "fc1",
                    self._weight(ir.weights, f"{prefix}.feed_forward.gate_proj.weight"),
                )
                self._load_linear_weight(
                    block.ff,
                    "fc2",
                    self._weight(ir.weights, f"{prefix}.feed_forward.up_proj.weight"),
                )
                self._load_linear_weight(
                    block.ff,
                    "fc3",
                    self._weight(ir.weights, f"{prefix}.feed_forward.down_proj.weight"),
                )

            self._copy_param(
                self.final_norm.scale,
                self._dense_weight(ir.weights, "final_norm.weight"),
            )
            self._load_linear_weight(
                self,
                "out_head",
                self._weight(ir.weights, "lm_head.weight"),
            )
        self.eval()

    @staticmethod
    def _copy_param(
        param: nn.Parameter | torch.Tensor | None, value: torch.Tensor
    ) -> None:
        """
        Copy parameter, verify shape before.

        Args:
            value: Value to validate, assign, or convert.
        """
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
        """
        Copy parameter, if exist.

        Args:
            value: Value to validate, assign, or convert.
        """
        if param is None:
            return
        if value is None:
            raise KeyError("missing Gemma3 bias weight")
        Gemma3Model._copy_param(param, value)

    @staticmethod
    def _optional_weight(
        weights: dict[str, Any], candidate: str
    ) -> torch.Tensor | None:
        """
        Return optional weight data when present.
        Only Tensor is supported here, raise an error if QuantizedWeight.

        Args:
            weights: Mapping of tensor names to loaded tensor data.
            name: Canonical or source field name to resolve.
        """
        value = weights.get(candidate)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, QuantizedWeight):
            raise TypeError(f"Weight {candidate!r} is quantized")
        return None

    @staticmethod
    def _weight(
        weights: dict[str, Any], candidate: str
    ) -> torch.Tensor | QuantizedWeight:
        """
        Return weight data. Can be Tensor or QuantizedWeight.

        Args:
            weights: Mapping of tensor names to loaded tensor data.
            name: Canonical or source field name to resolve.
        """
        value = weights.get(candidate)
        if isinstance(value, (torch.Tensor, QuantizedWeight)):
            return value
        raise KeyError(f"missing Gemma3 weight {candidate!r}")

    @staticmethod
    def _dense_weight(weights: dict[str, Any], name: str) -> torch.Tensor:
        """
        Return weight data, can only be dense Tensor.

        Args:
            weights: Mapping of tensor names to loaded tensor data.
            name: Canonical or source field name to resolve.
        """
        value = Gemma3Model._weight(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Gemma3 weight {name!r} is not a dense tensor")
        return value

    def _load_linear_weight(
        self, parent: nn.Module, attr: str, value: torch.Tensor | QuantizedWeight
    ) -> None:
        """
        Copy value into parent.attr. For example parent can be
        Gemma3GroupedQueryAttention and attr would be "W_query".
        If QuantizedWeight check that model is quantized too.

        Args:
            value: Value to validate, assign,
                or convert.
        """
        module = getattr(parent, attr)
        if not isinstance(module, (nn.Linear, QuantizedLinear)):
            raise TypeError(f"{attr!r} is not a linear module")
        in_features = int(module.in_features)
        out_features = int(module.out_features)

        if isinstance(value, QuantizedWeight):
            if self.weight_mode != "quantized":
                raise TypeError(
                    f"Gemma3 weight for {attr!r} is quantized but model is dense"
                )
            bias = None if module.bias is None else module.bias.detach()
            setattr(
                parent,
                attr,
                QuantizedLinear(
                    value,
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                ),
            )
            return

        if not isinstance(module, nn.Linear):
            raise TypeError(f"{attr!r} cannot accept a dense weight")
        self._copy_param(module.weight, value)


class Gemma3Tokenizer:
    """
    Thin wrapper around a Hugging Face tokenizer JSON file or GGUF file.

    ``encode`` maps text to token IDs and ``decode`` maps token IDs back to text
    without skipping special tokens.  GGUF files commonly embed the Hugging Face
    tokenizer JSON in metadata, so a ``.gguf`` path can be used directly.
    """

    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        if tok_file.suffix.lower() == ".gguf":
            self.tok = self._tokenizer_from_gguf(tok_file)
        else:
            from_file = cast(Callable[[str], Tokenizer], cast(Any, Tokenizer).from_file)
            self.tok = from_file(str(tok_file))
        self.eos_token_id = self.convert_tokens_to_ids("<end_of_turn>")
        self.pad_token_id = self.eos_token_id

    @staticmethod
    def _tokenizer_from_gguf(gguf_path: Path) -> Tokenizer:
        """Load tokenizer data embedded in a GGUF file."""
        from gguf import GGUFReader

        from .gguf import find_gguf_file, tokenizer_json_from_gguf

        tokenizer_json = tokenizer_json_from_gguf(gguf_path)
        if tokenizer_json is None:
            reader = GGUFReader(find_gguf_file(gguf_path))
            tokenizer_model = cast(
                str, reader.fields["tokenizer.ggml.model"].contents()
            )
            if tokenizer_model != "llama":
                raise ValueError(
                    f"unsupported Gemma3 GGUF tokenizer model {tokenizer_model!r}"
                )

            tokens = cast(list[str], reader.fields["tokenizer.ggml.tokens"].contents())
            scores = cast(
                list[float], reader.fields["tokenizer.ggml.scores"].contents()
            )
            unk_id = cast(
                int, reader.fields["tokenizer.ggml.unknown_token_id"].contents()
            )
            tokenizers_models = __import__("tokenizers.models").models
            tokenizers_decoders = __import__("tokenizers.decoders").decoders
            tok = Tokenizer(
                tokenizers_models.Unigram(
                    list(zip(tokens, scores)),
                    unk_id=unk_id,
                    byte_fallback=True,
                )
            )
            tok.decoder = tokenizers_decoders.ByteFallback()
            return tok

        from_str = cast(Callable[[str], Tokenizer], cast(Any, Tokenizer).from_str)
        return from_str(tokenizer_json)

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "Gemma3Tokenizer":
        """Build a Gemma3 tokenizer from tokenizer data embedded in a GGUF file."""
        return cls(gguf_path)

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token ids.

        Args:
            text: Input text to encode.

        Returns:
            Encoded token ids.
        """
        encode = cast(Callable[[str], Any], cast(Any, self.tok).encode)
        encoded = encode(text)
        return cast(list[int], encoded.ids)

    def encode_instruct_prompt(self, user_text: str) -> list[int]:
        """Encode a minimal Gemma3 user-to-model chat prompt."""
        encoded = self.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(encoded, dict):
            raise TypeError("expected tokenized chat template output")
        return encoded["input_ids"]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]] | str:
        """
        Apply Gemma's text-only chat wire format.

        This intentionally covers the user/model turns used by the functional
        tests and eval adapters without depending on ``transformers``.
        """
        prompt = "<bos>"
        for message in messages:
            role = message["role"]
            if role == "assistant":
                role = "model"
            if role not in {"user", "model"}:
                raise ValueError(f"unsupported Gemma3 chat role {role!r}")
            prompt += f"<start_of_turn>{role}\n{message['content']}<end_of_turn>\n"
        if add_generation_prompt:
            prompt += "<start_of_turn>model\n"
        if not tokenize:
            return prompt
        return {"input_ids": self.encode(prompt)}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        """Return the token id for a special token string when known."""
        token_to_id = cast(Any, self.tok).token_to_id
        token_id = cast(int | None, token_to_id(token))
        return token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode token ids into text.

        Args:
            ids: Token ids to decode.

        Returns:
            Decoded text.
        """
        decode = cast(Any, self.tok).decode
        return cast(str, decode(ids, skip_special_tokens=False))
