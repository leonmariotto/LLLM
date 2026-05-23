"""
LLama2.
This implementation use good old MHA (Multi-Head Attention) so it is
NOT COMPATIBLE as is with largest Llama2 model which already used GQA.
A GQA implementation is present in Llama3 and a patch to support them
here could be done. This concern: Llama2 43B and 70B.
"""

from __future__ import annotations

import torch
from torch import nn
import sentencepiece as spm
from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypedDict, cast

from .rope import precompute_rope_cache, apply_rope
from .norm import RMSNorm
from .kv_cache import KVCache

if TYPE_CHECKING:
    from .model_ir import ModelIR, ModelWeightsIR


class Llama2Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    rope_theta: float
    dtype: torch.dtype


class Llama2FeedForward(nn.Module):
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
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class Llama2MultiHeadAttention(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Attention mechanism involve 3 trainable matrix : query, key, values.

    Implement Causal mask and dropout.
    Default dropout is 0 which result in identity matrix.

    Multi-headed attention: d_out is splited in num_head parts. Each head
    produce a part of d_out (head_dim, calculated at init), and at the end
    context_vec is reshaped to the correct size.
    So things can be parallel.

    Implement RoPe. Enabled by default.
    """

    # Need to tell pyright that the "mask" registered by register_buffer method
    # is an tensor, to avoid typing errors.
    mask: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,  # No dropout by default.
        qkv_bias: bool = False,  # No bias.
        use_rope: bool = True,  # Use RoPe by default.
        rope_base: float = 10000.0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        assert num_heads != 0, "num_head shall not be 0"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.context_length = context_length
        self.use_rope = use_rope

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(
            d_out, d_out, bias=qkv_bias
        )  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        if self.use_rope:
            cos, sin = precompute_rope_cache(
                seq_len=self.context_length,
                head_dim=self.head_dim,
                base=rope_base,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, d_in]``.
            pos: Optional starting token position used for RoPE
                or cached decoding.
            kv_cache: Optional key/value cache used during
                autoregressive decoding.
            layer_idx: Layer index used to read or update the cache.

        Returns:
            Hidden states with shape ``[batch, tokens, d_out]``.
        """
        b, num_tokens, d_in = x.shape

        assert self.d_in == d_in, "invalid d_in (embedding size)"

        # As in `CausalAttention`, for inputs where `num_tokens` exceeds
        # `context_length`, this will result in errors in the mask creation further
        # below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that
        # inputs do not exceed `context_length` before reaching this forward method.

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # About tensor.view and tensor.transpose methods:
        # Tensor view method reshape a tensor, without moving elements in memory.
        # Whereas transpose change how dimensions are indexed.
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys_new = keys_new.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values_new = values_new.transpose(1, 2)

        # Get the absolute position for the current layer.
        next_pos = 0
        if kv_cache is not None:
            if self.use_rope is False:
                raise ValueError("RoPE must be enabled for using KVCache")
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            next_pos = kv_cache.layer_next_pos(layer_idx)

        start_pos = next_pos if pos is None else pos
        if self.use_rope:
            # Add RoPE after Q/K projection and head reshaping and before computing
            # attention scores.
            # We pass to apply_rope only sin/cos for the current position. Compute each index
            # position from a pos offset.
            # If pos is None we infer from the cache absolute next position.
            position_ids = torch.arange(
                start_pos,
                start_pos + num_tokens,
                device=x.device,
            )
            assert int(position_ids[-1]) < self.context_length, (
                "RoPE position exceeds precomputed context length"
            )
            cos = self.rope_cos[position_ids]
            sin = self.rope_sin[position_ids]
            queries = apply_rope(queries, cos, sin)
            keys_new = apply_rope(keys_new, cos, sin)

        if kv_cache is None:
            keys, values = keys_new, values_new
            key_start_pos = start_pos if self.use_rope else 0
        else:
            assert layer_idx is not None
            cache_view = kv_cache.update(
                layer_idx, keys_new, values_new, start_pos=start_pos
            )
            keys, values = cache_view.keys, cache_view.values
            key_start_pos = cache_view.start_pos

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # `queries` has shape (batch, num_heads, num_tokens, head_dim).
        # So shape[-2] is the query-token dimension.
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        key_positions = torch.arange(
            key_start_pos,
            key_start_pos + num_tokens_K,
            device=x.device,
        )
        query_positions = torch.arange(
            start_pos,
            start_pos + num_tokens_Q,
            device=x.device,
        )
        mask_bool = key_positions[None, :] > query_positions[:, None]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)  # optional projection


class Llama2TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama2Config):
        super().__init__()
        self.att = Llama2MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            rope_base=cfg["rope_theta"],
            dtype=cfg["dtype"],
        )
        self.ff = Llama2FeedForward(cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"])

        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(
        self,
        x: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ):
        # Shortcut connection for attention block
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, emb_dim]``.
            pos: Optional starting token position used for RoPE or cached decoding.
            kv_cache: Optional key/value cache used during autoregressive decoding.
            layer_idx: Layer index used to read or update the cache.

        Returns:
            Hidden states with shape ``[batch, tokens, emb_dim]``.
        """
        shortcut = x
        x = self.norm1(x)
        if kv_cache is None:
            x = self.att(x, pos)  # Shape [batch_size, num_tokens, emb_size]
        else:
            x = self.att(x, pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + shortcut  # Add the original input back


class Llama2Tokenizer:
    def __init__(self, tokenizer_file: str) -> None:
        sp = spm.SentencePieceProcessor()
        load = cast(Callable[[str], bool], getattr(sp, "load"))
        load(tokenizer_file)
        self.tokenizer = sp

    def get_eos(self) -> int | None:
        eos_id = cast(Callable[[], int], getattr(self.tokenizer, "eos_id"))
        eos = eos_id()
        return eos if eos >= 0 else None

    def encode(self, input: str) -> list[int]:
        encode = cast(
            Callable[..., list[int]],
            getattr(self.tokenizer, "encode"),
        )
        return encode(input, out_type=int)

    def decode(self, tok: list[int]) -> str:
        decode = cast(
            Callable[[Sequence[int]], str],
            getattr(self.tokenizer, "decode"),
        )
        return decode(tok)


class Llama2Model(nn.Module):
    def __init__(self, cfg: Llama2Config):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.Sequential(
            *[Llama2TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

    @staticmethod
    def config_from_ir(ir: ModelIR) -> Llama2Config:
        if ir.architecture != "llama2":
            raise ValueError(f"expected llama2 IR, got {ir.architecture!r}")
        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "hidden_dim": ir.config.require_int("intermediate_size"),
            "rope_theta": ir.config.require_float("rope_theta"),
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
            pos: Optional starting token position used for RoPE or cached decoding.
            kv_cache: Optional key/value cache used during autoregressive decoding.

        Returns:
            Logits with shape ``[batch, tokens, vocab_size]``.
        """
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        for layer_idx, module in enumerate(self.trf_blocks):
            block = cast(Llama2TransformerBlock, module)
            x = block(x, pos=pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = self.final_norm(x)
        return self.out_head(x)

    def load_ir_weights(self, ir: ModelIR) -> None:
        """Copy canonical Llama2 IR tensors into this model."""
        if ir.architecture != "llama2":
            raise ValueError(f"expected llama2 IR, got {ir.architecture!r}")
        weights = ir.weights
        with torch.no_grad():
            embedding_weight = self._optional_weight(weights, "token_embedding.weight")
            if embedding_weight is None:
                embedding_weight = self._weight(weights, "lm_head.weight")
            self._copy_param(self.tok_emb.weight, embedding_weight)

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Llama2TransformerBlock, module)
                prefix = f"layers.{layer_idx}"

                self._copy_param(
                    block.att.W_query.weight,
                    self._weight(weights, f"{prefix}.attention.q_proj.weight"),
                )
                self._copy_param(
                    block.att.W_key.weight,
                    self._weight(weights, f"{prefix}.attention.k_proj.weight"),
                )
                self._copy_param(
                    block.att.W_value.weight,
                    self._weight(weights, f"{prefix}.attention.v_proj.weight"),
                )
                self._copy_param(
                    block.att.out_proj.weight,
                    self._weight(weights, f"{prefix}.attention.o_proj.weight"),
                )

                self._copy_param(
                    block.norm1.weight,
                    self._weight(weights, f"{prefix}.input_norm.weight"),
                )
                self._copy_param(
                    block.norm2.weight,
                    self._weight(weights, f"{prefix}.post_attention_norm.weight"),
                )

                self._copy_param(
                    block.ff.fc1.weight,
                    self._weight(weights, f"{prefix}.feed_forward.gate_proj.weight"),
                )
                self._copy_param(
                    block.ff.fc2.weight,
                    self._weight(weights, f"{prefix}.feed_forward.up_proj.weight"),
                )
                self._copy_param(
                    block.ff.fc3.weight,
                    self._weight(weights, f"{prefix}.feed_forward.down_proj.weight"),
                )

            self._copy_param(
                self.final_norm.weight, self._weight(weights, "final_norm.weight")
            )
            self._copy_param(
                self.out_head.weight, self._weight(weights, "lm_head.weight")
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
    def _optional_weight(weights: ModelWeightsIR, name: str) -> torch.Tensor | None:
        if name in weights:
            value = weights[name]
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(f"Llama2 weight {name!r} is quantized")
        return None

    @staticmethod
    def _weight(weights: ModelWeightsIR, name: str) -> torch.Tensor:
        if name in weights:
            value = weights[name]
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(f"Llama2 weight {name!r} is quantized")
        raise KeyError(f"missing Llama2 IR weight {name!r}")
