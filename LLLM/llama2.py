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
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypedDict, cast

from .rope import precompute_rope_cache, apply_rope
from .norm import RMSNorm
from .kv_cache import KVCache

if TYPE_CHECKING:
    from .fetch import FetchedModel


class Llama2Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    rope_theta: float
    dtype: torch.dtype


def llama2_config_from_fetched(config: dict[str, Any]) -> Llama2Config:
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
        "n_layers": _int_config(config, "num_hidden_layers", fallback_key="n_layers"),
        "hidden_dim": _hidden_dim_config(config),
        "rope_theta": _float_config(config, "rope_theta", default=10000.0),
        "dtype": torch.float32,
    }


# TODO use torch.functional version
class SiLU(nn.Module):
    """
    Implement SiLu activation function (aka Swish function).
    """

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Llama2FeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dtype: Optional[torch.dtype]):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
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
        """
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
        forward method is called by the nn.Module __call__ method.
        x is expected to be a batch of tensor of d_in size.
        (number of batch, number of token, embedding size)
        return a tensor of d_out size.

        pos: contains the (context-wide relative) starting index of the sequence.
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

        # Get the cache length for the current layer.
        past_tokens = 0
        if kv_cache is not None:
            if self.use_rope is False:
                raise ValueError("RoPE must be enabled for using KVCache")
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            past_tokens = kv_cache.layer_seq_len(layer_idx)

        if self.use_rope:
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
            cos = self.rope_cos[position_ids]
            sin = self.rope_sin[position_ids]
            queries = apply_rope(queries, cos, sin)
            keys_new = apply_rope(keys_new, cos, sin)

        if kv_cache is None:
            keys, values = keys_new, values_new
        else:
            assert layer_idx is not None
            keys, values = kv_cache.update(layer_idx, keys_new, values_new)

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

        # self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        if kv_cache is None:
            x = self.att(x, pos)  # Shape [batch_size, num_tokens, emb_size]
        else:
            x = self.att(x, pos, kv_cache=kv_cache, layer_idx=layer_idx)
        # x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class Llama2Tokenizer:
    def __init__(self, tokenizer_file: str) -> None:
        sp = spm.SentencePieceProcessor()
        load = cast(Callable[[str], bool], getattr(sp, "load"))
        load(tokenizer_file)
        self.tokenizer = sp

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
        # x = self.drop_emb(x)
        for layer_idx, module in enumerate(self.trf_blocks):
            block = cast(Llama2TransformerBlock, module)
            x = block(x, pos=pos, kv_cache=kv_cache, layer_idx=layer_idx)
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
                block = cast(Llama2TransformerBlock, module)
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
