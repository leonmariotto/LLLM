"""
LLama3
Support 3.1 and 3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, cast, Any
from importlib import import_module
import os
from pathlib import Path

from loguru import logger

import torch
import torch.nn as nn
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from .llama2 import Llama2FeedForward
from .norm import RMSNorm
from .rope import precompute_rope_cache, apply_rope, RopeFrequencyConfig
from .kv_cache import KVCache
from .quantization import QuantizedLinear, QuantizedWeight, WeightMode

if TYPE_CHECKING:
    from .model_ir import ModelIR


class Llama3Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_kv_groups: int  # GQA
    n_layers: int
    hidden_dim: int
    rope_theta: float
    rope_interleaved: bool
    freq_config: RopeFrequencyConfig | None
    dtype: torch.dtype


def _rope_scaling_float(rope_scaling: dict[str, Any], key: str) -> float:
    value = rope_scaling.get(key)
    if not isinstance(value, float):
        raise ValueError(f"IR rope_scaling field {key!r} must be a float")
    return value


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
        rope_interleaved: bool = False,
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
        Args:
            x: Hidden states with shape ``[batch, tokens, d_in]``.
            cos: RoPE cosine cache with shape
                ``[context_length, head_dim]``.
            sin: RoPE sine cache with shape
                ``[context_length, head_dim]``.
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

        # Get the absolute position for the current layer.
        next_pos = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            next_pos = kv_cache.layer_next_pos(layer_idx)

        # Add RoPE after Q/K projection and head reshaping and before computing
        # attention scores.
        # We pass to apply_rope only sin/cos for the current position. Compute each index
        # position from a pos offset.
        # If pos is None we infer from the cache absolute next position.
        start_pos = next_pos if pos is None else pos
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
            key_start_pos = start_pos
        else:
            assert layer_idx is not None
            cache_view = kv_cache.update(
                layer_idx, keys_new, values_new, start_pos=start_pos
            )
            keys, values = cache_view.keys, cache_view.values
            key_start_pos = cache_view.start_pos

        # Expand grouped K/V heads to match query heads.
        keys = keys.repeat_interleave(self.kv_group_size, dim=1)
        values = values.repeat_interleave(self.kv_group_size, dim=1)

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


class Llama3TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.att = Llama3GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_interleaved=cfg["rope_interleaved"],
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
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, emb_dim]``.
            cos: RoPE cosine cache with shape
                ``[context_length, head_dim]``.
            sin: RoPE sine cache with shape
                ``[context_length, head_dim]``.
            pos: Optional starting token position used for RoPE or cached decoding.
            kv_cache: Optional key/value cache used during autoregressive decoding.
            layer_idx: Layer index used to read or update the cache.

        Returns:
            Hidden states with shape ``[batch, tokens, emb_dim]``.
        """
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
        return x + shortcut  # Add the original input back


class Llama3Model(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: Llama3Config, weight_mode: WeightMode = "dense"):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.weight_mode = weight_mode
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

    @staticmethod
    def config_from_ir(ir: ModelIR) -> Llama3Config:
        if ir.architecture != "llama3":
            raise ValueError(f"expected llama3 IR, got {ir.architecture!r}")

        rope_scaling = ir.config.get("rope_scaling")
        freq_config: RopeFrequencyConfig | None = None
        if rope_scaling is not None:
            if not isinstance(rope_scaling, dict):
                raise ValueError("IR config field 'rope_scaling' must be a dict")
            rope_scaling = cast(dict[str, Any], rope_scaling)
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type != "llama3":
                raise ValueError(f"unsupported rope_scaling rope_type: {rope_type!r}")
            freq_config = {
                "factor": _rope_scaling_float(rope_scaling, "factor"),
                "low_freq_factor": _rope_scaling_float(rope_scaling, "low_freq_factor"),
                "high_freq_factor": _rope_scaling_float(
                    rope_scaling, "high_freq_factor"
                ),
                "original_context_len": int(
                    rope_scaling["original_max_position_embeddings"]
                ),
            }

        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_kv_groups": ir.config.require_int("num_key_value_heads"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "hidden_dim": ir.config.require_int("intermediate_size"),
            "rope_theta": ir.config.require_float("rope_theta"),
            "rope_interleaved": bool(ir.config.get("rope_interleaved", False)),
            "freq_config": freq_config,
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
        return self.out_head(x)

    def load_ir_weights(self, ir: ModelIR) -> None:
        """Copy canonical dense Llama3 IR tensors into this model."""
        if ir.architecture != "llama3":
            raise ValueError(f"expected llama3 IR, got {ir.architecture!r}")
        with torch.no_grad():
            embedding_weight = self._optional_dense_weight(
                ir.weights, "token_embedding.weight"
            )
            if embedding_weight is None:
                embedding_weight = self._dense_weight(ir.weights, "lm_head.weight")
            self._copy_param(self.tok_emb.weight, embedding_weight)

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Llama3TransformerBlock, module)
                prefix = f"layers.{layer_idx}"

                self._copy_param(
                    block.att.W_query.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.q_proj.weight"),
                )
                self._copy_param(
                    block.att.W_key.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.k_proj.weight"),
                )
                self._copy_param(
                    block.att.W_value.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.v_proj.weight"),
                )
                self._copy_param(
                    block.att.out_proj.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.o_proj.weight"),
                )

                self._copy_param(
                    block.norm1.weight,
                    self._dense_weight(ir.weights, f"{prefix}.input_norm.weight"),
                )
                self._copy_param(
                    block.norm2.weight,
                    self._dense_weight(
                        ir.weights, f"{prefix}.post_attention_norm.weight"
                    ),
                )

                self._copy_param(
                    block.ff.fc1.weight,
                    self._dense_weight(
                        ir.weights, f"{prefix}.feed_forward.gate_proj.weight"
                    ),
                )
                self._copy_param(
                    block.ff.fc2.weight,
                    self._dense_weight(
                        ir.weights, f"{prefix}.feed_forward.up_proj.weight"
                    ),
                )
                self._copy_param(
                    block.ff.fc3.weight,
                    self._dense_weight(
                        ir.weights, f"{prefix}.feed_forward.down_proj.weight"
                    ),
                )

            self._copy_param(
                self.final_norm.weight,
                self._dense_weight(ir.weights, "final_norm.weight"),
            )
            self._copy_param(
                self.out_head.weight,
                self._dense_weight(ir.weights, "lm_head.weight"),
            )

        self.eval()

    def load_quantized_ir_weights(self, ir: ModelIR) -> None:
        """Install quantized Llama3 IR linear weights and copy dense tensors."""
        if self.weight_mode != "quantized":
            raise ValueError(
                "load_quantized_ir_weights requires weight_mode='quantized'"
            )
        if ir.architecture != "llama3":
            raise ValueError(f"expected llama3 IR, got {ir.architecture!r}")
        with torch.no_grad():
            embedding_weight = self._optional_dense_weight(
                ir.weights, "token_embedding.weight"
            )
            if embedding_weight is None:
                embedding_weight = self._dense_weight(ir.weights, "lm_head.weight")
            self._copy_param(self.tok_emb.weight, embedding_weight)

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Llama3TransformerBlock, module)
                prefix = f"layers.{layer_idx}"

                self._load_linear_weight(
                    block.att,
                    "W_query",
                    self._weight(ir.weights, f"{prefix}.attention.q_proj.weight"),
                )
                self._load_linear_weight(
                    block.att,
                    "W_key",
                    self._weight(ir.weights, f"{prefix}.attention.k_proj.weight"),
                )
                self._load_linear_weight(
                    block.att,
                    "W_value",
                    self._weight(ir.weights, f"{prefix}.attention.v_proj.weight"),
                )
                self._load_linear_weight(
                    block.att,
                    "out_proj",
                    self._weight(ir.weights, f"{prefix}.attention.o_proj.weight"),
                )

                self._copy_param(
                    block.norm1.weight,
                    self._dense_weight(ir.weights, f"{prefix}.input_norm.weight"),
                )
                self._copy_param(
                    block.norm2.weight,
                    self._dense_weight(
                        ir.weights, f"{prefix}.post_attention_norm.weight"
                    ),
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
                self.final_norm.weight,
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
        if param is None:
            raise ValueError("cannot copy into missing parameter")
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch for parameter: expected {tuple(param.shape)}, "
                f"got {tuple(value.shape)}"
            )
        param.copy_(value)

    @staticmethod
    def _optional_weight(weights: dict[str, Any], name: str) -> torch.Tensor | None:
        return Llama3Model._optional_dense_weight(weights, name)

    @staticmethod
    def _optional_dense_weight(
        weights: dict[str, Any], name: str
    ) -> torch.Tensor | None:
        """
        Try to load a weight named "name".
        If not return NULL.
        If the weight exist and is quantized raise error.
        If exist and is dense return it.
        """
        for candidate in Llama3Model._weight_names(name):
            value = weights.get(candidate)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, QuantizedWeight):
                raise TypeError(
                    f"weight {candidate!r} is quantized; use load_quantized_ir_weights"
                )
        return None

    @staticmethod
    def _weight(weights: dict[str, Any], name: str) -> torch.Tensor | QuantizedWeight:
        """Return a weight by this model's canonical name or known HF alias."""
        for candidate in Llama3Model._weight_names(name):
            if candidate in weights:
                return weights[candidate]
        names = ", ".join(Llama3Model._weight_names(name))
        raise KeyError(f"missing Llama weight {name!r}; tried {names}")

    @staticmethod
    def _dense_weight(weights: dict[str, Any], name: str) -> torch.Tensor:
        value = Llama3Model._weight(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"weight {name!r} is not a dense tensor")
        return value

    @staticmethod
    def _load_linear_weight(
        parent: nn.Module, attr: str, value: torch.Tensor | QuantizedWeight
    ) -> None:
        """
        parent: nn.Module based class that we will load with weight.
        attr: the attr name of the nn.Module class that will be loaded.
        """
        module = getattr(parent, attr)
        if not isinstance(module, (nn.Linear, QuantizedLinear)):
            raise TypeError(f"{attr!r} is not a linear module")
        in_features = int(module.in_features)
        out_features = int(module.out_features)

        if isinstance(value, QuantizedWeight):
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
        Llama3Model._copy_param(module.weight, value)

    @staticmethod
    def _weight_names(name: str) -> list[str]:
        return [name]


class Llama3Tokenizer:
    """
    Thin wrapper around tiktoken that keeps track of Llama-3 special IDs.
    Need to be init with a tokenizer file.
    """

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        self.hf_tokenizer: Any | None = None
        self.tiktok: tiktoken.Encoding | None = None

        self.special = self._llama3_special_tokens()

        try:
            mergeable = load_tiktoken_bpe(model_path)
        except ValueError:
            tokenizer_json_path = Path(model_path).with_name("tokenizer.json")
            if not tokenizer_json_path.is_file():
                raise

            tokenizers = cast(Any, import_module("tokenizers"))
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_json_path))
        else:
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

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "Llama3Tokenizer":
        """
        Build a Llama 3 tokenizer from tokenizer data embedded in a GGUF file.

        Many GGUF files store a full Hugging Face ``tokenizer.json`` payload in
        metadata.  When present, that is the most faithful source because it
        preserves normal tokens, special tokens, and decoder behavior.
        """
        logger.info("Load tokenizer info from file %s" % gguf_path)
        tokenizer = cls.__new__(cls)
        tokenizer.hf_tokenizer = None
        tokenizer.tiktok = None
        tokenizer.special = cls._llama3_special_tokens()

        from .gguf import tokenizer_json_from_gguf, tokenizer_mergeable_ranks_from_gguf

        tokenizer_json = tokenizer_json_from_gguf(gguf_path)
        if tokenizer_json is not None:
            tokenizers = cast(Any, import_module("tokenizers"))
            tokenizer.hf_tokenizer = tokenizers.Tokenizer.from_str(tokenizer_json)
            return tokenizer

        try:
            mergeable = tokenizer_mergeable_ranks_from_gguf(gguf_path)
        except ValueError:
            mergeable = None
        if mergeable is not None:
            tokenizer.tiktok = tiktoken.Encoding(
                name=Path(gguf_path).name,
                pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                r"|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+"
                r"|\s+(?!\S)"
                r"|\s+",
                mergeable_ranks=mergeable,
                special_tokens=tokenizer.special,
            )
            return tokenizer

        gguf_file = Path(gguf_path)
        for sidecar_name in ("tokenizer.model", "tokenizer.json"):
            sidecar = gguf_file.with_name(sidecar_name)
            if sidecar.is_file():
                return cls(str(sidecar))

        raise ValueError(
            f"{gguf_path} does not contain tokenizer.huggingface.json and no "
            "tokenizer.model/tokenizer.json sidecar was found"
        )

    @staticmethod
    def _llama3_special_tokens() -> dict[str, int]:
        """Return Meta's fixed Llama 3 special-token id map."""
        special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        special.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if 128002 + i not in special.values()
            }
        )
        return special

    def get_eos(self) -> int | None:
        return self.special["<|eot_id|>"]

    def encode(self, input: str, bos: bool = False, eos: bool = False) -> list[int]:
        if self.hf_tokenizer is not None:
            ids = cast(list[int], self.hf_tokenizer.encode(input).ids)
        elif self.tiktok is not None:
            ids = self.tiktok.encode(input)
        else:
            raise RuntimeError("Llama3Tokenizer is not initialized")

        ids = ([self.special["<|begin_of_text|>"]] if bos else []) + ids
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def encode_instruct_prompt(self, user_text: str) -> list[int]:
        """
        Encode a minimal Llama 3 user -> assistant chat prompt.

        Instruct checkpoints are trained on this wire format.  We insert special
        token ids directly instead of writing special-token strings into the
        text, which avoids tokenizer-specific escaping behavior.
        """
        ids = [self.special["<|begin_of_text|>"], self.special["<|start_header_id|>"]]
        ids += self.encode("user")
        ids += [self.special["<|end_header_id|>"]]
        ids += self.encode("\n\n" + user_text)
        ids += [self.special["<|eot_id|>"], self.special["<|start_header_id|>"]]
        ids += self.encode("assistant")
        ids += [self.special["<|end_header_id|>"]]
        ids += self.encode("\n\n")
        return ids

    def decode(self, tok: list[int]) -> str:
        if self.hf_tokenizer is not None:
            return cast(str, self.hf_tokenizer.decode(tok, skip_special_tokens=False))
        if self.tiktok is not None:
            return self.tiktok.decode(tok)
        raise RuntimeError("Llama3Tokenizer is not initialized")
