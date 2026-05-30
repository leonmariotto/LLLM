"""
Qwen3 dense text decoder.

Implements the non-MoE Qwen3 decoder; Qwen3-MoE expert routing is
intentionally out of scope.
Recommended sampling parameters:
Thinking: temperature=0.6, top_p=0.95, top_k=20
Non-thinking (instruct): tasks: temperature=0.7, top_p=0.80, top_k=20
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast

import torch
import torch.nn as nn

from .kv_cache import KVCache
from .llama2 import Llama2FeedForward
from .norm import RMSNorm
from .qwen2 import Qwen2Tokenizer
from .quantization import QuantizedLinear, QuantizedWeight, WeightMode
from .rope import apply_rope, precompute_rope_cache

if TYPE_CHECKING:
    from .model_ir import ModelIR


class Qwen3Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_kv_groups: int
    n_layers: int
    hidden_dim: int
    head_dim: int
    rope_theta: float
    rope_interleaved: bool
    rms_norm_eps: float
    attention_bias: bool
    dtype: torch.dtype


class Qwen3GroupedQueryAttention(nn.Module):
    """Qwen3 GQA with per-head Q/K RMSNorm before RoPE."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        head_dim: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        rope_interleaved: bool = False,
        rms_norm_eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        assert num_heads != 0, "num_head shall not be 0"
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups."
        )

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_groups * head_dim
        self.context_length = context_length
        self.num_kv_groups = num_kv_groups
        self.kv_group_size = num_heads // num_kv_groups
        self.rope_interleaved = rope_interleaved
        self.scaling = self.head_dim**-0.5

        self.W_query = nn.Linear(d_in, self.q_size, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.kv_size, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.kv_size, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(self.q_size, d_out, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
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
        b, num_tokens, d_in = x.shape
        assert self.d_in == d_in, "invalid d_in (embedding size)"

        queries = self.W_query(x)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        queries = queries.transpose(1, 2)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)

        queries = self.q_norm(queries)
        keys_new = self.k_norm(keys_new)

        next_pos = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            next_pos = kv_cache.layer_next_pos(layer_idx)

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

        keys = keys.repeat_interleave(self.kv_group_size, dim=1)
        values = values.repeat_interleave(self.kv_group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        num_tokens_q = queries.shape[-2]
        num_tokens_k = keys.shape[-2]
        key_positions = torch.arange(
            key_start_pos,
            key_start_pos + num_tokens_k,
            device=x.device,
        )
        query_positions = torch.arange(
            start_pos,
            start_pos + num_tokens_q,
            device=x.device,
        )
        mask_bool = key_positions[None, :] > query_positions[:, None]

        attn_scores = attn_scores * self.scaling
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.q_size)
        return self.out_proj(context_vec)


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.att = Qwen3GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            head_dim=cfg["head_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            qkv_bias=cfg["attention_bias"],
            rope_interleaved=cfg["rope_interleaved"],
            rms_norm_eps=cfg["rms_norm_eps"],
            dtype=cfg["dtype"],
        )
        self.ff = Llama2FeedForward(cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"])
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=cfg["rms_norm_eps"])
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=cfg["rms_norm_eps"])

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
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin, pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + shortcut


class Qwen3Model(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: Qwen3Config, weight_mode: WeightMode = "dense"):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.weight_mode = weight_mode
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.trf_blocks = nn.ModuleList(
            Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        )
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg["rms_norm_eps"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

        cos, sin = precompute_rope_cache(
            head_dim=cfg["head_dim"],
            base=cfg["rope_theta"],
            seq_len=cfg["context_length"],
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def config_from_ir(ir: ModelIR) -> Qwen3Config:
        if ir.architecture != "qwen3":
            raise ValueError(f"expected qwen3 IR, got {ir.architecture!r}")

        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_kv_groups": ir.config.require_int("num_key_value_heads"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "hidden_dim": ir.config.require_int("intermediate_size"),
            "head_dim": ir.config.require_int("head_dim"),
            "rope_theta": ir.config.require_float("rope_theta"),
            "rope_interleaved": bool(ir.config.get("rope_interleaved", False)),
            "rms_norm_eps": ir.config.require_float("rms_norm_eps"),
            "attention_bias": bool(ir.config.get("attention_bias", False)),
            "dtype": torch.float32,
        }

    def forward(
        self,
        in_idx: torch.Tensor,
        pos: int | None = None,
        *,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = self.tok_emb(in_idx)
        for layer_idx, module in enumerate(self.trf_blocks):
            block = cast(Qwen3TransformerBlock, module)
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
        if ir.architecture != "qwen3":
            raise ValueError(f"expected qwen3 IR, got {ir.architecture!r}")
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight,
                self._dense_weight(ir.weights, "token_embedding.weight"),
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Qwen3TransformerBlock, module)
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
                self._copy_param(
                    block.att.q_norm.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.q_norm.weight"),
                )
                self._copy_param(
                    block.att.k_norm.weight,
                    self._dense_weight(ir.weights, f"{prefix}.attention.k_norm.weight"),
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

    def load_quantized_ir_weights(self, ir: ModelIR) -> None:
        if self.weight_mode != "quantized":
            raise ValueError(
                "load_quantized_ir_weights requires weight_mode='quantized'"
            )
        self.load_ir_weights(ir)

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
            raise KeyError("missing Qwen3 bias weight")
        Qwen3Model._copy_param(param, value)

    @staticmethod
    def _optional_weight(weights: dict[str, Any], name: str) -> torch.Tensor | None:
        value = weights.get(name)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, QuantizedWeight):
            raise TypeError(f"Qwen3 weight {name!r} is quantized")
        return None

    @staticmethod
    def _weight(weights: dict[str, Any], name: str) -> torch.Tensor | QuantizedWeight:
        value = weights.get(name)
        if isinstance(value, (torch.Tensor, QuantizedWeight)):
            return value
        raise KeyError(f"missing Qwen3 weight {name!r}")

    @staticmethod
    def _dense_weight(weights: dict[str, Any], name: str) -> torch.Tensor:
        value = Qwen3Model._weight(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Qwen3 weight {name!r} is not a dense tensor")
        return value

    def _load_linear_weight(
        self, parent: nn.Module, attr: str, value: torch.Tensor | QuantizedWeight
    ) -> None:
        module = getattr(parent, attr)
        if not isinstance(module, (nn.Linear, QuantizedLinear)):
            raise TypeError(f"{attr!r} is not a linear module")
        in_features = int(module.in_features)
        out_features = int(module.out_features)

        if isinstance(value, QuantizedWeight):
            if self.weight_mode != "quantized":
                raise TypeError(
                    f"Qwen3 weight for {attr!r} is quantized but model is dense"
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


class Qwen3Tokenizer(Qwen2Tokenizer):
    """Qwen3 tokenizer wrapper; Qwen3 uses the same ChatML token family as Qwen2."""
