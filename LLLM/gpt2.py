"""GPT2 (Generative Pretrained Tranformer)"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, TypeAlias, Dict, Optional
from typing import TypedDict, Literal

import torch
from torch import nn

import tiktoken
from tiktoken.core import Encoding
from tiktoken_ext.openai_public import gpt2 as gpt2_tiktoken_base_args

from .norm import LayerNorm
from .rope import apply_rope, precompute_rope_cache
from .generator import Generator

if TYPE_CHECKING:
    from .model_ir import ModelIR, ModelWeightsIR

PositionalEncoding = Literal["gpt2", "rope"]


class GPT2Config(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    positional_encoding: PositionalEncoding


class GPT2Model(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Initialization is done with a config dict.

    - Input: batch of tokenized sentence.
    - Output: a forward pass produce logits for each positions in each batch
    Logits are a dictionary-wide scores vector. With softmax function applied on it it
    become a probability distribution of the token.
    For input tokens [x1, x2, x3] it predict x2 from x1, x3 for x1 and x2, and finally
    x4 from all.
    The prediction is applied of all token in the input for training purpose.
    At inference only the next token matter.
    """

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.positional_encoding = cfg["positional_encoding"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb: nn.Embedding | None = None
        if self.positional_encoding == "gpt2":
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[GPT2TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    @staticmethod
    def config_from_ir(ir: ModelIR) -> GPT2Config:
        """
        Build a model configuration from normalized IR.

        Args:
            ir: Normalized model IR that supplies configuration
                and canonical weights.

        Returns:
            Configuration dictionary accepted by the model constructor.
        """
        if ir.architecture != "gpt2":
            raise ValueError(f"expected gpt2 IR, got {ir.architecture!r}")
        positional_encoding = ir.config.get("positional_encoding", "gpt2")
        if positional_encoding not in {"gpt2", "rope"}:
            raise ValueError(
                "IR config field 'positional_encoding' must be gpt2 or rope"
            )
        return {
            "vocab_size": ir.config.require_int("vocab_size"),
            "context_length": ir.config.require_int("context_length"),
            "emb_dim": ir.config.require_int("hidden_size"),
            "n_heads": ir.config.require_int("num_attention_heads"),
            "n_layers": ir.config.require_int("num_hidden_layers"),
            "drop_rate": float(ir.config.get("dropout", 0.0)),
            "qkv_bias": bool(ir.config.get("qkv_bias", True)),
            "positional_encoding": cast(PositionalEncoding, positional_encoding),
        }

    def load_ir_weights(self, ir: ModelIR) -> None:
        """
        Copy canonical GPT2 IR tensors into this model.

        GPT2 stores linear projection weights in Conv1D layout
        ``[input_dim, output_dim]``. PyTorch Linear expects
        ``[output_dim, input_dim]``, so projection weights are transposed while
        biases are copied as-is.
        """
        if ir.architecture != "gpt2":
            raise ValueError(f"expected gpt2 IR, got {ir.architecture!r}")
        weights = ir.weights
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight, self._weight(weights, "token_embedding.weight")
            )
            if self.pos_emb is None:
                raise ValueError("GPT2 IR weights require absolute position embeddings")
            self._copy_param(
                self.pos_emb.weight, self._weight(weights, "position_embedding.weight")
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(GPT2TransformerBlock, module)
                prefix = f"layers.{layer_idx}"

                q_weight, k_weight, v_weight = self._weight(
                    weights, f"{prefix}.attention.qkv_proj.weight"
                ).chunk(3, dim=1)
                q_bias, k_bias, v_bias = self._weight(
                    weights, f"{prefix}.attention.qkv_proj.bias"
                ).chunk(3, dim=0)

                self._copy_param(block.att.W_query.weight, q_weight.T)
                self._copy_param(block.att.W_key.weight, k_weight.T)
                self._copy_param(block.att.W_value.weight, v_weight.T)
                self._copy_param(block.att.W_query.bias, q_bias)
                self._copy_param(block.att.W_key.bias, k_bias)
                self._copy_param(block.att.W_value.bias, v_bias)

                self._copy_param(
                    block.att.out_proj.weight,
                    self._weight(weights, f"{prefix}.attention.o_proj.weight").T,
                )
                self._copy_param(
                    block.att.out_proj.bias,
                    self._weight(weights, f"{prefix}.attention.o_proj.bias"),
                )

                self._copy_param(
                    block.norm1.scale,
                    self._weight(weights, f"{prefix}.input_norm.weight"),
                )
                self._copy_param(
                    block.norm1.shift,
                    self._weight(weights, f"{prefix}.input_norm.bias"),
                )
                self._copy_param(
                    block.norm2.scale,
                    self._weight(weights, f"{prefix}.post_attention_norm.weight"),
                )
                self._copy_param(
                    block.norm2.shift,
                    self._weight(weights, f"{prefix}.post_attention_norm.bias"),
                )

                self._copy_param(
                    block.ff.fc1.weight,
                    self._weight(weights, f"{prefix}.feed_forward.up_proj.weight").T,
                )
                self._copy_param(
                    block.ff.fc1.bias,
                    self._weight(weights, f"{prefix}.feed_forward.up_proj.bias"),
                )
                self._copy_param(
                    block.ff.fc2.weight,
                    self._weight(weights, f"{prefix}.feed_forward.down_proj.weight").T,
                )
                self._copy_param(
                    block.ff.fc2.bias,
                    self._weight(weights, f"{prefix}.feed_forward.down_proj.bias"),
                )

            self._copy_param(
                self.final_norm.scale, self._weight(weights, "final_norm.weight")
            )
            self._copy_param(
                self.final_norm.shift, self._weight(weights, "final_norm.bias")
            )
            self._copy_param(self.out_head.weight, self._lm_head_weight(weights))

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
    def _lm_head_weight(weights: ModelWeightsIR) -> torch.Tensor:
        if "lm_head.weight" in weights:
            value = weights["lm_head.weight"]
            if isinstance(value, torch.Tensor):
                return value
        return GPT2Model._weight(weights, "token_embedding.weight")

    @staticmethod
    def _weight(weights: ModelWeightsIR, name: str) -> torch.Tensor:
        if name in weights:
            value = weights[name]
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(f"GPT2 weight {name!r} is quantized")
        raise KeyError(f"missing GPT2 IR weight {name!r}")

    def forward(self, in_idx: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        """
        Args:
            in_idx: Token ids with shape ``[batch, tokens]``.
            pos: Optional starting token position used for RoPE.

        Returns:
            Logits with shape ``[batch, tokens, vocab_size]``.
        """
        _, seq_len = in_idx.shape  # batch_size, seq_len
        tok_embeds = self.tok_emb(in_idx)
        if self.positional_encoding == "gpt2":
            assert self.pos_emb is not None
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
            pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
            x = tok_embeds + pos_embeds
        else:
            x = tok_embeds
        x = self.drop_emb(x)
        # x is the hidden state between transformers block. Output of transformer 1
        # is the input of transformer 2 and so on.
        for blk in self.trf_blocks:
            x = blk(x, pos)
        x = self.final_norm(x)
        return self.out_head(x)


TokenId: TypeAlias = int


class GPT2Tokenizer:
    def __init__(
        self, name: str = "custom", extra_special_tokens: Dict[str, int] = {}
    ) -> None:
        base_args = gpt2_tiktoken_base_args()
        base_special_tokens: dict[str, int] = cast(
            dict[str, int], base_args.get("special_tokens")
        )
        base_mergeable_ranks: dict[bytes, int] = cast(
            dict[bytes, int], base_args.get("mergeable_ranks")
        )
        special_tokens: Dict[str, int] = base_special_tokens | extra_special_tokens
        self.tiktok: Encoding = tiktoken.Encoding(
            name=f"gpt2_{name}",
            pat_str=str(base_args["pat_str"]),
            mergeable_ranks=base_mergeable_ranks,
            special_tokens={
                **special_tokens,
            },
        )

    @property
    def vocabulary_size(self) -> int:
        """Surface encoder capacity directly from tiktoken for quick inspection."""
        return self.tiktok.n_vocab

    @property
    def eos_token_id(self) -> TokenId:
        """Keep the canonical special token id accessible to callers and tests."""
        return self.tiktok.eot_token

    @property
    def special_tokens(self) -> set[str]:
        """Return list of special token of the tokenizer."""
        return set(self.tiktok.special_tokens_set)

    def encode(self, in_str: str) -> list[TokenId]:
        """
        Encode text into token ids.

        Args:
            in_str: Input string to encode.

        Returns:
            Encoded token ids.
        """
        return self.tiktok.encode(in_str, allowed_special="all")

    def decode(self, in_tok: list[TokenId]) -> str:
        """
        Decode token ids into text.

        Args:
            in_tok: Token ids to decode.

        Returns:
            Decoded text.
        """
        return self.tiktok.decode(in_tok)

    def token_count(self, in_str: str) -> int:
        """
        Implement token count behavior.

        Args:
            in_str: Input string to encode.
        """
        return len(self.encode(in_str))


class GeneratorGPT2(Generator):
    """GPT-2 text generator using the legacy full-context, cache-less path."""

    def _generate_tokens(
        self,
        input_tokens: list[int],
        *,
        stop_at_eos: bool,
        max_generated_token: int,
        eos: int | None,
        context_size: int,
        temperature: float,
        top_k: int | None,
    ) -> tuple[list[int], int]:
        self.model.eval()
        idx = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._model_device(),
        )
        generated_token_count = 0

        for _ in range(max_generated_token):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :]
            logits = self._filter_logits(logits, top_k)
            idx_next = self._select_next_token(logits, temperature)
            if stop_at_eos and eos is not None and bool((idx_next == eos).all().item()):
                break
            idx = torch.cat((idx, idx_next), dim=1)
            generated_token_count += int(idx_next.shape[0])

        return cast(
            list[int], cast(Any, idx.squeeze(0)).tolist()
        ), generated_token_count


class GPT2MultiHeadAttention(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Attention mechanism involve 3 trainable matrix : query, key, values.

    Implement Causal mask and dropout.

    Multi-headed attention: d_out is splited in num_head parts. Each head
    produce a part of d_out (head_dim, calculated at init), and at the end
    context_vec is reshaped to the correct size.
    So things can be parallel.

    There is a optional RoPe but for GPT2 it should not be used. Disabled by default.
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
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_rope: bool = False,
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
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        if self.use_rope:
            cos, sin = precompute_rope_cache(
                seq_len=self.context_length,
                head_dim=self.head_dim,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, d_in]``.
            pos: Optional starting token position used for RoPE.

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

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys, values = keys_new, values_new

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.use_rope:
            # Add RoPE after Q/K projection and head reshaping and before computing
            # attention scores.
            if pos is None:
                position_ids = torch.arange(num_tokens, device=x.device)
            else:
                position_ids = torch.arange(
                    pos,
                    pos + num_tokens,
                    device=x.device,
                )
            assert int(position_ids[-1]) < self.context_length, (
                "RoPE position exceeds precomputed context length"
            )
            cos = self.rope_cos[position_ids]
            sin = self.rope_sin[position_ids]
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # `queries` has shape (batch, num_heads, num_tokens, head_dim).
        # So shape[-2] is the query-token dimension.
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        # Original mask truncated to the number of tokens and converted to boolean.
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)  # optional projection


class GPT2FeedForward(nn.Module):
    """FeedForward: expansion -> activation (GeLu) -> contraction."""

    def __init__(self, embedded_dimension: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedded_dimension, expansion_factor * embedded_dimension)
        self.fc2 = nn.Linear(expansion_factor * embedded_dimension, embedded_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states with shape ``[..., emb_dim]``.

        Returns:
            Hidden states with shape ``[..., emb_dim]``.
        """
        x = self.fc1(x)
        x = nn.functional.gelu(x, approximate="tanh")
        return self.fc2(x)


class GPT2TransformerBlock(nn.Module):
    """
    Transformer implementation.

    Transformer use the following components :
    - MultiHeadAttention: attention mechanism, implemented in attention.py
    - FeedForward: a small nn that do an expansion (* 4), an activation function pass
    (GeLu, not ReLu), and a contraction back to the original input dimension.
    - LayerNorm: normalization is used to improve model math efficiency, without it
    the model will struggle to find weights that minimize its loss function due to problems
    like vanishing or exploding gradients. Normalization layer adjust the output of a nn to
    have a mean of 0 and a variance of 1 (== "unit variance").
    - Shortcuts: another technics used to improve training. It mitigate the problem of
    vanishing gradient, when they become prgressivly smaller as they propagate through
    layers, making it difficult to train earlier weights (first layers got gradients
    too small). So we create shortcut connection that add the N layer input to N+1 layer
    input (so N+1 input become N output + N input).
    """

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.att = GPT2MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_rope=cfg["positional_encoding"] == "rope",
        )
        self.ff = GPT2FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut: nn.Module = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        # Shortcut connection for attention block
        """
        Args:
            x: Hidden states with shape ``[batch, tokens, emb_dim]``.
            pos: Optional starting token position used for RoPE
                when the block is configured with RoPE.

        Returns:
            Hidden states with shape ``[batch, tokens, emb_dim]``.
        """
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, pos)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut  # Add the original input back
