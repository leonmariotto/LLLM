"""
GPT2 (Generative Pretrained Tranformer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, TypeAlias, Dict, Optional
from typing import Any, TypedDict, Literal

import torch
from torch import nn

import tiktoken
from tiktoken.core import Encoding
from tiktoken_ext.openai_public import gpt2 as gpt2_tiktoken_base_args

from .norm import LayerNorm
from .rope import apply_rope, precompute_rope_cache

if TYPE_CHECKING:
    from .fetch import FetchedModel

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


def gpt2_config_from_fetched(config: dict[str, Any]) -> GPT2Config:
    """Translate a Hugging Face GPT2 config into LLLM GPT2Config.
    GPT2_CONFIG_124M: GPT2Config = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
        "positional_encoding": "gpt2",
    }

    GPT2_CONFIG_355M: GPT2Config = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 1024,  # Embedding dimension
        "n_heads": 16,  # Number of attention heads
        "n_layers": 24,  # Number of layers
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
        "positional_encoding": "gpt2",
    }
    """

    def _int_config(
        config: dict[str, Any], key: str, *, fallback_key: str | None = None
    ) -> int:
        value = config.get(key)
        if value is None and fallback_key is not None:
            value = config.get(fallback_key)
        if not isinstance(value, int):
            raise ValueError(f"config value {key!r} must be an int")
        return value

    return {
        "vocab_size": _int_config(config, "vocab_size"),
        "context_length": _int_config(config, "n_ctx", fallback_key="n_positions"),
        "emb_dim": _int_config(config, "n_embd"),
        "n_heads": _int_config(config, "n_head"),
        "n_layers": _int_config(config, "n_layer"),
        "drop_rate": float(config.get("resid_pdrop", 0.0)),
        "qkv_bias": True,  # Always use bias.
        "positional_encoding": "gpt2",  # No RoPE in GPT2.
    }


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

    def load_fetched_model(self, fetched: FetchedModel) -> None:
        """
        Copy Hugging Face GPT2 tensors into this model.

        Hugging Face GPT2 stores linear projection weights in Conv1D layout
        ``[input_dim, output_dim]``. PyTorch Linear expects
        ``[output_dim, input_dim]``, so projection weights are transposed while
        biases are copied as-is.
        """
        if fetched.model_type != "gpt2":
            raise NotImplementedError(f"unsupported model_type: {fetched.model_type}")
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight, self._weight(fetched.weights, "wte.weight")
            )
            if self.pos_emb is None:
                raise ValueError(
                    "GPT2 fetched.weights require absolute position embeddings"
                )
            self._copy_param(
                self.pos_emb.weight, self._weight(fetched.weights, "wpe.weight")
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(GPT2TransformerBlock, module)
                prefix = f"h.{layer_idx}"

                q_weight, k_weight, v_weight = self._weight(
                    fetched.weights, f"{prefix}.attn.c_attn.weight"
                ).chunk(3, dim=1)
                q_bias, k_bias, v_bias = self._weight(
                    fetched.weights, f"{prefix}.attn.c_attn.bias"
                ).chunk(3, dim=0)

                self._copy_param(block.att.W_query.weight, q_weight.T)
                self._copy_param(block.att.W_key.weight, k_weight.T)
                self._copy_param(block.att.W_value.weight, v_weight.T)
                self._copy_param(block.att.W_query.bias, q_bias)
                self._copy_param(block.att.W_key.bias, k_bias)
                self._copy_param(block.att.W_value.bias, v_bias)

                self._copy_param(
                    block.att.out_proj.weight,
                    self._weight(fetched.weights, f"{prefix}.attn.c_proj.weight").T,
                )
                self._copy_param(
                    block.att.out_proj.bias,
                    self._weight(fetched.weights, f"{prefix}.attn.c_proj.bias"),
                )

                self._copy_param(
                    block.norm1.scale,
                    self._weight(fetched.weights, f"{prefix}.ln_1.weight"),
                )
                self._copy_param(
                    block.norm1.shift,
                    self._weight(fetched.weights, f"{prefix}.ln_1.bias"),
                )
                self._copy_param(
                    block.norm2.scale,
                    self._weight(fetched.weights, f"{prefix}.ln_2.weight"),
                )
                self._copy_param(
                    block.norm2.shift,
                    self._weight(fetched.weights, f"{prefix}.ln_2.bias"),
                )

                fc = cast(nn.Linear, block.ff.layers[0])
                proj = cast(nn.Linear, block.ff.layers[2])
                self._copy_param(
                    fc.weight,
                    self._weight(fetched.weights, f"{prefix}.mlp.c_fc.weight").T,
                )
                self._copy_param(
                    fc.bias, self._weight(fetched.weights, f"{prefix}.mlp.c_fc.bias")
                )
                self._copy_param(
                    proj.weight,
                    self._weight(fetched.weights, f"{prefix}.mlp.c_proj.weight").T,
                )
                self._copy_param(
                    proj.bias,
                    self._weight(fetched.weights, f"{prefix}.mlp.c_proj.bias"),
                )

            self._copy_param(
                self.final_norm.scale, self._weight(fetched.weights, "ln_f.weight")
            )
            self._copy_param(
                self.final_norm.shift, self._weight(fetched.weights, "ln_f.bias")
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
    def _lm_head_weight(weights: dict[str, torch.Tensor]) -> torch.Tensor:
        if "lm_head.weight" in weights:
            return weights["lm_head.weight"]
        return GPT2Model._weight(weights, "wte.weight")

    @staticmethod
    def _weight(weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name in weights:
            return weights[name]
        prefixed_name = f"transformer.{name}"
        if prefixed_name in weights:
            return weights[prefixed_name]
        raise KeyError(f"missing GPT2 weight {name!r} or {prefixed_name!r}")

    def forward(self, in_idx: torch.Tensor, pos: int | None = None) -> torch.Tensor:
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
        logits = self.out_head(x)
        return logits


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
        # Surface encoder capacity directly from tiktoken for quick inspection.
        return self.tiktok.n_vocab

    @property
    def eos_token_id(self) -> TokenId:
        # Keep the canonical special token id accessible to callers and tests.
        return self.tiktok.eot_token

    @property
    def special_tokens(self) -> set[str]:
        return set(self.tiktok.special_tokens_set)

    def encode(self, in_str: str) -> list[TokenId]:
        return self.tiktok.encode(in_str, allowed_special="all")
        # TODO should return tensor here ??
        # encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        # return encoded_tensor

    def decode(self, in_tok: list[TokenId]) -> str:
        return self.tiktok.decode(in_tok)

    def token_count(self, in_str: str) -> int:
        return len(self.encode(in_str))


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

        # NOTE on tensor.view and tensor.transpose methods.
        # Tensor view method reshape a tensor, without moving elements in memory.
        # Whereas transpose change how dimensions are indexed.
        # So, for example :
        #     tensor([[0, 1, 2],
        #         [3, 4, 5]])
        # y.view(3,2)
        #     tensor([[0, 1],
        #         [2, 3],
        #         [4, 5]])
        # y.transpose(0, 1)
        #     tensor([[0, 3],
        #             [1, 4],
        #             [2, 5]])

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
            # TODO somewhat messy, clean it up.
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

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


# TODO use torch GeLu and remove this, or move some in comments.
class GELU(nn.Module):
    """
    Implement the GeLu activation function approximation (computationally cheaper).
    An optimized version is present torch.nn.functional.gelu but keep it here
    for illustration purpose.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class GPT2FeedForward(nn.Module):
    """
    FeedForward: expansion -> activation (GeLu) -> contraction.
    """

    def __init__(self, embedded_dimension: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedded_dimension, expansion_factor * embedded_dimension),
            GELU(),
            nn.Linear(expansion_factor * embedded_dimension, embedded_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
        x = x + shortcut  # Add the original input back

        return x
