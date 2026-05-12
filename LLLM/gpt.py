"""
Generative Pretrained Tranformer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, TypeAlias

import torch
from torch import nn

from .transformer import LayerNorm
from .transformer import TransformerBlock

import tiktoken
from tiktoken.core import Encoding

if TYPE_CHECKING:
    from .fetch import FetchedModel


PositionalEncoding = Literal["gpt2", "rope"]


class GPTConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    positional_encoding: PositionalEncoding


# TODO becasue we fetch models, and parameters are present in models, we don't need
# the following dict. This can be removed.
GPT_CONFIG_124M: GPTConfig = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
    "positional_encoding": "gpt2",
}

GPT_CONFIG_355M: GPTConfig = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,  # Embedding dimension
    "n_heads": 16,  # Number of attention heads
    "n_layers": 24,  # Number of layers
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
    "positional_encoding": "gpt2",
}


class GPTModel(nn.Module):
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

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.positional_encoding = cfg["positional_encoding"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb: nn.Embedding | None = None
        if self.positional_encoding == "gpt2":
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def load_fetched_model(self, fetched: FetchedModel) -> None:
        """
        Copy Hugging Face GPT-2 tensors into this model.

        Hugging Face GPT-2 stores linear projection weights in Conv1D layout
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
                    "GPT-2 fetched.weights require absolute position embeddings"
                )
            self._copy_param(
                self.pos_emb.weight, self._weight(fetched.weights, "wpe.weight")
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(TransformerBlock, module)
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
        return GPTModel._weight(weights, "wte.weight")

    @staticmethod
    def _weight(weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name in weights:
            return weights[name]
        prefixed_name = f"transformer.{name}"
        if prefixed_name in weights:
            return weights[prefixed_name]
        raise KeyError(f"missing GPT-2 weight {name!r} or {prefixed_name!r}")

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
    def __init__(self) -> None:
        self.tiktok: Encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocabulary_size(self) -> int:
        # Surface encoder capacity directly from tiktoken for quick inspection.
        return self.tiktok.n_vocab

    @property
    def end_of_text_token(self) -> TokenId:
        # Keep the canonical special token id accessible to callers and tests.
        return self.tiktok.eot_token

    @property
    def special_tokens(self) -> set[str]:
        return set(self.tiktok.special_tokens_set)

    def encode(self, in_str: str) -> list[TokenId]:
        return self.tiktok.encode(in_str, allowed_special={"<|endoftext|>"})
        # TODO should return tensor here ??
        # encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        # return encoded_tensor

    def decode(self, in_tok: list[TokenId]) -> str:
        return self.tiktok.decode(in_tok)

    def token_count(self, in_str: str) -> int:
        return len(self.encode(in_str))


def gpt_config_from_fetched(config: dict[str, Any]) -> GPTConfig:
    """Translate a Hugging Face GPT-2 config into LLLM GPTConfig."""

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
        "qkv_bias": True,
        "positional_encoding": "gpt2",  # No RoPE
    }
