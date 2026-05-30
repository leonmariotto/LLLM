"""
Qwen2 / Qwen2.5 text decoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, cast

import tiktoken
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .kv_cache import KVCache
from .llama2 import Llama2FeedForward
from .norm import RMSNorm
from .quantization import QuantizedLinear, QuantizedWeight, WeightMode
from .rope import apply_rope, precompute_rope_cache
from .generator_with_tool import AssistantOutput, ToolCall

if TYPE_CHECKING:
    from .model_ir import ModelIR


class Qwen2Config(TypedDict):
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


class Qwen2GroupedQueryAttention(nn.Module):
    """
    Qwen2 causal grouped-query self-attention.

    Qwen2 uses separate Q/K/V projections with optional biases, a bias-less output
    projection, split-half RoPE in Hugging Face checkpoints, and GQA K/V head
    repetition.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        head_dim: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        rope_interleaved: bool = False,
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


class Qwen2TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.att = Qwen2GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            head_dim=cfg["head_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            qkv_bias=cfg["attention_bias"],
            rope_interleaved=cfg["rope_interleaved"],
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


class Qwen2Model(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: Qwen2Config, weight_mode: WeightMode = "dense"):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.weight_mode = weight_mode
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.trf_blocks = nn.ModuleList(
            Qwen2TransformerBlock(cfg) for _ in range(cfg["n_layers"])
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
    def config_from_ir(ir: ModelIR) -> Qwen2Config:
        if ir.architecture != "qwen2":
            raise ValueError(f"expected qwen2 IR, got {ir.architecture!r}")

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
            "attention_bias": bool(ir.config.get("attention_bias", True)),
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
            block = cast(Qwen2TransformerBlock, module)
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
        if ir.architecture != "qwen2":
            raise ValueError(f"expected qwen2 IR, got {ir.architecture!r}")
        with torch.no_grad():
            self._copy_param(
                self.tok_emb.weight,
                self._dense_weight(ir.weights, "token_embedding.weight"),
            )

            for layer_idx, module in enumerate(self.trf_blocks):
                block = cast(Qwen2TransformerBlock, module)
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
            raise KeyError("missing Qwen2 bias weight")
        Qwen2Model._copy_param(param, value)

    @staticmethod
    def _optional_weight(weights: dict[str, Any], name: str) -> torch.Tensor | None:
        value = weights.get(name)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, QuantizedWeight):
            raise TypeError(f"Qwen2 weight {name!r} is quantized")
        return None

    @staticmethod
    def _weight(weights: dict[str, Any], name: str) -> torch.Tensor | QuantizedWeight:
        value = weights.get(name)
        if isinstance(value, (torch.Tensor, QuantizedWeight)):
            return value
        raise KeyError(f"missing Qwen2 weight {name!r}")

    @staticmethod
    def _dense_weight(weights: dict[str, Any], name: str) -> torch.Tensor:
        value = Qwen2Model._weight(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Qwen2 weight {name!r} is not a dense tensor")
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
                    f"Qwen2 weight for {attr!r} is quantized but model is dense"
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


class Qwen2Tokenizer:
    """Thin wrapper around Qwen2 Hugging Face tokenizer.json or GGUF tokenizer."""

    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        self.tok: Tokenizer | None = None
        self.tiktok: tiktoken.Encoding | None = None
        self.special: dict[str, int] = {}
        if tok_file.suffix.lower() == ".gguf":
            self.tiktok = self._tiktoken_from_gguf(tok_file)
        else:
            from_file = cast(Callable[[str], Tokenizer], cast(Any, Tokenizer).from_file)
            self.tok = from_file(str(tok_file))
        self.eos_token_id = self.convert_tokens_to_ids("<|im_end|>")
        self.pad_token_id = self.convert_tokens_to_ids("<|endoftext|>")

    def get_eos(self) -> int | None:
        return self.eos_token_id

    @staticmethod
    def _qwen2_pat_str() -> str:
        """Return Qwen2's GPT-2 BPE pre-tokenization regex."""
        return (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            r"|[^\r\n\p{L}\p{N}]?\p{L}+"
            r"|\p{N}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+"
            r"|\s+(?!\S)"
            r"|\s+"
        )

    def _tiktoken_from_gguf(self, gguf_path: Path) -> tiktoken.Encoding:
        """Build a tiktoken tokenizer from Qwen GGUF GPT-2 BPE fields."""
        from gguf import GGUFReader

        from .gguf import find_gguf_file, tokenizer_mergeable_ranks_from_gguf

        gguf_file = find_gguf_file(gguf_path)
        reader = GGUFReader(gguf_file)
        tokens = cast(list[str], reader.fields["tokenizer.ggml.tokens"].contents())
        token_types = cast(
            list[int], reader.fields["tokenizer.ggml.token_type"].contents()
        )
        self.special = {
            token: token_id
            for token_id, (token, token_type) in enumerate(zip(tokens, token_types))
            if token_type != 1
        }
        return tiktoken.Encoding(
            name=gguf_file.name,
            pat_str=self._qwen2_pat_str(),
            mergeable_ranks=tokenizer_mergeable_ranks_from_gguf(gguf_file),
            special_tokens=self.special,
        )

    def encode(self, input: str) -> list[int]:
        if self.tiktok is not None:
            return self.tiktok.encode(input)
        if self.tok is None:
            raise ValueError("Qwen2 tokenizer is not initialized")
        encode = cast(Callable[[str], Any], cast(Any, self.tok).encode)
        encoded = encode(input)
        return cast(list[int], encoded.ids)

    def encode_instruct_prompt(
        self,
        user_text: str,
        *,
        enable_thinking: bool = True,
    ) -> list[int]:
        encoded = self.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not isinstance(encoded, dict):
            raise TypeError("expected tokenized chat template output")
        return encoded["input_ids"]

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> dict[str, list[int]] | str:
        if tools is not None:
            prompt = self._apply_tool_chat_template(messages, tools)
        else:
            prompt = ""
            for message in messages:
                role = self._message_role(message)
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"unsupported Qwen2 chat role {role!r}")
                content = self._message_content(message)
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
            if not enable_thinking:
                prompt += "<think>\n\n</think>\n\n"
        if not tokenize:
            return prompt
        return {"input_ids": self.encode(prompt)}

    def parse_assistant_output(self, completion: str) -> AssistantOutput:
        """Parse Qwen2.5 XML tool calls from one assistant completion."""
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
        matches = list(pattern.finditer(completion))
        content = pattern.sub("", completion)
        if "<tool_call>" in content or "</tool_call>" in content:
            raise ValueError("unmatched <tool_call> tag")

        tool_calls: list[ToolCall] = []
        for match in matches:
            try:
                raw_call = cast(object, json.loads(match.group(1)))
            except json.JSONDecodeError as error:
                raise ValueError("tool call must contain valid JSON") from error
            if not isinstance(raw_call, dict):
                raise ValueError("tool call must be a JSON object")
            call_dict = cast(dict[str, object], raw_call)
            name = call_dict.get("name")
            arguments = call_dict.get("arguments")
            if not isinstance(name, str) or not name:
                raise ValueError("tool call name must be a non-empty string")
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must be a JSON object")
            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=cast(dict[str, object], arguments),
                )
            )
        return AssistantOutput(
            content=self._strip_think_blocks(content).strip(),
            tool_calls=tuple(tool_calls),
        )

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    def _apply_tool_chat_template(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[dict[str, object]],
    ) -> str:
        """Render the Qwen2.5 Hermes-style XML tool template."""
        first_is_system = bool(messages) and self._message_role(messages[0]) == "system"
        if first_is_system:
            system_content = self._message_content(messages[0])
        else:
            system_content = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            )
        prompt = (
            "<|im_start|>system\n"
            f"{system_content}\n\n"
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> "
            "XML tags:\n<tools>"
        )
        for tool in tools:
            prompt += f"\n{json.dumps(tool, ensure_ascii=False)}"
        prompt += (
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )

        start_index = 1 if first_is_system else 0
        for index in range(start_index, len(messages)):
            message = messages[index]
            role = self._message_role(message)
            content = self._message_content(message)
            if role in {"system", "user"}:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += self._format_tool_assistant_message(message, content)
            elif role == "tool":
                previous_is_tool = (
                    index > start_index
                    and self._message_role(messages[index - 1]) == "tool"
                )
                next_is_tool = (
                    index + 1 < len(messages)
                    and self._message_role(messages[index + 1]) == "tool"
                )
                if not previous_is_tool:
                    prompt += "<|im_start|>user"
                prompt += f"\n<tool_response>\n{content}\n</tool_response>"
                if not next_is_tool:
                    prompt += "<|im_end|>\n"
            else:
                raise ValueError(f"unsupported Qwen2 chat role {role!r}")
        return prompt

    def _format_tool_assistant_message(
        self,
        message: Mapping[str, object],
        content: str,
    ) -> str:
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            return f"<|im_start|>assistant\n{content}<|im_end|>\n"
        if not isinstance(raw_calls, list):
            raise TypeError("assistant tool_calls must be a list[ToolCall]")
        raw_call_items = cast(list[object], raw_calls)
        if not all(isinstance(call, ToolCall) for call in raw_call_items):
            raise TypeError("assistant tool_calls must be a list[ToolCall]")

        prompt = "<|im_start|>assistant"
        if content:
            prompt += f"\n{content}"
        for call in cast(list[ToolCall], raw_call_items):
            prompt += (
                "\n<tool_call>\n"
                f"{json.dumps({'name': call.name, 'arguments': call.arguments}, ensure_ascii=False)}\n"
                "</tool_call>"
            )
        return prompt + "<|im_end|>\n"

    @staticmethod
    def _message_role(message: Mapping[str, object]) -> str:
        role = message.get("role")
        if not isinstance(role, str):
            raise TypeError("message role must be a string")
        return role

    @staticmethod
    def _message_content(message: Mapping[str, object]) -> str:
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("message content must be a string")
        return content

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if self.tiktok is not None:
            return self.special.get(token)
        if self.tok is None:
            raise ValueError("Qwen2 tokenizer is not initialized")
        token_to_id = cast(Any, self.tok).token_to_id
        return cast(int | None, token_to_id(token))

    def decode(self, tok: list[int]) -> str:
        if self.tiktok is not None:
            return self.tiktok.decode(tok)
        if self.tok is None:
            raise ValueError("Qwen2 tokenizer is not initialized")
        decode = cast(Any, self.tok).decode
        return cast(str, decode(tok, skip_special_tokens=False))
