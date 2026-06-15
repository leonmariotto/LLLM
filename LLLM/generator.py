"""
High-Level Generator class that provide text generation function
from a raw model.
Manage KVCache: KVCache is created and destroyed in a single generation.
(_generate_tokens).
Tool-less version. May be archived in a future iteration.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import time
import weakref
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast, List

from .tool_common import ToolCall

from loguru import logger
from pydantic import BaseModel
import torch

from .kv_cache import KVCache


class TensorModel(Protocol):
    def eval(self) -> Any: ...

    def __call__(
        self, idx: torch.Tensor, *, kv_cache: KVCache | None = None
    ) -> torch.Tensor: ...


class Tokenizer(Protocol):
    def encode(self, input: str) -> list[int]: ...

    def decode(self, tok: list[int]) -> str: ...

    def get_eos(self) -> int | None: ...


@dataclass(frozen=True)
class AssistantOutput:
    """Assistant message content plus any parsed tool calls."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class ChatMessage(TypedDict):
    """Minimal internal chat message shape accepted by local generators."""

    role: str
    content: str
    tool_calls: NotRequired[list[ToolCall]]


@dataclass(frozen=True)
class ChatCompletion:
    """Single-turn local chat completion result."""

    message: AssistantOutput
    raw_completion: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: Literal["stop", "length"]


class CompletionParseError(ValueError):
    """Raised when raw generated text cannot be parsed as an assistant message."""

    def __init__(self, raw_completion: str, parse_error: ValueError) -> None:
        super().__init__(str(parse_error))
        self.raw_completion = raw_completion
        self.parse_error = parse_error


class ChatTokenizer(Tokenizer, Protocol):
    """Tokenizer operations required by ``Generator.generate_completion``."""

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> dict[str, list[int]] | str: ...

    def parse_assistant_output(self, completion: str) -> AssistantOutput: ...


JsonKind = Literal["str", "int", "float", "bool"]


class ConstraintTokenizer(Protocol):
    """Tokenizer operations needed to build token-level JSON masks."""

    def decode(self, tok: list[int]) -> str: ...

    def get_eos(self) -> int | None: ...


@dataclass(frozen=True)
class JsonValueSpec:
    """Minimal JSON value specification supported by the constrained decoder."""

    kind: JsonKind | Literal["list"]
    item: "JsonValueSpec | None" = None


@dataclass(frozen=True)
class JsonFieldSpec:
    """Fixed-order JSON object field specification."""

    name: str
    value: JsonValueSpec


@dataclass(frozen=True)
class JsonObjectSpec:
    """Minimal fixed-order object schema extracted from a Pydantic model."""

    model: type[BaseModel]
    fields: tuple[JsonFieldSpec, ...]

    @property
    def name(self) -> str:
        """Return the schema model name for logs and errors."""
        return self.model.__name__


@dataclass
class JsonConstraintStats:
    """Runtime counters for constrained decoding performance visibility."""

    trie_build_seconds: float = 0.0
    vocabulary_size: int = 0
    mask_cache_hits: int = 0
    mask_cache_misses: int = 0


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.token_ids: list[int] = []


class TokenTrie:
    """
    Trie over decoded token strings used to collect valid token ids.
    Stores every token’s decoded text, independent of any schema.
    This could be replaced by a dictionary lookup, but tries are more
    efficient, as for a given prefix, et search only a subset matching this
    prefix.
    """

    def __init__(self, tokenizer: ConstraintTokenizer) -> None:
        start = time.perf_counter()
        self.root = _TrieNode()
        self.vocabulary_size = _tokenizer_vocabulary_size(tokenizer)
        for token_id in range(self.vocabulary_size):
            try:
                token_text = tokenizer.decode([token_id])
            except Exception as error:  # pragma: no cover - defensive for tokenizers
                logger.debug("Skip undecodable token_id={} error={}", token_id, error)
                continue
            if token_text == "":
                continue
            self._insert(token_text, token_id)
        self.build_seconds = time.perf_counter() - start
        logger.info(
            "Built constrained decoding token trie vocabulary_size={} seconds={:.6f}",
            self.vocabulary_size,
            self.build_seconds,
        )

    def _insert(self, token_text: str, token_id: int) -> None:
        node = self.root
        for char in token_text:
            node = node.children.setdefault(char, _TrieNode())
        node.token_ids.append(token_id)

    def valid_token_ids(self, is_valid_prefix: "PrefixChecker") -> list[int]:
        """Return tokens whose decoded text keeps the JSON prefix valid."""
        token_ids: list[int] = []
        self._collect(self.root, "", is_valid_prefix, token_ids)
        return token_ids

    def _collect(
        self,
        node: _TrieNode,
        token_text: str,
        is_valid_prefix: "PrefixChecker",
        token_ids: list[int],
    ) -> None:
        if token_text and not is_valid_prefix(token_text):
            return
        token_ids.extend(node.token_ids)
        for char, child in node.children.items():
            self._collect(child, token_text + char, is_valid_prefix, token_ids)


PrefixChecker = Callable[[str], bool]


_TRIE_CACHE: weakref.WeakKeyDictionary[object, TokenTrie] = weakref.WeakKeyDictionary()


def schema_from_pydantic(response_format: type[BaseModel]) -> JsonObjectSpec:
    """Extract the supported schema subset from a Pydantic BaseModel class."""
    schema = response_format.model_json_schema()
    properties = cast(object, schema.get("properties"))
    required = cast(object, schema.get("required"))
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("response_format must define an object with required fields")
    schema_properties = cast(dict[str, object], properties)
    schema_required = cast(list[object], required)

    required_names = [name for name in schema_required if isinstance(name, str)]
    if len(required_names) != len(schema_properties) or set(required_names) != set(
        schema_properties
    ):
        raise ValueError("all response_format fields must be required")

    fields: list[JsonFieldSpec] = []
    for name in response_format.model_fields:
        raw_property = schema_properties.get(name)
        if not isinstance(raw_property, dict):
            raise ValueError(f"missing JSON schema for field {name!r}")
        fields.append(
            JsonFieldSpec(
                name=name,
                value=_value_spec_from_schema(cast(dict[str, Any], raw_property), name),
            )
        )

    logger.info(
        "Accepted constrained response_format={} fields={}",
        response_format.__name__,
        [field.name for field in fields],
    )
    return JsonObjectSpec(model=response_format, fields=tuple(fields))


class JsonConstrainedDecoder:
    """
    Stateful token-mask builder for one constrained JSON completion.
    Binded to a type.

    The generated completion may be either a JSON object or a Qwen-style
    ``<think>...</think>`` block followed by a JSON object. Only the JSON
    payload is used for completion detection and final schema validation.
    """

    def __init__(
        self,
        *,
        spec: JsonObjectSpec,
        tokenizer: ConstraintTokenizer,
        trie: TokenTrie,
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.trie = trie
        self.generated_text = ""
        self._mask_cache: dict[str, torch.Tensor] = {}
        self.stats = JsonConstraintStats(
            trie_build_seconds=trie.build_seconds,
            vocabulary_size=trie.vocabulary_size,
        )
        logger.info("Constrained JSON decoding enabled schema={}", spec.name)

    def is_complete(self) -> bool:
        """Return whether the current generated text is a complete valid object."""
        payload = _json_payload_after_optional_think(self.generated_text)
        if payload is None:
            return False
        return _parse_complete_object(payload, self.spec)

    def append_token(self, token_id: int) -> None:
        """Append one selected token to the tracked JSON completion."""
        self.generated_text += self.tokenizer.decode([token_id])

    def mask_for_next_token(
        self, logits: torch.Tensor, eos: int | None
    ) -> torch.Tensor:
        """Return a boolean mask where true entries are valid next tokens."""
        cache_key = self.generated_text
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            self.stats.mask_cache_hits += 1
            return cached.to(device=logits.device)

        self.stats.mask_cache_misses += 1
        if self.is_complete():
            mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
            if eos is not None and eos < logits.shape[-1]:
                mask[eos] = True
            self._mask_cache[cache_key] = mask
            return mask.to(device=logits.device)

        def is_valid_token_suffix(token_text: str) -> bool:
            # Tokens may span several JSON grammar transitions, so validate the
            # whole appended prefix instead of checking one character at a time.
            return _parse_valid_constrained_prefix(
                self.generated_text + token_text,
                self.spec,
            )

        token_ids = self.trie.valid_token_ids(is_valid_token_suffix)
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
        valid_ids = [token_id for token_id in token_ids if token_id < logits.shape[-1]]
        if valid_ids:
            mask[torch.tensor(valid_ids, dtype=torch.long)] = True
        if eos is not None and eos < logits.shape[-1]:
            mask[eos] = False
        self._mask_cache[cache_key] = mask
        return mask.to(device=logits.device)

    def validate_final(self) -> None:
        """Validate the completed JSON text with the original Pydantic model."""
        payload = _json_payload_after_optional_think(self.generated_text)
        if payload is None:
            logger.error(
                "Constrained JSON final validation failed schema={} text={!r}",
                self.spec.name,
                self.generated_text,
            )
            raise ValueError("generated text ended before constrained JSON payload")
        try:
            self.spec.model.model_validate_json(payload)
        except ValueError as error:
            logger.error(
                "Constrained JSON final validation failed schema={} text={!r}",
                self.spec.name,
                self.generated_text,
            )
            raise ValueError(
                "generated JSON failed response_format validation"
            ) from error
        logger.info(
            "Constrained JSON final validation succeeded schema={} chars={} "
            "mask_cache_hits={} mask_cache_misses={}",
            self.spec.name,
            len(payload),
            self.stats.mask_cache_hits,
            self.stats.mask_cache_misses,
        )


def apply_json_constraint_mask(
    logits: torch.Tensor,
    decoder: JsonConstrainedDecoder,
    eos: int | None,
) -> torch.Tensor:
    """Mask logits to tokens accepted by the constrained JSON decoder."""
    mask = decoder.mask_for_next_token(logits, eos)
    if not bool(mask.any().item()):
        logger.error(
            "No valid constrained JSON token schema={} partial={!r}",
            decoder.spec.name,
            decoder.generated_text,
        )
        raise RuntimeError("no valid token remains for constrained JSON output")
    return logits.masked_fill(~mask.unsqueeze(0), float("-inf"))


def _trie_for_tokenizer(tokenizer: ConstraintTokenizer) -> TokenTrie:
    """
    Try to return an existing trie for this tokenizer.
    If none exists, build a new TokenTrie. Try to cache it.
    Return it either way.

    _TRIE_CACHE is a weakref.WeakKeyDictionary, that mean it's garbage collected
    with tokenizer.
    """
    try:
        cached = _TRIE_CACHE.get(cast(object, tokenizer))
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    trie = TokenTrie(tokenizer)
    try:
        _TRIE_CACHE[cast(object, tokenizer)] = trie
    except TypeError:
        logger.debug("Tokenizer does not support weakref trie caching")
    return trie


def _tokenizer_vocabulary_size(tokenizer: ConstraintTokenizer) -> int:
    direct = getattr(tokenizer, "vocabulary_size", None)
    if isinstance(direct, int):
        return direct
    tiktok = getattr(tokenizer, "tiktok", None)
    n_vocab = getattr(tiktok, "n_vocab", None)
    if isinstance(n_vocab, int):
        return n_vocab
    tok = getattr(tokenizer, "tok", None)
    get_vocab_size = getattr(tok, "get_vocab_size", None)
    if callable(get_vocab_size):
        value = get_vocab_size()
        if isinstance(value, int):
            return value
    raise TypeError(
        "response_format requires a tokenizer exposing vocabulary_size, "
        "tiktok.n_vocab, or tok.get_vocab_size()"
    )


def _value_spec_from_schema(schema: dict[str, Any], path: str) -> JsonValueSpec:
    schema_type = schema.get("type")
    if schema_type == "string":
        return JsonValueSpec("str")
    if schema_type == "integer":
        return JsonValueSpec("int")
    if schema_type == "number":
        return JsonValueSpec("float")
    if schema_type == "boolean":
        return JsonValueSpec("bool")
    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            raise ValueError(f"list field {path!r} must define item schema")
        item = _value_spec_from_schema(cast(dict[str, Any], items), f"{path}[]")
        if item.kind == "list":
            raise ValueError(f"nested lists are not supported at {path!r}")
        return JsonValueSpec("list", item=item)
    logger.error(
        "Rejected unsupported response_format field={} schema={}", path, schema
    )
    raise ValueError(f"unsupported response_format field {path!r}")


def _parse_valid_prefix(text: str, spec: JsonObjectSpec) -> bool:
    try:
        position = _parse_object(text, 0, spec)
    except _Incomplete:
        return True
    except _Invalid:
        return False
    return position == len(text)


def _parse_complete_object(text: str, spec: JsonObjectSpec) -> bool:
    try:
        position = _parse_object(text, 0, spec)
    except (_Incomplete, _Invalid):
        return False
    return position == len(text)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _parse_valid_constrained_prefix(text: str, spec: JsonObjectSpec) -> bool:
    """Return whether text can become optional-think-plus-JSON output."""
    if _THINK_OPEN.startswith(text):
        return True

    payload = _json_payload_after_optional_think(text)
    if payload is None:
        return text.startswith(_THINK_OPEN)
    if payload == "":
        return True
    return _parse_valid_prefix(payload, spec)


def _json_payload_after_optional_think(text: str) -> str | None:
    """
    Return the JSON payload after an optional complete Qwen think block.

    ``None`` means generation is still inside an opened think block. Leading
    whitespace before the JSON payload is ignored in both direct and thinking
    modes.
    """
    if text.startswith(_THINK_OPEN):
        close_index = text.find(_THINK_CLOSE)
        if close_index == -1:
            return None
        return text[close_index + len(_THINK_CLOSE) :].lstrip()
    return text.lstrip()


def _parse_object(text: str, position: int, spec: JsonObjectSpec) -> int:
    position = _literal(text, position, "{")
    for index, field_spec in enumerate(spec.fields):
        if index > 0:
            position = _literal(text, position, ",")
        position = _literal(text, position, json.dumps(field_spec.name))
        position = _literal(text, position, ":")
        position = _parse_value(text, position, field_spec.value)
    return _literal(text, position, "}")


def _parse_value(text: str, position: int, spec: JsonValueSpec) -> int:
    if spec.kind == "str":
        return _parse_string(text, position)
    if spec.kind == "int":
        return _parse_number(text, position, allow_float=False)
    if spec.kind == "float":
        return _parse_number(text, position, allow_float=True)
    if spec.kind == "bool":
        return _parse_bool(text, position)
    if spec.kind == "list":
        if spec.item is None:
            raise AssertionError("list spec must include item spec")
        return _parse_list(text, position, spec.item)
    raise AssertionError(f"unsupported JSON value kind {spec.kind!r}")


def _literal(text: str, position: int, literal: str) -> int:
    remaining = text[position:]
    if len(remaining) < len(literal) and literal.startswith(remaining):
        raise _Incomplete
    if text.startswith(literal, position):
        return position + len(literal)
    raise _Invalid


def _parse_bool(text: str, position: int) -> int:
    for literal in ("true", "false"):
        try:
            return _literal(text, position, literal)
        except _Incomplete:
            raise
        except _Invalid:
            pass
    if position == len(text):
        raise _Incomplete
    raise _Invalid


def _parse_string(text: str, position: int) -> int:
    if position >= len(text):
        raise _Incomplete
    if text[position] != '"':
        raise _Invalid

    index = position + 1
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            candidate = text[position : index + 1]
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                raise _Invalid
            if not isinstance(value, str):
                raise _Invalid
            return index + 1
        elif ord(char) < 0x20:
            raise _Invalid
        index += 1
    raise _Incomplete


def _parse_number(text: str, position: int, *, allow_float: bool) -> int:
    if position >= len(text):
        raise _Incomplete
    index = position
    while index < len(text) and text[index] in "-+0123456789.eE":
        index += 1
    if index == position:
        raise _Invalid

    candidate = text[position:index]
    if candidate in {"-", "+", ".", "-.", "+."}:
        raise _Incomplete
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        if index == len(text):
            raise _Incomplete
        raise _Invalid
    if isinstance(value, bool):
        raise _Invalid
    if allow_float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise _Invalid
    elif not isinstance(value, int):
        raise _Invalid
    return index


def _parse_list(text: str, position: int, item: JsonValueSpec) -> int:
    position = _literal(text, position, "[")
    if position >= len(text):
        raise _Incomplete
    if text[position] == "]":
        return position + 1

    while True:
        position = _parse_value(text, position, item)
        if position >= len(text):
            raise _Incomplete
        if text[position] == ",":
            position += 1
            if position >= len(text):
                raise _Incomplete
            continue
        if text[position] == "]":
            return position + 1
        raise _Invalid


class _Incomplete(Exception):
    pass


class _Invalid(Exception):
    pass


class Generator:
    """High level text generation class."""

    def __init__(
        self,
        model: TensorModel,
        tokenizer: Tokenizer,
        cache_length: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cache_length = cache_length
        self.generated_token_count: List[int] = []
        self.generation_seconds: List[float] = []
        self.generated_sequence_logprob: List[float] = []
        self.mean_token_per_second = 0.0

    def generate(
        self,
        prompt: str,
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        include_prompt: bool = True,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text used as the initial generation context.
            stop_at_eos: When ``True``, stop generation before appending ``eos``
                if the next predicted token is the EOS token.
            max_generated_token: Maximum number of new tokens to generate.
            cache_length: Optional per-call KV cache length override. When not
                provided, the generator default from ``__init__`` is used.
            temperature: Sampling temperature. ``0.0`` uses deterministic greedy
                argmax decoding; values above zero sample from the scaled
                probability distribution.
            top_k: If set, restrict each next-token choice to the ``top_k``
                highest-logit tokens before decoding. Ignored when ``top_p`` is
                set.
            top_p: If set, restrict each next-token choice to the smallest set
                of high-probability tokens whose cumulative probability is at
                least ``top_p``. Uses top-p instead of top-k.
            include_prompt: When ``True``, return prompt plus generated text.
                When ``False``, return only the generated completion.
            response_format: Optional Pydantic model that enables hard
                constrained compact JSON output for the generated completion.

        Returns:
            Decoded text from the selected output tokens. The returned string is
            full prompt plus completion when ``include_prompt`` is true, otherwise
            completion text only. After each call, ``last_token_per_second``,
            ``last_generated_token_count``, and ``last_generation_seconds`` expose
            generation throughput metrics for that call.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        return self.generate_from_tokens(
            prompt_tokens,
            stop_at_eos=stop_at_eos,
            max_generated_token=max_generated_token,
            cache_length=cache_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            include_prompt=include_prompt,
            response_format=response_format,
        )

    def generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        include_prompt: bool = True,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """
        Generate text from already-encoded prompt tokens.

        This is useful for instruct/chat models where the prompt includes
        structural token ids that should not be represented as ordinary text.
        """
        start_time = time.perf_counter()
        (
            generated_tokens,
            generated_token_count,
            generated_sequence_logprob,
        ) = self._generate_tokens(
            prompt_tokens,
            stop_at_eos=stop_at_eos,
            max_generated_token=max_generated_token,
            cache_length=self.cache_length if cache_length is None else cache_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            response_format=response_format,
        )
        self._record_metrics(
            generated_token_count,
            time.perf_counter() - start_time,
            generated_sequence_logprob,
        )

        output_tokens = (
            generated_tokens
            if include_prompt
            else generated_tokens[len(prompt_tokens) :]
        )
        return self.tokenizer.decode(output_tokens)

    def generate_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        enable_thinking: bool = True,
        response_format: type[BaseModel] | None = None,
    ) -> ChatCompletion:
        """Generate and parse one assistant turn from structured chat messages.

        The method mirrors the useful part of standard completion APIs while
        staying local and typed. It does not execute requested tools; callers
        should inspect ``result.message.tool_calls`` or use ``GeneratorWithTool``
        for a full tool-execution loop.
        """
        tokenizer = cast(ChatTokenizer, self.tokenizer)
        prompt_tokens = self._encode_completion_messages(
            tokenizer,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        raw_completion = self.generate_from_tokens(
            prompt_tokens,
            stop_at_eos=stop_at_eos,
            max_generated_token=max_generated_token,
            cache_length=cache_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            include_prompt=False,
            response_format=response_format,
        )
        try:
            message = tokenizer.parse_assistant_output(raw_completion)
        except ValueError as error:
            raise CompletionParseError(raw_completion, error) from error
        generated_tokens = self.generated_token_count[-1]
        finish_reason = "length" if generated_tokens >= max_generated_token else "stop"
        return ChatCompletion(
            message=message,
            raw_completion=raw_completion,
            prompt_tokens=len(prompt_tokens),
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _encode_completion_messages(
        tokenizer: ChatTokenizer,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None,
        enable_thinking: bool,
    ) -> list[int]:
        """Apply a tokenizer chat template and validate tokenized output."""
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not isinstance(encoded, dict):
            raise TypeError("expected tokenized chat template output")
        encoded_dict = cast(dict[str, object], encoded)
        input_ids = encoded_dict.get("input_ids")
        if not isinstance(input_ids, list):
            raise TypeError("expected input_ids to be a list[int]")
        input_tokens = cast(list[object], input_ids)
        if not all(isinstance(token, int) for token in input_tokens):
            raise TypeError("expected input_ids to be a list[int]")
        return cast(list[int], input_tokens)

    def _generate_tokens(
        self,
        input_tokens: list[int],
        *,
        stop_at_eos: bool,
        max_generated_token: int,
        cache_length: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        response_format: type[BaseModel] | None,
    ) -> tuple[list[int], int, float]:
        """
        Implement top-k/top-p sampling and temperature.

        Args:
            temperature: Sampling temperature.
            top_k: Optional top-k sampling cutoff.
            top_p: Optional top-p sampling cutoff. Takes precedence over top-k.
        """
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")
        constrained_decoder = self._constrained_decoder(response_format)
        logger.info(
            "input_length={} cache_length={} max_generated_token={} "
            "temperature={} top_k={} top_p={} stop_at_eos={} response_format={}",
            len(input_tokens),
            cache_length,
            max_generated_token,
            temperature,
            top_k,
            top_p,
            stop_at_eos,
            response_format.__name__ if response_format is not None else None,
        )
        self.model.eval()
        idx = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._model_device(),
        )
        kv_cache = KVCache(cache_length=cache_length)
        generated_token_count = 0
        generated_sequence_logprob = 0.0
        logits = self._prefill(idx, kv_cache, cache_length)
        eos = self.tokenizer.get_eos()

        for step in range(max_generated_token):
            logits = logits[:, -1, :]
            logits = self._filter_logits(logits, top_k, top_p)
            if constrained_decoder is not None:
                logits = apply_json_constraint_mask(logits, constrained_decoder, eos)
            idx_next = self._select_next_token(logits, temperature)
            if stop_at_eos and eos is not None and bool((idx_next == eos).all().item()):
                logger.info(
                    "Model generate an EOS, stop, generated_token_count={}",
                    generated_token_count,
                )
                break
            generated_sequence_logprob += self._selected_token_logprob(
                logits,
                idx_next,
                temperature,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            generated_token_count += int(idx_next.shape[0])
            if constrained_decoder is not None:
                constrained_decoder.append_token(int(idx_next.item()))
                if constrained_decoder.is_complete():
                    logger.info(
                        "Constrained JSON complete generated_token_count={}",
                        generated_token_count,
                    )
                    break
            if step + 1 < max_generated_token:
                with torch.no_grad():
                    logits = self.model(idx_next, kv_cache=kv_cache)
            if generated_token_count % 256 == 0:
                logger.debug(
                    "Generating.. generated_token_count={}", generated_token_count
                )

        if constrained_decoder is not None:
            constrained_decoder.validate_final()

        return (
            cast(list[int], cast(Any, idx.squeeze(0)).tolist()),
            generated_token_count,
            generated_sequence_logprob,
        )

    def _constrained_decoder(
        self, response_format: type[BaseModel] | None
    ) -> JsonConstrainedDecoder | None:
        """Create a constrained JSON decoder when typed output is requested."""
        if response_format is None:
            logger.debug("Constrained JSON decoding disabled")
            return None
        spec = schema_from_pydantic(response_format)
        trie = _trie_for_tokenizer(self.tokenizer)
        return JsonConstrainedDecoder(spec=spec, tokenizer=self.tokenizer, trie=trie)

    def _prefill(
        self,
        idx: torch.Tensor,
        kv_cache: KVCache,
        cache_length: int,
    ) -> torch.Tensor:
        """Run prompt prefill in chunks bounded by the retained cache length."""
        if idx.shape[1] == 0:
            raise ValueError("input_tokens must contain at least one token")

        logits: torch.Tensor | None = None
        for start in range(0, idx.shape[1], cache_length):
            chunk = idx[:, start : start + cache_length]
            with torch.no_grad():
                logits = self.model(chunk, kv_cache=kv_cache)

        assert logits is not None
        return logits

    def _record_metrics(
        self,
        generated_token_count: int,
        elapsed: float,
        generated_sequence_logprob: float,
    ) -> None:
        """Record generation performance metrics."""
        self.generated_token_count += [generated_token_count]
        self.generation_seconds += [elapsed]
        self.generated_sequence_logprob += [generated_sequence_logprob]
        c_count: int = 0
        c_seconds: float = 0.0
        for c, s in zip(self.generated_token_count, self.generation_seconds):
            c_count += c
            c_seconds += s
        if c_count != 0:
            self.mean_token_per_second = float(c_count) / c_seconds
        logger.info(
            "Generated {} tokens in {} (mean: {} tokens/s, logprob: {})",
            generated_token_count,
            elapsed,
            self.mean_token_per_second,
            generated_sequence_logprob,
        )

    def _filter_logits(
        self,
        logits: torch.Tensor,
        top_k: int | None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Filter logits using top-p when set, otherwise top-k. This is done before
        selecting the logit with temperature.

        Args:
            logits: Model logits to filter or sample from.
            top_k: Optional top-k sampling cutoff.
            top_p: Optional top-p sampling cutoff. Takes precedence over top-k.
        """
        if top_p is not None:
            return self._filter_top_p_logits(logits, top_p)

        if top_k is None:
            return logits

        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        return torch.where(
            logits < min_val,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    def _filter_top_p_logits(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits to the nucleus whose cumulative probability reaches top_p."""
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the interval (0.0, 1.0]")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _select_next_token(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Select next token from top_k selection. Depending on temperature.

        Args:
            logits: Model logits to filter or sample from.
            temperature: Sampling temperature.
        """
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return torch.argmax(logits, dim=-1, keepdim=True)

    def _selected_token_logprob(
        self,
        logits: torch.Tensor,
        idx_next: torch.Tensor,
        temperature: float,
    ) -> float:
        """Return summed log probability of the selected next token batch."""
        if temperature > 0.0:
            logits = logits / temperature

        logprobs = torch.log_softmax(logits, dim=-1)
        selected_logprobs = logprobs.gather(dim=-1, index=idx_next)
        return float(selected_logprobs.sum().item())

    def _model_device(self) -> torch.device:
        """Gedt the device where the model sit."""
        if not isinstance(self.model, torch.nn.Module):
            return torch.device("cpu")

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
