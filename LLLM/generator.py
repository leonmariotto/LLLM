"""
High-Level Generator class that provide text generation function
from a raw model.
Manage KVCache: KVCache is created and destroyed in a single generation.
(_generate_tokens).
Tool-less version. May be archived in a future iteration.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast, List

from .tool_common import ToolCall

from loguru import logger
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
        logger.info(
            "input_length={} cache_length={} max_generated_token={} "
            "temperature={} top_k={} top_p={} stop_at_eos={}",
            len(input_tokens),
            cache_length,
            max_generated_token,
            temperature,
            top_k,
            top_p,
            stop_at_eos,
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
            if step + 1 < max_generated_token:
                with torch.no_grad():
                    logits = self.model(idx_next, kv_cache=kv_cache)
            if generated_token_count % 256 == 0:
                logger.debug(
                    "Generating.. generated_token_count={}", generated_token_count
                )

        return (
            cast(list[int], cast(Any, idx.squeeze(0)).tolist()),
            generated_token_count,
            generated_sequence_logprob,
        )

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
