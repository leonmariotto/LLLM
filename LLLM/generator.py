"""
High-Level Generator class that provide text generation function
from a raw model.
"""

import logging
import time
from typing import Any, Protocol, cast, List

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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.generated_token_count: List[int] = []
        self.generation_seconds: List[float] = []
        self.mean_token_per_second = 0.0

    def generate(
        self,
        prompt: str,
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        eos: int | None = None,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        include_prompt: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text used as the initial generation context.
            stop_at_eos: When ``True``, stop generation before appending ``eos``
                if the next predicted token is the EOS token.
            max_generated_token: Maximum number of new tokens to generate.
            eos: Token id treated as end-of-sequence when ``stop_at_eos`` is
                enabled. If ``None``, no EOS stopping is applied.
            cache_length: Optional per-call KV cache length override. When not
                provided, the generator default from ``__init__`` is used.
            temperature: Sampling temperature. ``0.0`` uses deterministic greedy
                argmax decoding; values above zero sample from the scaled
                probability distribution.
            top_k: If set, restrict each next-token choice to the ``top_k``
                highest-logit tokens before decoding.
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
            eos=eos,
            cache_length=cache_length,
            temperature=temperature,
            top_k=top_k,
            include_prompt=include_prompt,
        )

    def generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        eos: int | None = None,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        include_prompt: bool = True,
    ) -> str:
        """
        Generate text from already-encoded prompt tokens.

        This is useful for instruct/chat models where the prompt includes
        structural token ids that should not be represented as ordinary text.
        """
        start_time = time.perf_counter()
        generated_tokens, generated_token_count = self._generate_tokens(
            prompt_tokens,
            stop_at_eos=stop_at_eos,
            max_generated_token=max_generated_token,
            eos=eos,
            cache_length=self.cache_length if cache_length is None else cache_length,
            temperature=temperature,
            top_k=top_k,
        )
        self._record_metrics(generated_token_count, time.perf_counter() - start_time)

        output_tokens = (
            generated_tokens
            if include_prompt
            else generated_tokens[len(prompt_tokens) :]
        )
        return self.tokenizer.decode(output_tokens)

    def _generate_tokens(
        self,
        input_tokens: list[int],
        *,
        stop_at_eos: bool,
        max_generated_token: int,
        eos: int | None,
        cache_length: int,
        temperature: float,
        top_k: int | None,
    ) -> tuple[list[int], int]:
        """
        Implement top-k sampling and temperature.

        Args:
            temperature: Sampling temperature.
            top_k: Optional top-k sampling cutoff.
        """
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")
        self.model.eval()
        idx = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._model_device(),
        )
        kv_cache = KVCache(cache_length=cache_length)
        generated_token_count = 0
        logits = self._prefill(idx, kv_cache, cache_length)

        for step in range(max_generated_token):
            logits = logits[:, -1, :]
            logits = self._filter_logits(logits, top_k)
            idx_next = self._select_next_token(logits, temperature)
            if stop_at_eos and eos is not None and bool((idx_next == eos).all().item()):
                break
            idx = torch.cat((idx, idx_next), dim=1)
            generated_token_count += int(idx_next.shape[0])
            if step + 1 < max_generated_token:
                with torch.no_grad():
                    logits = self.model(idx_next, kv_cache=kv_cache)

        return cast(
            list[int], cast(Any, idx.squeeze(0)).tolist()
        ), generated_token_count

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

    def _record_metrics(self, generated_token_count: int, elapsed: float) -> None:
        """Record generation performance metrics."""
        self.generated_token_count += [generated_token_count]
        self.generation_seconds += [elapsed]
        c_count: int = 0
        c_seconds: float = 0.0
        for c, s in zip(self.generated_token_count, self.generation_seconds):
            c_count += c
            c_seconds += s
        if c_count != 0:
            self.mean_token_per_second = float(c_count) / c_seconds
        self.logger.info(
            "Generated %s tokens in %.4fs (mean: %.2f tokens/s)",
            generated_token_count,
            elapsed,
            self.mean_token_per_second,
        )

    def _filter_logits(self, logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        """
        Filter logits to select only top_k logits. This is done before selecting
        the logit with temperature.

        Args:
            logits: Model logits to filter or sample from.
            top_k: Optional top-k sampling cutoff.
        """
        if top_k is None:
            return logits

        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        return torch.where(
            logits < min_val,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

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

    def _model_device(self) -> torch.device:
        """Gedt the device where the model sit."""
        if not isinstance(self.model, torch.nn.Module):
            return torch.device("cpu")

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
