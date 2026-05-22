import logging
from typing import Any, cast

import torch
from torch import nn

from ..LLLM.generator import Generator


class DigitTokenizer:
    def encode(self, input: str) -> list[int]:
        return [int(char) for char in input]

    def decode(self, tok: list[int]) -> str:
        return "".join(str(token) for token in tok)


class RecordingGreedyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_contexts: list[torch.Tensor] = []

    def forward(
        self, idx: torch.Tensor, *, kv_cache: object | None = None
    ) -> torch.Tensor:
        self.seen_contexts.append(idx.clone())
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, 10, device=idx.device)

        next_token = (idx[:, -1] + 1) % 10
        logits[torch.arange(batch_size), -1, next_token] = 1.0
        return logits


def test_generator_prefills_prompt_then_uses_one_token_steps() -> None:
    model = RecordingGreedyModel()
    generator = Generator(model=model, tokenizer=DigitTokenizer(), context_size=2)

    generated = generator.generate("456", max_generated_token=3)

    assert generated == "456789"
    seen_contexts = [
        cast(list[list[int]], cast(Any, ctx).tolist()) for ctx in model.seen_contexts
    ]
    assert seen_contexts == [[[4, 5, 6]], [[7]], [[8]]]


def test_generator_can_return_completion_only() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        context_size=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        include_prompt=False,
    )

    assert generated == "789"


def test_generator_stops_before_eos_token() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        context_size=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        eos=7,
        include_prompt=False,
    )

    assert generated == ""


def test_generator_can_continue_through_eos_token() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        context_size=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        eos=7,
        stop_at_eos=False,
        include_prompt=False,
    )

    assert generated == "789"


def test_generator_exposes_and_logs_throughput_metrics(caplog) -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        context_size=2,
    )

    with caplog.at_level(logging.INFO, logger=generator.logger.name):
        generated = generator.generate("456", max_generated_token=3)

    assert generated == "456789"
    assert generator.generated_token_count == [3]
    assert generator.generation_seconds[0] > 0.0
    assert generator.mean_token_per_second > 0.0
    assert "Generated 3 tokens" in caplog.text
    assert "tokens/s" in caplog.text


def test_generator_with_tiny_cached_model_is_deterministic() -> None:
    model_a = RecordingGreedyModel()
    generator_a = Generator(model_a, DigitTokenizer(), context_size=8)
    generated_a = generator_a.generate("01", max_generated_token=4)

    model_b = RecordingGreedyModel()
    generator_b = Generator(model_b, DigitTokenizer(), context_size=8)
    generated_b = generator_b.generate("01", max_generated_token=4)

    assert generated_a == "012345"
    assert generated_b == "012345"
