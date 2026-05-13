import logging
from typing import Any, Callable, cast

import torch
from torch import nn

from ..LLLM.generator import Generator
from ..LLLM.gpt import GPTConfig, GPTModel


_manual_seed = cast(Callable[[int], torch.Generator], cast(Any, torch).manual_seed)


class DigitTokenizer:
    def encode(self, input: str) -> list[int]:
        return [int(char) for char in input]

    def decode(self, tok: list[int]) -> str:
        return "".join(str(token) for token in tok)


class RecordingGreedyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_contexts: list[torch.Tensor] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.seen_contexts.append(idx.clone())
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, 10, device=idx.device)

        next_token = (idx[:, -1] + 1) % 10
        logits[torch.arange(batch_size), -1, next_token] = 1.0
        return logits


def _tiny_gpt_config() -> GPTConfig:
    return {
        "vocab_size": 5,
        "context_length": 4,
        "emb_dim": 3,
        "n_heads": 1,
        "n_layers": 0,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "gpt2",
    }


def test_generator_appends_greedy_tokens_and_crops_context() -> None:
    model = RecordingGreedyModel()
    generator = Generator(model=model, tokenizer=DigitTokenizer(), context_size=2)

    generated = generator.generate("456", max_generated_token=3)

    assert generated == "456789"
    seen_contexts = [
        cast(list[list[int]], cast(Any, ctx).tolist()) for ctx in model.seen_contexts
    ]
    assert seen_contexts == [[[5, 6]], [[6, 7]], [[7, 8]]]


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


def test_generator_with_tiny_gpt_is_deterministic() -> None:
    cfg = _tiny_gpt_config()

    _manual_seed(123)
    model_a = GPTModel(cfg)
    generator_a = Generator(model_a, DigitTokenizer(), context_size=cfg["context_length"])
    generated_a = generator_a.generate("01", max_generated_token=4)

    _manual_seed(123)
    model_b = GPTModel(cfg)
    generator_b = Generator(model_b, DigitTokenizer(), context_size=cfg["context_length"])
    generated_b = generator_b.generate("01", max_generated_token=4)

    assert generated_a == "011344"
    assert generated_b == "011344"
