import logging
import math
from typing import Any, cast

import pytest
import torch
from torch import nn

from ..LLLM.generator import AssistantOutput, Generator
from ..LLLM.tool_common import ToolCall


class DigitTokenizer:
    def __init__(self, eos: int | None = None) -> None:
        self.eos = eos

    def encode(self, input: str) -> list[int]:
        return [int(char) for char in input]

    def decode(self, tok: list[int]) -> str:
        return "".join(str(token) for token in tok)

    def get_eos(self) -> int | None:
        return self.eos


class DigitChatTokenizer(DigitTokenizer):
    def __init__(self, eos: int | None = None) -> None:
        super().__init__(eos)
        self.messages: list[list[dict[str, object]]] = []
        self.tools: list[list[dict[str, object]] | None] = []
        self.enable_thinking: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> dict[str, list[int]] | str:
        self.messages.append([dict(message) for message in messages])
        self.tools.append(tools)
        self.enable_thinking.append(enable_thinking)
        prompt = "12"
        if add_generation_prompt:
            prompt += "3"
        if not tokenize:
            return prompt
        return {"input_ids": self.encode(prompt)}

    def parse_assistant_output(self, completion: str) -> AssistantOutput:
        if completion == "45":
            return AssistantOutput("parsed", (ToolCall(name="lookup", arguments={"x": 1}),))
        return AssistantOutput(completion)


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


class FilterProbeGenerator(Generator):
    def filter_logits_for_test(
        self,
        logits: torch.Tensor,
        *,
        top_k: int | None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        return self._filter_logits(logits, top_k, top_p)


def test_generator_prefills_prompt_then_uses_one_token_steps() -> None:
    model = RecordingGreedyModel()
    generator = Generator(model=model, tokenizer=DigitTokenizer(), cache_length=2)

    generated = generator.generate("456", max_generated_token=3)

    assert generated == "456789"
    seen_contexts = [
        cast(list[list[int]], cast(Any, ctx).tolist()) for ctx in model.seen_contexts
    ]
    assert seen_contexts == [[[4, 5]], [[6]], [[7]], [[8]]]


def test_generator_can_return_completion_only() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        include_prompt=False,
    )

    assert generated == "789"


def test_generator_completion_uses_chat_template_and_parses_output() -> None:
    tokenizer = DigitChatTokenizer()
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=tokenizer,
        cache_length=8,
    )
    messages = [{"role": "user", "content": "question"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    completion = generator.generate_completion(
        messages,
        tools=tools,
        max_generated_token=2,
        temperature=0.0,
        top_k=None,
        top_p=None,
        enable_thinking=False,
    )

    assert completion.raw_completion == "45"
    assert completion.message == AssistantOutput(
        "parsed",
        (ToolCall(name="lookup", arguments={"x": 1}),),
    )
    assert completion.prompt_tokens == 3
    assert completion.generated_tokens == 2
    assert completion.finish_reason == "length"
    assert tokenizer.messages == [messages]
    assert tokenizer.tools == [tools]
    assert tokenizer.enable_thinking == [False]


def test_generator_stops_before_eos_token() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(eos=7),
        cache_length=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        include_prompt=False,
    )

    assert generated == ""


def test_generator_can_continue_through_eos_token() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(eos=7),
        cache_length=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        stop_at_eos=False,
        include_prompt=False,
    )

    assert generated == "789"


def test_generator_exposes_and_logs_throughput_metrics() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=2,
    )

    generated = generator.generate("456", max_generated_token=3)

    assert generated == "456789"
    assert generator.generated_token_count == [3]
    assert generator.generated_sequence_logprob[0] == pytest.approx(
        3.0 * (1.0 - math.log(math.e + 9.0))
    )
    assert generator.generation_seconds[0] > 0.0
    assert generator.mean_token_per_second > 0.0


def test_generator_logprob_excludes_prompt_tokens() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=8,
    )

    generated = generator.generate("123456", max_generated_token=1)

    assert generated == "1234567"
    assert generator.generated_token_count == [1]
    assert generator.generated_sequence_logprob[0] == pytest.approx(
        1.0 - math.log(math.e + 9.0)
    )


def test_generator_excludes_stopping_eos_from_logprob_metric() -> None:
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(eos=7),
        cache_length=2,
    )

    generated = generator.generate(
        "456",
        max_generated_token=3,
        include_prompt=False,
    )

    assert generated == ""
    assert generator.generated_token_count == [0]
    assert generator.generated_sequence_logprob == [0.0]


def test_generator_with_tiny_cached_model_is_deterministic() -> None:
    model_a = RecordingGreedyModel()
    generator_a = Generator(model_a, DigitTokenizer(), cache_length=8)
    generated_a = generator_a.generate("01", max_generated_token=4)

    model_b = RecordingGreedyModel()
    generator_b = Generator(model_b, DigitTokenizer(), cache_length=8)
    generated_b = generator_b.generate("01", max_generated_token=4)

    assert generated_a == "012345"
    assert generated_b == "012345"


def test_filter_logits_uses_top_k_by_default() -> None:
    generator = FilterProbeGenerator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=2,
    )
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])

    filtered = generator.filter_logits_for_test(logits, top_k=2)

    is_finite = cast(list[list[bool]], cast(Any, torch.isfinite(filtered)).tolist())
    assert is_finite == [[False, True, True, False]]


def test_filter_logits_uses_top_p_instead_of_top_k_when_enabled() -> None:
    generator = FilterProbeGenerator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=2,
    )
    logits = torch.zeros(1, 4)

    filtered = generator.filter_logits_for_test(logits, top_k=1, top_p=0.74)

    assert torch.isfinite(filtered).sum().item() == 3


def test_filter_logits_rejects_invalid_top_p() -> None:
    generator = FilterProbeGenerator(
        model=RecordingGreedyModel(),
        tokenizer=DigitTokenizer(),
        cache_length=2,
    )

    with pytest.raises(ValueError, match="top_p"):
        generator.filter_logits_for_test(torch.zeros(1, 4), top_k=None, top_p=0.0)
