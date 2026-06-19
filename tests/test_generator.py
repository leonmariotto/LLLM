import logging
import math
import time
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel
import pytest
import torch
from torch import nn

from ..LLLM.generator import (
    AssistantOutput,
    ChatMessage,
    Generator,
    JsonConstrainedDecoder,
    TokenTrie,
    schema_from_pydantic,
)
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


class JsonTokenizer:
    def __init__(self) -> None:
        self.tokens = [
            "",
            "\n",
            " ",
            "<",
            ">",
            "/",
            "{",
            "}",
            '"',
            ":",
            ",",
            "[",
            "]",
            "n",
            "a",
            "d",
            "m",
            "e",
            "o",
            "c",
            "k",
            "t",
            "r",
            "u",
            "f",
            "h",
            "i",
            "l",
            "s",
            "p",
            "y",
            "z",
            "1",
            "2",
            "3",
            ".",
            "-",
            "x",
            "bad",
            "<eos>",
        ]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.eos = self.token_to_id["<eos>"]

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens)

    def encode(self, input: str) -> list[int]:
        return [0] if input == "" else [self.token_to_id[char] for char in input]

    def decode(self, tok: list[int]) -> str:
        output = ""
        for token_id in tok:
            token = self.tokens[token_id]
            if token != "<eos>":
                output += token
        return output

    def get_eos(self) -> int | None:
        return self.eos


class ScriptedJsonModel(nn.Module):
    def __init__(self, tokenizer: JsonTokenizer, target: str) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.target_ids = tokenizer.encode(target)
        self.step = 0
        self.invalid_high_steps = {0}

    def forward(
        self, idx: torch.Tensor, *, kv_cache: object | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = idx.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocabulary_size),
            -100.0,
            device=idx.device,
        )
        if self.step in self.invalid_high_steps:
            logits[:, -1, self.tokenizer.token_to_id["bad"]] = 100.0
        if self.step < len(self.target_ids):
            logits[:, -1, self.target_ids[self.step]] = 1.0
        else:
            logits[:, -1, self.tokenizer.eos] = 1.0
        self.step += 1
        return logits


class CountingTokenTrie(TokenTrie):
    def __init__(self, tokenizer: JsonTokenizer) -> None:
        super().__init__(tokenizer)
        self.valid_token_call_count = 0

    def valid_token_ids(self, is_valid_so_far: Callable[[str], bool]) -> list[int]:
        self.valid_token_call_count += 1
        return super().valid_token_ids(is_valid_so_far)


class JsonProbe(BaseModel):
    name: str
    ok: bool
    scores: list[int]


class OptionalJsonProbe(BaseModel):
    name: str | None = None
    ok: bool


class AddressProbe(BaseModel):
    city: str


class NestedJsonProbe(BaseModel):
    name: str
    address: AddressProbe


class AddressWithOptionalZipProbe(BaseModel):
    city: str
    zip: int | None = None


class NestedOptionalFieldProbe(BaseModel):
    address: AddressWithOptionalZipProbe


class NullableNestedJsonProbe(BaseModel):
    name: str
    address: AddressProbe | None = None


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
    messages: list[ChatMessage] = [{"role": "user", "content": "question"}]
    tools: list[dict[str, object]] = [
        {"type": "function", "function": {"name": "lookup"}}
    ]

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
    assert completion.trace is None
    assert tokenizer.messages == [messages]
    assert tokenizer.tools == [tools]
    assert tokenizer.enable_thinking == [False]


def test_generator_completion_trace_includes_rendered_prompt_and_raw_completion() -> None:
    tokenizer = DigitChatTokenizer()
    generator = Generator(
        model=RecordingGreedyModel(),
        tokenizer=tokenizer,
        cache_length=8,
    )

    completion = generator.generate_completion(
        [{"role": "user", "content": "question"}],
        max_generated_token=2,
        enable_thinking=False,
        trace_enabled=True,
    )

    assert completion.trace is not None
    assert completion.trace["rendered_prompt"] == "123"
    assert completion.trace["raw_completion"] == "45"
    assert completion.trace["prompt_tokens"] == 3
    assert completion.trace["generated_tokens"] == 2
    assert completion.trace["finish_reason"] == "length"
    assert completion.trace["generation_config"]["enable_thinking"] is False


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
    assert math.isclose(
        generator.generated_sequence_logprob[0],
        3.0 * (1.0 - math.log(math.e + 9.0)),
        abs_tol=1e-6,
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
    assert math.isclose(
        generator.generated_sequence_logprob[0],
        1.0 - math.log(math.e + 9.0),
        abs_tol=1e-6,
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


def test_response_format_masks_invalid_tokens_and_returns_json_string() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","ok":true,"scores":[1,2,3]}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=JsonProbe,
    )

    assert generated == target
    assert JsonProbe.model_validate_json(generated) == JsonProbe(
        name="max",
        ok=True,
        scores=[1, 2, 3],
    )


def test_response_format_allows_think_block_before_json() -> None:
    tokenizer = JsonTokenizer()
    target = '<think>hidden</think>\n\n{"name":"max","ok":true,"scores":[1,2,3]}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=JsonProbe,
    )

    assert generated == target
    json_payload = generated.rsplit("</think>", maxsplit=1)[-1].strip()
    assert JsonProbe.model_validate_json(json_payload) == JsonProbe(
        name="max",
        ok=True,
        scores=[1, 2, 3],
    )


def test_response_format_constraint_is_applied_before_top_k_sampling() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","ok":true,"scores":[1,2,3]}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=JsonProbe,
        temperature=0.6,
        top_k=1,
    )

    assert generated == target


def test_response_format_allows_optional_field_to_be_omitted() -> None:
    tokenizer = JsonTokenizer()
    target = '{"ok":true}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=OptionalJsonProbe,
    )

    assert generated == target
    assert OptionalJsonProbe.model_validate_json(generated) == OptionalJsonProbe(
        ok=True
    )


def test_response_format_allows_optional_field_value() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","ok":true}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=OptionalJsonProbe,
    )

    assert generated == target
    assert OptionalJsonProbe.model_validate_json(generated) == OptionalJsonProbe(
        name="max",
        ok=True,
    )


def test_response_format_allows_optional_field_null() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":null,"ok":false}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=OptionalJsonProbe,
    )

    assert generated == target
    assert OptionalJsonProbe.model_validate_json(generated) == OptionalJsonProbe(
        name=None,
        ok=False,
    )


def test_response_format_allows_required_nested_model_field() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","address":{"city":"paris"}}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=NestedJsonProbe,
    )

    assert generated == target
    assert NestedJsonProbe.model_validate_json(generated) == NestedJsonProbe(
        name="max",
        address=AddressProbe(city="paris"),
    )


def test_response_format_allows_nested_optional_field_to_be_omitted() -> None:
    tokenizer = JsonTokenizer()
    target = '{"address":{"city":"paris"}}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=NestedOptionalFieldProbe,
    )

    assert generated == target
    assert NestedOptionalFieldProbe.model_validate_json(
        generated
    ) == NestedOptionalFieldProbe(
        address=AddressWithOptionalZipProbe(city="paris"),
    )


def test_response_format_allows_nullable_nested_model_to_be_omitted() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max"}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=NullableNestedJsonProbe,
    )

    assert generated == target
    assert NullableNestedJsonProbe.model_validate_json(
        generated
    ) == NullableNestedJsonProbe(name="max")


def test_response_format_allows_nullable_nested_model_null() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","address":null}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=NullableNestedJsonProbe,
    )

    assert generated == target
    assert NullableNestedJsonProbe.model_validate_json(
        generated
    ) == NullableNestedJsonProbe(name="max", address=None)


def test_response_format_allows_nullable_nested_model_object() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","address":{"city":"paris"}}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    generated = generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=NullableNestedJsonProbe,
    )

    assert generated == target
    assert NullableNestedJsonProbe.model_validate_json(
        generated
    ) == NullableNestedJsonProbe(
        name="max",
        address=AddressProbe(city="paris"),
    )


def test_response_format_rejects_complete_json_with_required_field_omitted() -> None:
    tokenizer = JsonTokenizer()
    spec = schema_from_pydantic(OptionalJsonProbe)
    decoder = JsonConstrainedDecoder(
        spec=spec,
        tokenizer=tokenizer,
        trie=TokenTrie(tokenizer),
    )

    decoder.generated_text = "{}"

    assert not decoder.is_complete()


def test_response_format_is_compatible_with_generate_completion() -> None:
    class JsonChatTokenizer(JsonTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tools: list[dict[str, object]] | None = None,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
            enable_thinking: bool = True,
        ) -> dict[str, list[int]] | str:
            _ = messages, tools, add_generation_prompt, enable_thinking
            if not tokenize:
                return ""
            return {"input_ids": [0]}

        def parse_assistant_output(self, completion: str) -> AssistantOutput:
            return AssistantOutput(completion)

    tokenizer = JsonChatTokenizer()
    target = '{"name":"max","ok":false,"scores":[]}'
    generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    completion = generator.generate_completion(
        [{"role": "user", "content": "typed"}],
        max_generated_token=128,
        response_format=JsonProbe,
    )

    assert completion.raw_completion == target
    assert completion.message == AssistantOutput(target)


def test_response_format_allows_simple_nested_schema() -> None:
    class Nested(BaseModel):
        value: str

    class Supported(BaseModel):
        nested: Nested

    spec = schema_from_pydantic(Supported)

    assert spec.fields[0].name == "nested"
    assert spec.fields[0].value.kind == "object"
    assert spec.fields[0].value.fields[0].name == "value"


def test_response_format_rejects_external_ref_schema() -> None:
    class Unsupported(BaseModel):
        value: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "properties": {"value": {"$ref": "https://example.test/schema.json"}},
                "required": ["value"],
                "type": "object",
            }

    with pytest.raises(ValueError, match=r"unsupported response_format \$ref"):
        schema_from_pydantic(Unsupported)


def test_response_format_rejects_malformed_ref_schema() -> None:
    class Unsupported(BaseModel):
        value: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "properties": {"value": {"$ref": "#/$defs/Nested/extra"}},
                "required": ["value"],
                "type": "object",
                "$defs": {"Nested": {"type": "string"}},
            }

    with pytest.raises(ValueError, match=r"unsupported response_format \$ref"):
        schema_from_pydantic(Unsupported)


def test_response_format_rejects_recursive_model_schema() -> None:
    class RecursiveNode(BaseModel):
        value: str
        child: "RecursiveNode | None" = None

    RecursiveNode.model_rebuild()

    with pytest.raises(ValueError, match="recursive response_format models"):
        schema_from_pydantic(RecursiveNode)


def test_response_format_rejects_unsupported_union_schema() -> None:
    class Unsupported(BaseModel):
        value: str | int

    with pytest.raises(ValueError, match="unsupported"):
        schema_from_pydantic(Unsupported)


def test_token_trie_returns_only_valid_tokens_for_current_prefix() -> None:
    tokenizer = JsonTokenizer()
    trie = TokenTrie(tokenizer)

    token_ids = trie.valid_token_ids(lambda token_text: token_text.startswith("{"))
    token_texts = {tokenizer.decode([token_id]) for token_id in token_ids}

    assert "{" in token_texts
    assert "bad" not in token_texts


def test_response_format_uses_fast_mask_inside_think_text() -> None:
    tokenizer = JsonTokenizer()
    trie = CountingTokenTrie(tokenizer)
    decoder = JsonConstrainedDecoder(
        spec=schema_from_pydantic(JsonProbe),
        tokenizer=tokenizer,
        trie=trie,
    )
    decoder.generated_text = "<think>hidden"
    logits = torch.zeros(1, tokenizer.vocabulary_size)

    mask = decoder.mask_for_next_token(logits, tokenizer.eos)

    assert trie.valid_token_call_count == 0
    assert decoder.stats.fast_think_mask_hits == 1
    assert bool(mask[tokenizer.token_to_id["bad"]])
    assert not bool(mask[tokenizer.eos])


def test_response_format_uses_fast_mask_inside_string_value() -> None:
    tokenizer = JsonTokenizer()
    trie = CountingTokenTrie(tokenizer)
    decoder = JsonConstrainedDecoder(
        spec=schema_from_pydantic(JsonProbe),
        tokenizer=tokenizer,
        trie=trie,
    )
    decoder.generated_text = '{"name":"ma'
    logits = torch.zeros(1, tokenizer.vocabulary_size)

    mask = decoder.mask_for_next_token(logits, tokenizer.eos)

    assert trie.valid_token_call_count == 0
    assert decoder.stats.fast_text_mask_hits == 1
    assert bool(mask[tokenizer.token_to_id["x"]])
    assert bool(mask[tokenizer.token_to_id['"']])
    assert not bool(mask[tokenizer.token_to_id["\n"]])
    assert not bool(mask[tokenizer.eos])


def test_response_format_keeps_strict_mask_for_object_keys() -> None:
    tokenizer = JsonTokenizer()
    trie = CountingTokenTrie(tokenizer)
    decoder = JsonConstrainedDecoder(
        spec=schema_from_pydantic(JsonProbe),
        tokenizer=tokenizer,
        trie=trie,
    )
    decoder.generated_text = '{"'
    logits = torch.zeros(1, tokenizer.vocabulary_size)

    _ = decoder.mask_for_next_token(logits, tokenizer.eos)

    assert trie.valid_token_call_count == 1
    assert decoder.stats.trie_mask_misses == 1
    assert decoder.stats.fast_text_mask_hits == 0


def test_response_format_returns_to_strict_mask_after_string_value_closes() -> None:
    tokenizer = JsonTokenizer()
    trie = CountingTokenTrie(tokenizer)
    decoder = JsonConstrainedDecoder(
        spec=schema_from_pydantic(JsonProbe),
        tokenizer=tokenizer,
        trie=trie,
    )
    decoder.generated_text = '{"name":"max"'
    logits = torch.zeros(1, tokenizer.vocabulary_size)

    mask = decoder.mask_for_next_token(logits, tokenizer.eos)

    assert trie.valid_token_call_count == 1
    assert decoder.stats.trie_mask_misses == 1
    assert bool(mask[tokenizer.token_to_id[","]])
    assert not bool(mask[tokenizer.token_to_id["x"]])


def test_response_format_performance_overhead_is_logged_and_bounded() -> None:
    tokenizer = JsonTokenizer()
    target = '{"name":"max","ok":true,"scores":[1,2,3]}'

    baseline_generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )
    constrained_generator = Generator(
        model=ScriptedJsonModel(tokenizer, target),
        tokenizer=tokenizer,
        cache_length=8,
    )

    baseline_start = time.perf_counter()
    baseline_generator.generate("", max_generated_token=len(target), include_prompt=False)
    baseline_seconds = time.perf_counter() - baseline_start

    constrained_start = time.perf_counter()
    constrained_generator.generate(
        "",
        max_generated_token=128,
        include_prompt=False,
        response_format=JsonProbe,
    )
    constrained_seconds = time.perf_counter() - constrained_start

    overhead_ratio = constrained_seconds / max(baseline_seconds, 1e-9)
    logging.getLogger(__name__).info(
        "structured output overhead ratio %.3f "
        "baseline_seconds=%.6f constrained_seconds=%.6f",
        overhead_ratio,
        baseline_seconds,
        constrained_seconds,
    )
    assert overhead_ratio < 250.0
