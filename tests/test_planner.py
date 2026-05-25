# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import Any

from click.testing import CliRunner
import pytest

from ..LLLM import planner as planner_module


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.enable_thinking: list[bool] = []

    def encode_instruct_prompt(
        self,
        prompt: str,
        *,
        enable_thinking: bool = True,
    ) -> list[int]:
        self.prompts.append(prompt)
        self.enable_thinking.append(enable_thinking)
        return [len(prompt), int(enable_thinking)]


class FakeGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.tokenizer = FakeTokenizer()
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "max_generated_token": max_generated_token,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "include_prompt": include_prompt,
            }
        )
        return self.outputs[len(self.calls) - 1]


def test_generate_expansions_uses_numbered_sampled_calls() -> None:
    generator = FakeGenerator(["one", "two", "three"])
    planner = planner_module.Planner(
        generator,
        options=planner_module.PlannerGenerationOptions(max_generated_token=123),
    )

    expansions = planner.generate_expansions("Build storage", expansion_count=3)

    assert expansions == ("one", "two", "three")
    assert len(generator.tokenizer.prompts) == 3
    for number, prompt in enumerate(generator.tokenizer.prompts, start=1):
        assert f"Candidate number: {number}" in prompt
        assert "Build storage" in prompt
        assert "Do not write the final implementation plan." in prompt
    assert generator.tokenizer.enable_thinking == [True, True, True]
    assert generator.calls == [
        {
            "prompt_tokens": [len(prompt), 1],
            "max_generated_token": 123,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": None,
            "include_prompt": False,
        }
        for prompt in generator.tokenizer.prompts
    ]


def test_summary_includes_expansions_and_strips_visible_thinking() -> None:
    generator = FakeGenerator(["<think>draft</think>\n- Reviewed scope"])
    planner = planner_module.Planner(generator)

    summary = planner.synthesize_summary(
        "Create a cache",
        ("focus on API", "focus on invalidation"),
    )

    assert summary == "- Reviewed scope"
    prompt = generator.tokenizer.prompts[0]
    assert "Create a cache" in prompt
    assert "Candidate 1:\nfocus on API" in prompt
    assert "Candidate 2:\nfocus on invalidation" in prompt
    assert generator.calls[0]["temperature"] == 0.0
    assert generator.calls[0]["top_k"] is None
    assert generator.calls[0]["top_p"] is None


def test_revision_summary_preserves_previous_summary_and_all_feedback() -> None:
    generator = FakeGenerator(["- summary two", "- summary three"])
    planner = planner_module.Planner(generator)
    expansions = ("interpretation",)

    planner.synthesize_summary(
        "request",
        expansions,
        previous_summary="- summary one",
        feedback=("Must use sqlite",),
    )
    planner.synthesize_summary(
        "request",
        expansions,
        previous_summary="- summary two",
        feedback=("Must use sqlite", "Support expiry"),
    )

    second_prompt = generator.tokenizer.prompts[1]
    assert "Previous bullet summary:\n- summary two" in second_prompt
    assert "1. Must use sqlite" in second_prompt
    assert "2. Support expiry" in second_prompt
    assert second_prompt.index("Must use sqlite") < second_prompt.index(
        "Support expiry"
    )


def test_generate_task_plan_uses_only_approved_context_and_greedy_options() -> None:
    generator = FakeGenerator(["<think>work</think>\n## Steps\n1. Implement"])
    planner = planner_module.Planner(generator)

    task_plan = planner.generate_task_plan("request", "- approved behavior")

    assert task_plan == "## Steps\n1. Implement"
    prompt = generator.tokenizer.prompts[0]
    assert "request" in prompt
    assert "- approved behavior" in prompt
    assert "Independent interpretations" not in prompt
    assert generator.calls[0]["temperature"] == 0.0
    assert generator.calls[0]["top_k"] is None
    assert generator.calls[0]["top_p"] is None


def _patch_cli_generator(
    monkeypatch: pytest.MonkeyPatch,
    outputs: Sequence[str],
) -> FakeGenerator:
    generator = FakeGenerator(outputs)

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        return generator

    monkeypatch.setattr(
        planner_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )
    return generator


def test_planner_cli_accepts_positional_request_and_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(
        monkeypatch,
        ["expansion", "- summary", "## Plan\n1. work"],
    )

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1", "Make an index"],
        input="approve\n",
    )

    assert result.exit_code == 0
    assert "- summary" in result.output
    assert "## Plan\n1. work" in result.output
    assert "Make an index" in generator.tokenizer.prompts[0]
    assert len(generator.calls) == 3


def test_planner_cli_prompts_for_request_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(
        monkeypatch,
        ["expansion", "- summary", "## Plan"],
    )

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1"],
        input="Design a cache\napprove\n",
    )

    assert result.exit_code == 0
    assert "Request:" in result.output
    assert "Design a cache" in generator.tokenizer.prompts[0]


def test_planner_cli_revision_then_approval_regenerates_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(
        monkeypatch,
        ["expansion", "- initial", "- revised", "## Plan"],
    )

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1", "Design a cache"],
        input="revise\nRequire expiry\napprove\n",
    )

    assert result.exit_code == 0
    assert "- initial" in result.output
    assert "- revised" in result.output
    assert "## Plan" in result.output
    assert "Previous bullet summary:\n- initial" in generator.tokenizer.prompts[2]
    assert "1. Require expiry" in generator.tokenizer.prompts[2]
    assert len(generator.calls) == 4


def test_planner_cli_cancel_does_not_generate_final_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(monkeypatch, ["expansion", "- summary"])

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1", "Stop after review"],
        input="cancel\n",
    )

    assert result.exit_code == 0
    assert "Planning cancelled." in result.output
    assert len(generator.calls) == 2


def test_planner_cli_closed_action_input_cancels_without_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(monkeypatch, ["expansion", "- summary"])

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1", "Stop after review"],
        input="",
    )

    assert result.exit_code == 0
    assert "Planning cancelled." in result.output
    assert "Aborted!" not in result.output
    assert len(generator.calls) == 2


def test_planner_cli_closed_revision_feedback_input_cancels_without_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _patch_cli_generator(monkeypatch, ["expansion", "- summary"])

    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "1", "Stop during revision"],
        input="revise\n",
    )

    assert result.exit_code == 0
    assert "Planning cancelled." in result.output
    assert "Aborted!" not in result.output
    assert len(generator.calls) == 2


def test_planner_cli_rejects_non_positive_expansion_count() -> None:
    result = CliRunner().invoke(
        planner_module.planner_cli,
        ["--expansion-count", "0", "bad"],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--expansion-count'" in result.output
