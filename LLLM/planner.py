"""
Specialized self-refinment with user feedback for plan creation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Protocol

import click
from loguru import logger


DEFAULT_PLANNER_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_EXPANSION_COUNT = 5
DEFAULT_CACHE_LENGTH = 16384
DEFAULT_MAX_GENERATED_TOKEN = 2048
LOG_LEVELS = ("trace", "debug", "info", "success", "warning", "error", "critical")


class TextGenerator(Protocol):
    tokenizer: Any

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
    ) -> str: ...


@dataclass(frozen=True)
class PlannerGenerationOptions:
    max_generated_token: int = DEFAULT_MAX_GENERATED_TOKEN
    enable_thinking: bool = True
    expansion_temperature: float = 0.8
    expansion_top_k: int | None = 50
    expansion_top_p: float | None = None
    refinement_temperature: float = 0.0
    refinement_top_k: int | None = None
    refinement_top_p: float | None = None


@dataclass(frozen=True)
class PlannerResult:
    request: str
    expansions: tuple[str, ...]
    feedback: tuple[str, ...]
    approved_summary: str
    task_plan: str


def strip_think_blocks(text: str) -> str:
    """Remove complete or unterminated Qwen thinking blocks from output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)


def _generate_instruct(
    generator: TextGenerator,
    prompt: str,
    *,
    enable_thinking: bool,
    max_generated_token: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> str:
    tokenizer = getattr(generator, "tokenizer", None)
    encode_prompt = getattr(tokenizer, "encode_instruct_prompt", None)
    if encode_prompt is None:
        raise TypeError("generator tokenizer must implement encode_instruct_prompt")
    prompt_tokens = encode_prompt(prompt, enable_thinking=enable_thinking)
    completion = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        include_prompt=False,
    )
    return strip_think_blocks(completion).strip()


class Planner:
    """Produce a reviewable task understanding and then an approved plan."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        options: PlannerGenerationOptions | None = None,
    ) -> None:
        self.generator = generator
        self.options = options if options is not None else PlannerGenerationOptions()
        if self.options.max_generated_token <= 0:
            raise ValueError("max_generated_token must be positive")

    def generate_expansions(
        self,
        request: str,
        *,
        expansion_count: int = DEFAULT_EXPANSION_COUNT,
    ) -> tuple[str, ...]:
        if expansion_count <= 0:
            raise ValueError("expansion_count must be positive")

        expansions: list[str] = []
        for candidate_number in range(1, expansion_count + 1):
            prompt = self._build_expansion_prompt(request, candidate_number)
            expansion = _generate_instruct(
                self.generator,
                prompt,
                enable_thinking=self.options.enable_thinking,
                max_generated_token=self.options.max_generated_token,
                temperature=self.options.expansion_temperature,
                top_k=self.options.expansion_top_k,
                top_p=self.options.expansion_top_p,
            )
            logger.info(
                "Planner generate expansion {}/{} : [{}]",
                candidate_number,
                expansion_count,
                expansion,
            )
            expansions.append(expansion)
        return tuple(expansions)

    def synthesize_summary(
        self,
        request: str,
        expansions: Sequence[str],
        *,
        previous_summary: str | None = None,
        feedback: Sequence[str] = (),
    ) -> str:
        if not expansions:
            raise ValueError("expansions must not be empty")
        logger.info("Planner generate summary from {} expansions.", len(expansions))
        prompt = self._build_summary_prompt(
            request,
            expansions,
            previous_summary=previous_summary,
            feedback=feedback,
        )
        return self._generate_refinement(prompt)

    def generate_task_plan(self, request: str, approved_summary: str) -> str:
        prompt = (
            "Create a concrete implementation task plan based only on the approved "
            "understanding below and the original request.\n"
            "Return Markdown. Include actionable ordered steps, an "
            "assumptions/decisions section, and validation criteria. Do not ask for "
            "approval again.\n\n"
            f"Original request:\n{request.strip()}\n\n"
            f"Approved understanding:\n{approved_summary.strip()}\n"
        )
        logger.info("Planner generate final task plan.")
        return self._generate_refinement(prompt)

    def _generate_refinement(self, prompt: str) -> str:
        return _generate_instruct(
            self.generator,
            prompt,
            enable_thinking=self.options.enable_thinking,
            max_generated_token=self.options.max_generated_token,
            temperature=self.options.refinement_temperature,
            top_k=self.options.refinement_top_k,
            top_p=self.options.refinement_top_p,
        )

    @staticmethod
    def _build_expansion_prompt(request: str, candidate_number: int) -> str:
        return (
            "Independently analyze the user's request before a task plan is written.\n"
            "Infer intent, task boundaries, constraints, ambiguities, and expected "
            "completion criteria. Explore a useful interpretation distinct from "
            "other candidates. Do not write the final implementation plan.\n"
            "You can include concrete tasks, subtask and design choice if relevant.\n"
            f"Candidate number: {candidate_number}\n\n"
            f"Original request:\n{request.strip()}\n"
        )

    @staticmethod
    def _build_summary_prompt(
        request: str,
        expansions: Sequence[str],
        *,
        previous_summary: str | None,
        feedback: Sequence[str],
    ) -> str:
        prompt = (
            "Agregate interpretations into a reviewable understanding of the user's request.\n"
            "Return only a Markdown bullet list capturing most of the ideas, "
            "except when it's not relevant or duplicate.\n"
            f"Original request:\n{request.strip()}\n\n"
            "Independent interpretations:\n"
        )
        prompt += "\n\n".join(
            f"Candidate {index}:\n{expansion.strip()}"
            for index, expansion in enumerate(expansions, start=1)
        )
        if previous_summary is not None:
            prompt += f"\n\nPrevious bullet summary:\n{previous_summary.strip()}\n"
        if feedback:
            prompt += "\nUser refinement feedback in order:\n"
            prompt += "\n".join(
                f"{index}. {item.strip()}"
                for index, item in enumerate(feedback, start=1)
            )
            prompt += "\nRevise the bullet summary to incorporate all feedback.\n"
        return prompt


def _build_qwen3_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
) -> TextGenerator:
    """Load the local Qwen3 generator used for every planning stage."""
    from .fetch import fetch_model_ir
    from .generator import Generator
    from .qwen3 import Qwen3Model, Qwen3Tokenizer

    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _configure_cli_logging(verbosity: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())


def _run_planning_session(
    planner: Planner,
    request: str,
    *,
    expansion_count: int,
) -> PlannerResult | None:
    expansions = planner.generate_expansions(
        request,
        expansion_count=expansion_count,
    )
    feedback: list[str] = []
    summary = planner.synthesize_summary(request, expansions)

    while True:
        click.echo(summary)
        try:
            action = click.prompt(
                "Action",
                type=click.Choice(
                    ("approve", "revise", "cancel"), case_sensitive=False
                ),
            ).lower()
        except click.Abort:
            click.echo("\nPlanning cancelled.")
            return None
        if action == "cancel":
            click.echo("Planning cancelled.")
            return None
        if action == "revise":
            try:
                feedback.append(click.prompt("Revision feedback"))
            except click.Abort:
                click.echo("\nPlanning cancelled.")
                return None
            summary = planner.synthesize_summary(
                request,
                expansions,
                previous_summary=summary,
                feedback=feedback,
            )
            continue

        task_plan = planner.generate_task_plan(request, summary)
        click.echo(task_plan)
        return PlannerResult(
            request=request,
            expansions=expansions,
            feedback=tuple(feedback),
            approved_summary=summary,
            task_plan=task_plan,
        )


@click.command(help="Turn a request into a reviewed, approved Markdown task plan.")
@click.argument("request", required=False)
@click.option(
    "--model",
    default=DEFAULT_PLANNER_MODEL_REPO_ID,
    show_default=True,
    help="Hugging Face repo id or local path for a Qwen3-compatible model.",
)
@click.option(
    "--expansion-count",
    default=DEFAULT_EXPANSION_COUNT,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of independent request interpretations to generate.",
)
@click.option(
    "--max-generated-token",
    default=DEFAULT_MAX_GENERATED_TOKEN,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum tokens generated by each planning stage.",
)
@click.option(
    "--cache-length",
    default=DEFAULT_CACHE_LENGTH,
    show_default=True,
    type=click.IntRange(min=1),
    help="KV cache length used by the generator.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    default=False,
    help="Load model artifacts from the local Hugging Face cache only.",
)
@click.option(
    "--verbosity",
    default="warning",
    show_default=True,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Logging level written to stderr.",
)
def planner_cli(
    request: str | None,
    model: str,
    expansion_count: int,
    max_generated_token: int,
    cache_length: int,
    local_files_only: bool,
    verbosity: str,
) -> None:
    """Run the planner approval loop and print a plan after approval."""
    _configure_cli_logging(verbosity)
    resolved_request: str = (
        request if request is not None else click.prompt("Request", type=str)
    )
    generator = _build_qwen3_generator(
        model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    planner = Planner(
        generator,
        options=PlannerGenerationOptions(
            max_generated_token=max_generated_token,
        ),
    )
    _run_planning_session(planner, resolved_request, expansion_count=expansion_count)


if __name__ == "__main__":
    planner_cli()
