"""
Self-consistency loop for code production.

Coder currently targets C programs only.
It generates several candidate files, then check compilation.
Use Qwen2.5-coder 0.5B.
Then select the best candidate by doing a judging tournament
using annother model (Qwen3 0.6B).

It intentionally does not execute produced code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Protocol, cast

from loguru import logger

import click


DEFAULT_CODE_MODEL_REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_JUDGE_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_CACHE_LENGTH = 16384
LOG_LEVELS = ("trace", "debug", "info", "success", "warning", "error", "critical")
DEFAULT_SELF_REFINMENT_MAX_ITERATION = 3


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


class CompileProcessResult(Protocol):
    @property
    def returncode(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


CompileRunner = Callable[[Sequence[str]], CompileProcessResult]


@dataclass(frozen=True)
class CodeCandidate:
    index: int
    prompt: str
    raw_output: str
    source: str
    compile_result: CompileResult | None = None


@dataclass(frozen=True)
class CompileResult:
    success: bool
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class JudgeResult:
    candidate_a_index: int
    candidate_b_index: int
    winner_candidate_index: int
    raw_output: str
    prompt: str


@dataclass(frozen=True)
class CoderResult:
    task: str
    candidates: tuple[CodeCandidate, ...]
    judge_results: tuple[JudgeResult, ...]
    selected_candidate: CodeCandidate


class _CodeSelfConsistencyGenerator:
    """Generate several C source candidates for the same task."""

    def __init__(
        self,
        base: TextGenerator,
        *,
        sample_count: int = 5,
        max_generated_token: int = 2048,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.base = base
        self.sample_count = sample_count
        self.max_generated_token = max_generated_token
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate_candidates(
        self,
        task: str,
        *,
        sample_count: int | None = None,
    ) -> tuple[CodeCandidate, ...]:
        if sample_count is None:
            sample_count = self.sample_count
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        candidates: list[CodeCandidate] = []
        for index in range(sample_count):
            prompt = self.build_prompt(task, index)
            logger.info("Generating C candidate {}/{}", index + 1, sample_count)
            raw_output = _generate_instruct(
                self.base,
                prompt,
                enable_thinking=True,
                max_generated_token=self.max_generated_token,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            source = self.extract_c_source(raw_output)
            logger.info(
                "Generated C candidate {} with {} source bytes: {}",
                index,
                len(source.encode()),
                source,
            )
            candidates.append(
                CodeCandidate(
                    index=index,
                    prompt=prompt,
                    raw_output=raw_output,
                    source=source,
                )
            )
        return tuple(candidates)

    @staticmethod
    def build_prompt(task: str, sample_index: int) -> str:
        return (
            "Write compilable C11 code for the task below.\n"
            "Return only raw C code.\n"
            "Do not include markdown fences, shell commands, explanations, or tests "
            "outside the C file.\n"
            f"Candidate number: {sample_index + 1}\n\n"
            f"Task:\n{task.strip()}\n"
        )

    @staticmethod
    def build_refinement_prompt(
        candidate: CodeCandidate,
    ) -> str:
        if candidate.compile_result is None:
            raise ValueError("candidate has not been compiled")
        correction_goal = "Corrected complete C program:\n"
        if "unused variable" in candidate.compile_result.stderr.lower():
            correction_goal = (
                "Corrected complete C program with the unused declaration removed:\n"
            )
        return (
            "Revise the C11 program below to resolve all compiler warnings while "
            "preserving behavior.\n"
            "Do not return the program unchanged when a warning is present.\n"
            "For an unused-variable warning, remove the unused declaration unless "
            "the variable is required for observable behavior.\n"
            "Return only the complete revised raw C code.\n"
            "Do not include markdown fences, shell commands, explanations, or tests "
            "outside the C file.\n"
            f"Compiler warning output:\n```\n{candidate.compile_result.stderr.strip()}\n```\n\n"
            f"Current program:\n```c\n{candidate.source.strip()}\n```\n\n"
            f"{correction_goal}"
        )

    def generate_refined_candidate(
        self,
        candidate: CodeCandidate,
        *,
        iteration: int,
        candidate_index: int,
    ) -> CodeCandidate:
        prompt = self.build_refinement_prompt(
            candidate,
        )
        logger.info(
            "Generating C self-refinement iteration {} prompt = [{}]", iteration, prompt
        )
        raw_output = _generate_instruct(
            self.base,
            prompt,
            enable_thinking=True,
            max_generated_token=self.max_generated_token,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        source = self.extract_c_source(raw_output)
        logger.info(
            "Self refinment iteration {} produced source [{}]", iteration, source
        )
        return CodeCandidate(
            index=candidate_index,
            prompt=prompt,
            raw_output=raw_output,
            source=source,
        )

    @staticmethod
    def extract_c_source(text: str) -> str:
        text = _strip_think_blocks(text)
        match = re.search(r"```(?:c|C)?\s*\n(?P<code>.*?)```", text, re.DOTALL)
        if match is not None:
            return match.group("code").strip()
        return text.strip()

    @staticmethod
    def build_judge_prompt(
        task: str,
        candidate_a: CodeCandidate,
        candidate_b: CodeCandidate,
    ) -> str:
        return (
            "You are judging two C11 code candidates for the same task.\n"
            "Both candidates compiled successfully. Choose the candidate that is "
            "more functionally correct for the original task.\n"
            f"Original task:\n{task.strip()}\n\n"
            f"Program A:\n"
            f"```c\n{candidate_a.source.strip()}\n```\n\n"
            f"Program B:\n"
            f"```c\n{candidate_b.source.strip()}\n```\n"
            "Read the program text literally. Do not guess behavior that is not "
            "present in the code.\n"
            "Return only valid JSON with exactly this shape:\n"
            '{"judging": "<brief comparison, pros and cons>", "select": "<your choice between A and B>"}\n'
        )


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
    if not hasattr(generator, "generate_from_tokens"):
        raise TypeError("generator must implement generate_from_tokens")
    tokenizer = getattr(generator, "tokenizer", None)
    encode_prompt = getattr(tokenizer, "encode_instruct_prompt", None)
    if encode_prompt is None:
        raise TypeError("generator tokenizer must implement encode_instruct_prompt")
    prompt_tokens = encode_prompt(prompt, enable_thinking=enable_thinking)
    return generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        include_prompt=False,
    )


def _strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)


class Coder:
    """Generate, compile, and select a C candidate without executing it."""

    def __init__(
        self,
        code_generator: TextGenerator,
        judge_generator: TextGenerator,
        *,
        sample_count: int = 5,
        max_generated_token: int = 2048,
        code_temperature: float = 0.8,
        code_top_k: int | None = 50,
        code_top_p: float | None = None,
        judge_max_generated_token: int = 1024,
        judge_temperature: float = 0.0,
        judge_top_k: int | None = None,
        judge_top_p: float | None = None,
        compiler: str = "gcc",
        self_refinment_max_iteration: int = DEFAULT_SELF_REFINMENT_MAX_ITERATION,
        rng: random.Random | None = None,
        compile_runner: CompileRunner | None = None,
    ) -> None:
        if self_refinment_max_iteration < 0:
            raise ValueError("self_refinment_max_iteration must be non-negative")
        self._candidate_generator = _CodeSelfConsistencyGenerator(
            code_generator,
            sample_count=sample_count,
            max_generated_token=max_generated_token,
            temperature=code_temperature,
            top_k=code_top_k,
            top_p=code_top_p,
        )
        self.judge_generator = judge_generator
        self.judge_max_generated_token = judge_max_generated_token
        self.judge_temperature = judge_temperature
        self.judge_top_k = judge_top_k
        self.judge_top_p = judge_top_p
        self.compiler = compiler
        self.self_refinment_max_iteration = self_refinment_max_iteration
        self.rng = rng if rng is not None else random.Random()
        self.compile_runner = (
            compile_runner if compile_runner is not None else self._run_compile
        )

    def solve(self, task: str, *, sample_count: int | None = None) -> CoderResult:
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be positive")

        logger.info("Coder solve started")
        candidates = self.compile_candidates(
            self._candidate_generator.generate_candidates(
                task,
                sample_count=sample_count,
            )
        )
        judge_results = self.judge_successful_candidate_tournament(
            task,
            candidates,
        )
        selected_candidate = self.select_candidate(
            candidates,
            judge_results,
        )
        selected_candidate, refined_candidates = self.self_refine_selected_candidate(
            task,
            selected_candidate,
            next_candidate_index=max(candidate.index for candidate in candidates) + 1,
        )
        candidates += refined_candidates
        logger.info("Selected C candidate {}", selected_candidate.index)
        return CoderResult(
            task=task,
            candidates=candidates,
            judge_results=judge_results,
            selected_candidate=selected_candidate,
        )

    def self_refine_selected_candidate(
        self,
        task: str,
        selected_candidate: CodeCandidate,
        *,
        next_candidate_index: int,
    ) -> tuple[CodeCandidate, tuple[CodeCandidate, ...]]:
        refined_candidates: list[CodeCandidate] = []
        current_candidate = selected_candidate

        with tempfile.TemporaryDirectory(prefix="lllm-coder-refinement-") as directory:
            workdir = Path(directory)
            for iteration in range(1, self.self_refinment_max_iteration + 1):
                if not self.has_compilation_warning(current_candidate):
                    break

                refined_candidate = (
                    self._candidate_generator.generate_refined_candidate(
                        current_candidate,
                        iteration=iteration,
                        candidate_index=next_candidate_index + len(refined_candidates),
                    )
                )
                refined_candidate = self.compile_candidate(refined_candidate, workdir)
                refined_candidates.append(refined_candidate)
                refined_compile_result = self._get_compile_result(refined_candidate)
                if not refined_compile_result.success:
                    logger.warning(
                        "C self-refinement iteration {} failed compilation; keeping "
                        "candidate {}",
                        iteration,
                        current_candidate.index,
                    )
                    break
                current_candidate = refined_candidate

        return (
            current_candidate,
            tuple(refined_candidates),
        )

    def compile_candidates(
        self, candidates: Sequence[CodeCandidate]
    ) -> tuple[CodeCandidate, ...]:
        with tempfile.TemporaryDirectory(prefix="lllm-coder-") as directory:
            workdir = Path(directory)
            return tuple(
                self.compile_candidate(candidate, workdir) for candidate in candidates
            )

    def compile_candidate(
        self, candidate: CodeCandidate, workdir: Path
    ) -> CodeCandidate:
        source_path = workdir / f"candidate_{candidate.index}.c"
        object_path = workdir / f"candidate_{candidate.index}.o"
        source_path.write_text(candidate.source, encoding="utf-8")
        command = (
            self.compiler,
            "-x",
            "c",
            "-std=c11",
            "-Wall",
            "-Wextra",
            # "-Werror",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        )

        logger.info("Compiling C candidate {}", candidate.index)
        process = self.compile_runner(command)
        success = process.returncode == 0
        if success:
            logger.info("Compiled C candidate {}", candidate.index)
        else:
            logger.warning(
                "C candidate {} failed to compile with return code {}: {}",
                candidate.index,
                process.returncode,
                self._stderr_summary(process.stderr),
            )
        return replace(
            candidate,
            compile_result=CompileResult(
                success=success,
                command=command,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            ),
        )

    def judge_successful_candidate_tournament(
        self,
        task: str,
        candidates: Sequence[CodeCandidate],
    ) -> tuple[JudgeResult, ...]:
        successful_candidates = [
            candidate
            for candidate in candidates
            if candidate.compile_result is not None and candidate.compile_result.success
        ]
        if len(successful_candidates) < 2:
            return ()

        judge_results: list[JudgeResult] = []
        round_candidates = successful_candidates

        while len(round_candidates) > 1:
            next_round: list[CodeCandidate] = []
            for offset in range(0, len(round_candidates), 2):
                if offset + 1 >= len(round_candidates):
                    next_round.append(round_candidates[offset])
                    continue

                candidate_a = round_candidates[offset]
                candidate_b = round_candidates[offset + 1]
                match_winner = self._judge_candidate_pair(
                    task, candidate_a, candidate_b
                )
                judge_results.append(match_winner[0])
                next_round.append(match_winner[1])

            round_candidates = next_round

        return tuple(judge_results)

    def _judge_candidate_pair(
        self,
        task: str,
        candidate_a: CodeCandidate,
        candidate_b: CodeCandidate,
    ) -> tuple[JudgeResult, CodeCandidate]:
        prompt = self._candidate_generator.build_judge_prompt(
            task,
            candidate_a,
            candidate_b,
        )
        logger.info(
            "Judging C candidates {} vs {}",
            candidate_a.index,
            candidate_b.index,
        )
        raw_output = _generate_instruct(
            self.judge_generator,
            prompt,
            enable_thinking=False,
            max_generated_token=self.judge_max_generated_token,
            temperature=self.judge_temperature,
            top_k=self.judge_top_k,
            top_p=self.judge_top_p,
        )
        winner_side = self.parse_judge_winner(raw_output)
        if winner_side == "B":
            selected_candidate = candidate_b
        else:
            selected_candidate = candidate_a
        logger.info(
            "Judged C candidates {} vs {}; winner is {}: {}",
            candidate_a.index,
            candidate_b.index,
            selected_candidate.index,
            raw_output,
        )
        return (
            JudgeResult(
                candidate_a_index=candidate_a.index,
                candidate_b_index=candidate_b.index,
                winner_candidate_index=selected_candidate.index,
                raw_output=raw_output,
                prompt=prompt,
            ),
            selected_candidate,
        )

    def select_candidate(
        self,
        candidates: Sequence[CodeCandidate],
        judge_results: Sequence[JudgeResult] = (),
    ) -> CodeCandidate:
        """
        Select only candidate that compile.
        If the judge evaluated successful candidates, choose the tournament winner.
        """
        if not candidates:
            raise ValueError("cannot select from an empty candidate list")

        successful_candidates = [
            candidate
            for candidate in candidates
            if candidate.compile_result is not None and candidate.compile_result.success
        ]
        if not successful_candidates:
            logger.warning("No C candidates compiled successfully, return first")
            return candidates[0]

        if len(successful_candidates) == 1:
            logger.warning("Only on C candidates compiled successfully, return it")
            return successful_candidates[0]

        if judge_results:
            winner_index = judge_results[-1].winner_candidate_index
            selected = next(
                candidate
                for candidate in successful_candidates
                if candidate.index == winner_index
            )
            logger.info(
                "Selected tournament winner C candidate {}",
                selected.index,
            )
            return selected

        logger.info(
            "Got %d successful_candidates without judge scores, choose randomly"
            % len(successful_candidates)
        )
        return self.rng.choice(successful_candidates)

    @staticmethod
    def _run_compile(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _stderr_summary(stderr: str, *, max_length: int = 300) -> str:
        summary = " ".join(stderr.strip().split())
        if len(summary) <= max_length:
            return summary
        return f"{summary[: max_length - 3]}..."

    @staticmethod
    def _get_compile_result(candidate: CodeCandidate) -> CompileResult:
        if candidate.compile_result is None:
            raise ValueError("candidate has not been compiled")
        return candidate.compile_result

    @staticmethod
    def has_compilation_warning(candidate: CodeCandidate) -> bool:
        compile_result = Coder._get_compile_result(candidate)
        return compile_result.success and "warning:" in compile_result.stderr.lower()

    @staticmethod
    def parse_judge_winner(text: str) -> str:
        last_winner: str | None = None
        for payload in Coder._iter_json_objects(text):
            winner = Coder._validated_judge_select(payload, text)
            if winner is not None:
                last_winner = winner

        if last_winner is not None:
            return last_winner

        logger.warning("Could not parse valid judge JSON, keep A: {}", text)
        return "A"

    @staticmethod
    def _iter_json_objects(text: str) -> Sequence[object]:
        decoder = json.JSONDecoder()
        payloads: list[object] = []
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            payloads.append(payload)
        return tuple(payloads)

    @staticmethod
    def _validated_judge_select(payload: object, raw_output: str) -> str | None:
        if not isinstance(payload, dict):
            logger.warning("Judge JSON must be an object, ignore: {}", raw_output)
            return None

        payload_object = cast(Mapping[str, object], payload)
        judging = payload_object.get("judging")
        winner = payload_object.get("select")
        if not isinstance(judging, str):
            logger.warning(
                "Judge JSON missing string judging field, ignore: {}",
                raw_output,
            )
            return None
        if not isinstance(winner, str) or winner not in {"A", "B"}:
            logger.warning(
                "Judge JSON has invalid select field, ignore: {}", raw_output
            )
            return None
        return winner


def _build_qwen2_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
) -> TextGenerator:
    from .fetch import fetch_model_ir
    from .generator import Generator
    from .qwen2 import Qwen2Model, Qwen2Tokenizer

    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    cfg = Qwen2Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen2Tokenizer(str(path / "tokenizer.json"))
    model = Qwen2Model(cfg)
    model.load_ir_weights(ir)

    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_qwen3_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
) -> TextGenerator:
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


def _build_cli_coder(
    *,
    code_model: str,
    judge_model: str,
    max_generated_token: int,
    judge_max_generated_token: int,
    cache_length: int,
    code_top_p: float | None,
    self_refinment_max_iteration: int,
    local_files_only: bool,
) -> Coder:
    code_generator = _build_qwen2_generator(
        code_model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    judge_generator = _build_qwen3_generator(
        judge_model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    return Coder(
        code_generator,
        judge_generator,
        max_generated_token=max_generated_token,
        judge_max_generated_token=judge_max_generated_token,
        code_top_p=code_top_p,
        self_refinment_max_iteration=self_refinment_max_iteration,
    )


def _configure_cli_logging(verbosity: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())


@click.command(
    help=(
        "Read a coding instruction from stdin, generate C candidates, compile "
        'them, then use another "judge" LLM to choose the best in a tournament. '
        "Print the selected C source to stdout."
    )
)
@click.option(
    "--code-model",
    default=DEFAULT_CODE_MODEL_REPO_ID,
    show_default=True,
    help="Hugging Face repo id or local path for the Qwen2-compatible coder model.",
)
@click.option(
    "--judge-model",
    default=DEFAULT_JUDGE_MODEL_REPO_ID,
    show_default=True,
    help="Hugging Face repo id or local path for the Qwen3-compatible judge model.",
)
@click.option(
    "--sample-count",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of candidate programs to generate.",
)
@click.option(
    "--max-generated-token",
    default=4096,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum tokens to generate for each C candidate.",
)
@click.option(
    "--judge-max-generated-token",
    default=2048,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum tokens to generate for each judge decision.",
)
@click.option(
    "--cache-length",
    default=DEFAULT_CACHE_LENGTH,
    show_default=True,
    type=click.IntRange(min=1),
    help="KV cache length used by both generators.",
)
@click.option(
    "--code-top-p",
    default=0.90,
    show_default=True,
    type=float,
    help="Top-p sampling value for candidate generation.",
)
@click.option(
    "--self-refinment-max-iteration",
    default=DEFAULT_SELF_REFINMENT_MAX_ITERATION,
    show_default=True,
    type=click.IntRange(min=0),
    help="Maximum warning-driven revisions of the selected C program.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    help="Only use models already present in the local Hugging Face cache.",
)
@click.option(
    "--verbosity",
    default="info",
    show_default=True,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log verbosity.",
)
def coder_cli(
    code_model: str,
    judge_model: str,
    sample_count: int,
    max_generated_token: int,
    judge_max_generated_token: int,
    cache_length: int,
    code_top_p: float,
    self_refinment_max_iteration: int,
    local_files_only: bool,
    verbosity: str,
) -> None:
    _configure_cli_logging(verbosity)
    instruction = click.get_text_stream("stdin").read()
    if not instruction.strip():
        raise click.UsageError("expected a coding instruction on stdin")

    coder = _build_cli_coder(
        code_model=code_model,
        judge_model=judge_model,
        max_generated_token=max_generated_token,
        judge_max_generated_token=judge_max_generated_token,
        cache_length=cache_length,
        code_top_p=code_top_p,
        self_refinment_max_iteration=self_refinment_max_iteration,
        local_files_only=local_files_only,
    )
    result = coder.solve(instruction, sample_count=sample_count)
    click.echo(result.selected_candidate.source)
