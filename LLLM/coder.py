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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CompileResult:
    candidate_index: int
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
class JudgeScore:
    reason: str
    score: int


@dataclass(frozen=True)
class CoderResult:
    task: str
    candidates: tuple[CodeCandidate, ...]
    compile_results: tuple[CompileResult, ...]
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
            "Write one complete, self-contained C11 source file for the task below.\n"
            "Return only raw C code.\n"
            "Do not include markdown fences, shell commands, explanations, or tests "
            "outside the C file.\n"
            f"Candidate number: {sample_index + 1}\n\n"
            f"Task:\n{task.strip()}\n"
        )

    @staticmethod
    def extract_c_source(text: str) -> str:
        text = _strip_think_blocks(text)
        match = re.search(r"```(?:c|C)\s*\n(?P<code>.*?)```", text, re.DOTALL)
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
            "You are judging two C11 program candidates for the same task.\n"
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

    @staticmethod
    def build_score_prompt(task: str, candidate: CodeCandidate) -> str:
        return (
            "You are scoring one C11 program candidate for the original task.\n"
            "Read the program text literally. Do not guess behavior that is not "
            "present in the code.\n"
            "Score functional correctness from 0 to 100, where 100 fully solves "
            "the task and 0 is unrelated or unusable.\n"
            f"Original task:\n{task.strip()}\n\n"
            f"Program:\n"
            f"```c\n{candidate.source.strip()}\n```\n"
            "Return only valid JSON with exactly this shape:\n"
            '{"judging": "<brief analysis, pros and cons>", "score": <integer from 0 to 100>}\n'
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
        rng: random.Random | None = None,
        compile_runner: CompileRunner | None = None,
    ) -> None:
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
        self.rng = rng if rng is not None else random.Random()
        self.compile_runner = (
            compile_runner if compile_runner is not None else self._run_compile
        )

    def solve(self, task: str, *, sample_count: int | None = None) -> CoderResult:
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be positive")

        logger.info("Coder solve started")
        candidates = self._candidate_generator.generate_candidates(
            task,
            sample_count=sample_count,
        )
        compile_results = self.compile_candidates(candidates)
        judge_results = self.judge_successful_candidate_tournament(
            task,
            candidates,
            compile_results,
        )
        selected_candidate = self.select_candidate(
            candidates,
            compile_results,
            judge_results,
        )
        logger.info("Selected C candidate {}", selected_candidate.index)
        # WIP: JudgeScore is ignored for now.
        self.score_selected_candidate(task, selected_candidate)
        return CoderResult(
            task=task,
            candidates=candidates,
            compile_results=compile_results,
            judge_results=judge_results,
            selected_candidate=selected_candidate,
        )

    def score_selected_candidate(
        self,
        task: str,
        selected_candidate: CodeCandidate,
    ) -> JudgeScore | None:
        prompt = self._candidate_generator.build_score_prompt(
            task,
            selected_candidate,
        )
        logger.info("Scoring selected C candidate {}", selected_candidate.index)
        raw_output = _generate_instruct(
            self.judge_generator,
            prompt,
            enable_thinking=False,
            max_generated_token=self.judge_max_generated_token,
            temperature=self.judge_temperature,
            top_k=self.judge_top_k,
            top_p=self.judge_top_p,
        )
        judge_score = self.parse_judge_score(raw_output)
        logger.info(
            "Selected C candidate {} Judge score: {} reason: [{}]",
            selected_candidate.index,
            "unknown" if judge_score is None else judge_score.score,
            "unknown" if judge_score is None else judge_score.reason,
        )
        return judge_score

    def compile_candidates(
        self, candidates: Sequence[CodeCandidate]
    ) -> tuple[CompileResult, ...]:
        with tempfile.TemporaryDirectory(prefix="lllm-coder-") as directory:
            workdir = Path(directory)
            return tuple(
                self.compile_candidate(candidate, workdir) for candidate in candidates
            )

    def compile_candidate(
        self, candidate: CodeCandidate, workdir: Path
    ) -> CompileResult:
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
        return CompileResult(
            candidate_index=candidate.index,
            success=success,
            command=command,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    def judge_successful_candidate_tournament(
        self,
        task: str,
        candidates: Sequence[CodeCandidate],
        compile_results: Sequence[CompileResult],
    ) -> tuple[JudgeResult, ...]:
        successful_indexes = {
            result.candidate_index for result in compile_results if result.success
        }
        successful_candidates = [
            candidate
            for candidate in candidates
            if candidate.index in successful_indexes
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
        compile_results: Sequence[CompileResult],
        judge_results: Sequence[JudgeResult] = (),
    ) -> CodeCandidate:
        """
        Select only candidate that compile.
        If the judge evaluated successful candidates, choose the tournament winner.
        """
        if not candidates:
            raise ValueError("cannot select from an empty candidate list")

        successful_indexes = {
            result.candidate_index for result in compile_results if result.success
        }
        successful_candidates = [
            candidate
            for candidate in candidates
            if candidate.index in successful_indexes
        ]
        if not successful_candidates:
            logger.warning("No C candidates compiled successfully, return first")
            return candidates[0]

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
    def parse_judge_score(text: str) -> JudgeScore | None:
        last_score: JudgeScore | None = None
        for payload in Coder._iter_json_objects(text):
            score = Coder._validated_judge_score(payload, text)
            if score is not None:
                last_score = score
        if last_score is None:
            logger.warning("Could not parse valid judge score JSON: {}", text)
        return last_score

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

    @staticmethod
    def _validated_judge_score(
        payload: object,
        raw_output: str,
    ) -> JudgeScore | None:
        if not isinstance(payload, dict):
            logger.warning("Judge score JSON must be an object, ignore: {}", raw_output)
            return None

        payload_object = cast(Mapping[str, object], payload)
        judging = payload_object.get("judging")
        score = payload_object.get("score")
        if not isinstance(judging, str):
            logger.warning(
                "Judge score JSON missing string judging field, ignore: {}",
                raw_output,
            )
            return None
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or score < 0
            or score > 100
        ):
            logger.warning(
                "Judge score JSON has invalid score field, ignore: {}",
                raw_output,
            )
            return None
        return JudgeScore(reason=judging, score=score)


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
        local_files_only=local_files_only,
    )
    result = coder.solve(instruction, sample_count=sample_count)
    click.echo(result.selected_candidate.source)
