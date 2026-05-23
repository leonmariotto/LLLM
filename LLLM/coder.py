"""
Self-consistency loop for code production.

Coder currently targets self-contained C programs only.
It generates several candidate files, then check compilation.
Use Qwen2.5-coder 0.5B.
Then select the best candidate by doing a judging tournament
using annother model (Qwen3 0.6B).

It intentionally does not execute produced code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
from typing import Any, Protocol

from loguru import logger


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

    def generate_candidates(self, task: str) -> tuple[CodeCandidate, ...]:
        candidates: list[CodeCandidate] = []
        for index in range(self.sample_count):
            prompt = self.build_prompt(task, index)
            logger.info("Generating C candidate {}/{}", index + 1, self.sample_count)
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
            '{"judging": "<brief comparison>", "select": "<choose between A and B>"}\n'
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

    def solve(self, task: str) -> CoderResult:
        logger.info("Coder solve started")
        candidates = self._candidate_generator.generate_candidates(task)
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
        return CoderResult(
            task=task,
            candidates=candidates,
            compile_results=compile_results,
            judge_results=judge_results,
            selected_candidate=selected_candidate,
        )

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
        winner = successful_candidates[0]

        for challenger in successful_candidates[1:]:
            prompt = self._candidate_generator.build_judge_prompt(
                task,
                winner,
                challenger,
            )
            logger.info(
                "Judging C candidates {} vs {}",
                winner.index,
                challenger.index,
            )
            raw_output = _generate_instruct(
                self.judge_generator,
                prompt,
                # enable_thinking=True, # TODO whats better ?
                enable_thinking=False,
                max_generated_token=self.judge_max_generated_token,
                temperature=self.judge_temperature,
                top_k=self.judge_top_k,
                top_p=self.judge_top_p,
            )
            winner_side = self.parse_judge_winner(raw_output)
            if winner_side == "B":
                match_winner = challenger
            else:
                match_winner = winner
            logger.info(
                "Judged C candidates {} vs {}; winner is {}: {}",
                winner.index,
                challenger.index,
                match_winner.index,
                raw_output,
            )
            judge_results.append(
                JudgeResult(
                    candidate_a_index=winner.index,
                    candidate_b_index=challenger.index,
                    winner_candidate_index=match_winner.index,
                    raw_output=raw_output,
                    prompt=prompt,
                )
            )
            winner = match_winner

        return tuple(judge_results)

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
        logger.info("Raw judge output: {}", text)
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
    def _iter_json_objects(text: str) -> Sequence[Any]:
        decoder = json.JSONDecoder()
        payloads: list[Any] = []
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
    def _validated_judge_select(payload: Any, raw_output: str) -> str | None:
        if not isinstance(payload, dict):
            logger.warning("Judge JSON must be an object, ignore: {}", raw_output)
            return None

        judging = payload.get("judging")
        winner = payload.get("select")
        if not isinstance(judging, str):
            logger.warning(
                "Judge JSON missing string judging field, ignore: {}",
                raw_output,
            )
            return None
        if winner not in {"A", "B"}:
            logger.warning(
                "Judge JSON has invalid select field, ignore: {}", raw_output
            )
            return None
        return winner
