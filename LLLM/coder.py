"""
Self-consistency loop for code production.

Coder currently targets self-contained C programs only. It generates several
candidate files, compiles each one, and picks from the candidates that compile.
It intentionally does not execute produced code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import re
import subprocess
import tempfile
from typing import Protocol

from loguru import logger


class TextGenerator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
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
class CoderResult:
    task: str
    candidates: tuple[CodeCandidate, ...]
    compile_results: tuple[CompileResult, ...]
    selected_candidate: CodeCandidate


class CodeSelfConsistencyGenerator:
    """Generate several C source candidates for the same task."""

    def __init__(
        self,
        base: TextGenerator,
        *,
        sample_count: int = 5,
        max_generated_token: int = 2048,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.base = base
        self.sample_count = sample_count
        self.max_generated_token = max_generated_token
        self.temperature = temperature
        self.top_k = top_k

    def generate_candidates(self, task: str) -> tuple[CodeCandidate, ...]:
        candidates: list[CodeCandidate] = []
        for index in range(self.sample_count):
            prompt = self.build_prompt(task, index)
            logger.info("Generating C candidate {}/{}", index + 1, self.sample_count)
            raw_output = self.base.generate(
                prompt,
                max_generated_token=self.max_generated_token,
                temperature=self.temperature,
                top_k=self.top_k,
                include_prompt=False,
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
            "Return only C code, preferably in a single fenced ```c code block.\n"
            "Do not include shell commands, explanations, or tests outside the C file.\n"
            f"Candidate number: {sample_index + 1}\n\n"
            f"Task:\n{task.strip()}\n"
        )

    @staticmethod
    def extract_c_source(text: str) -> str:
        match = re.search(r"```(?:c|C)\s*\n(?P<code>.*?)```", text, re.DOTALL)
        if match is not None:
            return match.group("code").strip()
        return text.strip()


class Coder:
    """Generate, compile, and select a C candidate without executing it."""

    def __init__(
        self,
        generator: CodeSelfConsistencyGenerator,
        *,
        compiler: str = "gcc",
        rng: random.Random | None = None,
        compile_runner: CompileRunner | None = None,
    ) -> None:
        self.generator = generator
        self.compiler = compiler
        self.rng = rng if rng is not None else random.Random()
        self.compile_runner = (
            compile_runner if compile_runner is not None else self._run_compile
        )

    def solve(self, task: str) -> CoderResult:
        logger.info("Coder solve started")
        candidates = self.generator.generate_candidates(task)
        compile_results = self.compile_candidates(candidates)
        selected_candidate = self.select_candidate(candidates, compile_results)
        logger.info("Selected C candidate {}", selected_candidate.index)
        return CoderResult(
            task=task,
            candidates=candidates,
            compile_results=compile_results,
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
            "-Werror",
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

    def select_candidate(
        self,
        candidates: Sequence[CodeCandidate],
        compile_results: Sequence[CompileResult],
    ) -> CodeCandidate:
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
        logger.info(
            "Got %d successful_candidates, choose randomly" % len(successful_candidates)
        )
        if successful_candidates:
            return self.rng.choice(successful_candidates)

        logger.warning("No C candidates compiled successfully, return first")
        return candidates[0]

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
