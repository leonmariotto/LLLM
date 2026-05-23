from collections.abc import Sequence
from dataclasses import dataclass
import random

from ..LLLM.coder import (
    CodeCandidate,
    CodeSelfConsistencyGenerator,
    CompileResult,
    Coder,
)


class FakeTextGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.prompt_tokens: list[list[int]] = []

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
    ) -> str:
        self.prompts.append(prompt)
        return self.outputs[len(self.prompts) - 1]

    def generate_from_tokens(
        self,
        prompt_tokens: list[int],
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        include_prompt: bool = True,
    ) -> str:
        self.prompt_tokens.append(prompt_tokens)
        return self.outputs[len(self.prompt_tokens) - 1]


@dataclass(frozen=True)
class FakeCompileProcess:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def test_self_consistency_generator_requests_five_candidates() -> None:
    base = FakeTextGenerator(
        [
            "int main(void) { return 0; }",
            "int main(void) { return 1; }",
            "int main(void) { return 2; }",
            "int main(void) { return 3; }",
            "int main(void) { return 4; }",
        ]
    )
    generator = CodeSelfConsistencyGenerator(base)

    candidates = generator.generate_candidates("return a status code")

    assert len(candidates) == 5
    assert len(base.prompts) == 5
    assert candidates[0].source == "int main(void) { return 0; }"
    assert "Candidate number: 5" in base.prompts[4]
    assert "Do not include markdown fences" in base.prompts[0]


def test_self_consistency_generator_can_use_encoded_prompts() -> None:
    base = FakeTextGenerator(["int main(void) { return 0; }"])
    generator = CodeSelfConsistencyGenerator(
        base,
        encode_prompt=lambda prompt: [len(prompt)],
        sample_count=1,
    )

    candidates = generator.generate_candidates("return a status code")

    assert candidates[0].source == "int main(void) { return 0; }"
    assert base.prompts == []
    assert base.prompt_tokens == [[len(candidates[0].prompt)]]


def test_extract_c_source_prefers_fenced_c_block() -> None:
    text = "Here is code:\n```c\nint main(void) { return 0; }\n```\nDone"

    source = CodeSelfConsistencyGenerator.extract_c_source(text)

    assert source == "int main(void) { return 0; }"


def test_extract_c_source_falls_back_to_raw_completion() -> None:
    text = "\nint main(void) { return 0; }\n"

    source = CodeSelfConsistencyGenerator.extract_c_source(text)

    assert source == "int main(void) { return 0; }"


def test_coder_compiles_every_candidate_without_running_code() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "int main(void) { return 0; }"),
        CodeCandidate(1, "prompt", "raw", "int main(void) { return 1; }"),
    )
    commands: list[tuple[str, ...]] = []

    def compile_runner(command: Sequence[str]) -> FakeCompileProcess:
        commands.append(tuple(command))
        return FakeCompileProcess(returncode=0)

    coder = Coder(
        CodeSelfConsistencyGenerator(FakeTextGenerator([])),
        compile_runner=compile_runner,
    )

    compile_results = coder.compile_candidates(candidates)

    assert len(compile_results) == 2
    assert [result.success for result in compile_results] == [True, True]
    assert len(commands) == 2
    assert all(command[0] == "gcc" for command in commands)
    assert all("-c" in command for command in commands)
    assert all("-o" in command for command in commands)


def test_coder_default_gcc_compile_runner_compiles_source() -> None:
    candidate = CodeCandidate(0, "prompt", "raw", "int main(void) { return 0; }")
    coder = Coder(CodeSelfConsistencyGenerator(FakeTextGenerator([])))

    compile_results = coder.compile_candidates((candidate,))

    assert compile_results[0].success
    assert compile_results[0].command[0] == "gcc"
    assert "-c" in compile_results[0].command


def test_coder_selects_from_successes_with_seeded_rng() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source"),
        CodeCandidate(1, "prompt", "raw", "source"),
        CodeCandidate(2, "prompt", "raw", "source"),
    )
    compiler_results = (
        FakeCompileProcess(returncode=1),
        FakeCompileProcess(returncode=0),
        FakeCompileProcess(returncode=0),
    )
    generator = CodeSelfConsistencyGenerator(FakeTextGenerator([]))
    coder = Coder(generator, rng=random.Random(0))
    compile_results = tuple(
        CompileResult(
            candidate_index=index,
            success=process.returncode == 0,
            command=("gcc",),
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
        for index, process in enumerate(compiler_results)
    )

    selected = coder.select_candidate(candidates, compile_results)

    assert selected.index == 2


def test_coder_selects_first_candidate_when_all_compiles_fail() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source"),
        CodeCandidate(1, "prompt", "raw", "source"),
    )
    generator = CodeSelfConsistencyGenerator(FakeTextGenerator([]))
    coder = Coder(generator)
    compile_results = (
        CompileResult(
            candidate_index=0,
            success=False,
            command=("gcc",),
            returncode=1,
            stdout="",
            stderr="error",
        ),
        CompileResult(
            candidate_index=1,
            success=False,
            command=("gcc",),
            returncode=1,
            stdout="",
            stderr="error",
        ),
    )

    selected = coder.select_candidate(candidates, compile_results)

    assert selected == candidates[0]
