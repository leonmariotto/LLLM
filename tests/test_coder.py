from collections.abc import Sequence
from dataclasses import dataclass
import random

from ..LLLM.coder import (
    CodeCandidate,
    CompileResult,
    Coder,
    JudgeResult,
)


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


class FakeTextGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.tokenizer = FakeTokenizer()
        self.prompt_tokens: list[list[int]] = []
        self.calls: list[dict[str, object]] = []

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
        self.prompt_tokens.append(prompt_tokens)
        self.calls.append(
            {
                "max_generated_token": max_generated_token,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "include_prompt": include_prompt,
            }
        )
        return self.outputs[len(self.prompt_tokens) - 1]


@dataclass(frozen=True)
class FakeCompileProcess:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def test_coder_requests_five_code_candidates_with_thinking() -> None:
    code_generator = FakeTextGenerator(
        [
            "int main(void) { return 0; }",
            "int main(void) { return 1; }",
            "int main(void) { return 2; }",
            "int main(void) { return 3; }",
            "int main(void) { return 4; }",
        ]
    )
    judge_generator = FakeTextGenerator([])
    coder = Coder(code_generator, judge_generator)

    candidates = coder._candidate_generator.generate_candidates("return a status code")

    assert len(candidates) == 5
    assert len(code_generator.tokenizer.prompts) == 5
    assert judge_generator.tokenizer.prompts == []
    assert candidates[0].source == "int main(void) { return 0; }"
    assert "Candidate number: 5" in code_generator.tokenizer.prompts[4]
    assert "Do not include markdown fences" in code_generator.tokenizer.prompts[0]
    assert code_generator.tokenizer.enable_thinking == [True] * 5


def test_coder_code_generation_uses_configured_sampling_options() -> None:
    code_generator = FakeTextGenerator(["int main(void) { return 0; }"])
    coder = Coder(
        code_generator,
        FakeTextGenerator([]),
        sample_count=1,
        max_generated_token=123,
        code_temperature=0.7,
        code_top_k=17,
        code_top_p=0.91,
    )

    candidates = coder._candidate_generator.generate_candidates("return a status code")

    assert candidates[0].source == "int main(void) { return 0; }"
    assert code_generator.prompt_tokens == [
        [len(candidates[0].prompt), 1],
    ]
    assert code_generator.calls == [
        {
            "max_generated_token": 123,
            "temperature": 0.7,
            "top_k": 17,
            "top_p": 0.91,
            "include_prompt": False,
        }
    ]


def test_extract_c_source_prefers_fenced_c_block() -> None:
    text = "Here is code:\n```c\nint main(void) { return 0; }\n```\nDone"

    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))
    source = coder._candidate_generator.extract_c_source(text)

    assert source == "int main(void) { return 0; }"


def test_extract_c_source_falls_back_to_raw_completion() -> None:
    text = "\nint main(void) { return 0; }\n"

    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))
    source = coder._candidate_generator.extract_c_source(text)

    assert source == "int main(void) { return 0; }"


def test_extract_c_source_strips_thinking_before_code() -> None:
    text = (
        "<think>\nI should produce a tiny program.\n</think>\n"
        "int main(void) { return 0; }\n"
    )

    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))
    source = coder._candidate_generator.extract_c_source(text)

    assert source == "int main(void) { return 0; }"


def test_build_judge_prompt_contains_task_and_both_candidates() -> None:
    candidate_a = CodeCandidate(
        0,
        "prompt",
        "raw",
        "int main(void) { return 0; }",
    )
    candidate_b = CodeCandidate(
        1,
        "prompt",
        "raw",
        "int main(void) { return 1; }",
    )

    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))

    prompt = coder._candidate_generator.build_judge_prompt(
        "return success",
        candidate_a,
        candidate_b,
    )

    assert "return success" in prompt
    assert "int main(void) { return 0; }" in prompt
    assert "int main(void) { return 1; }" in prompt
    assert '"judging": "<brief comparison>"' in prompt
    assert '"select": "<A or B>"' in prompt


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
        FakeTextGenerator([]),
        FakeTextGenerator([]),
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
    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))

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
    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]), rng=random.Random(0))
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


def test_coder_runs_tournament_over_successful_candidates() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source 0"),
        CodeCandidate(1, "prompt", "raw", "source 1"),
        CodeCandidate(2, "prompt", "raw", "source 2"),
        CodeCandidate(3, "prompt", "raw", "source 3"),
    )
    code_generator = FakeTextGenerator([])
    judge_generator = FakeTextGenerator(
        [
            '{"judging": "candidate 2 is better", "select": "B"}',
            '{"judging": "candidate 2 remains better", "select": "A"}',
        ]
    )
    coder = Coder(
        code_generator,
        judge_generator,
        judge_max_generated_token=77,
        judge_temperature=0.0,
        judge_top_k=None,
        judge_top_p=None,
    )
    compile_results = (
        CompileResult(
            candidate_index=0,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        ),
        CompileResult(
            candidate_index=1,
            success=False,
            command=("gcc",),
            returncode=1,
            stdout="",
            stderr="error",
        ),
        CompileResult(
            candidate_index=2,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        ),
        CompileResult(
            candidate_index=3,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    judge_results = coder.judge_successful_candidate_tournament(
        "write a program",
        candidates,
        compile_results,
    )
    selected = coder.select_candidate(candidates, compile_results, judge_results)

    assert [
        (
            result.candidate_a_index,
            result.candidate_b_index,
            result.winner_candidate_index,
        )
        for result in judge_results
    ] == [
        (0, 2, 2),
        (2, 3, 2),
    ]
    assert "source 2" in judge_results[1].prompt
    assert "source 3" in judge_results[1].prompt
    assert selected.index == 2
    assert code_generator.tokenizer.prompts == []
    assert judge_generator.tokenizer.enable_thinking == [True, True]
    assert judge_generator.calls == [
        {
            "max_generated_token": 77,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "include_prompt": False,
        },
        {
            "max_generated_token": 77,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "include_prompt": False,
        },
    ]


def test_coder_selects_last_tournament_winner() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source"),
        CodeCandidate(1, "prompt", "raw", "source"),
    )
    compile_results = (
        CompileResult(
            candidate_index=0,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        ),
        CompileResult(
            candidate_index=1,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    judge_results = (
        JudgeResult(0, 1, 1, "Winner: B", "prompt"),
    )
    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))

    selected = coder.select_candidate(candidates, compile_results, judge_results)

    assert selected.index == 1


def test_coder_selects_first_candidate_when_all_compiles_fail() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source"),
        CodeCandidate(1, "prompt", "raw", "source"),
    )
    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))
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


def test_parse_judge_winner_accepts_valid_json() -> None:
    assert Coder.parse_judge_winner('{"judging": "A is better", "select": "A"}') == "A"
    assert Coder.parse_judge_winner('{"judging": "B is better", "select": "B"}') == "B"
    assert (
        Coder.parse_judge_winner(
            '<think>B is better</think>\n'
            '```json\n{"judging": "B is better", "select": "B"}\n```'
        )
        == "B"
    )
    assert (
        Coder.parse_judge_winner(
            '{"judging": "draft answer", "select": "A"}\n'
            '{"judging": "final answer", "select": "B"}'
        )
        == "B"
    )


def test_parse_judge_winner_falls_back_to_a_for_invalid_json() -> None:
    assert Coder.parse_judge_winner("Winner: Candidate B") == "A"
    assert Coder.parse_judge_winner('{"judging": "B is better"}') == "A"
    assert Coder.parse_judge_winner('{"judging": "B is better", "select": "C"}') == "A"
    assert Coder.parse_judge_winner('{"select": "B"}') == "A"
    assert Coder.parse_judge_winner('["B"]') == "A"
