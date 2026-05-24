from collections.abc import Sequence
from dataclasses import dataclass
import random

from click.testing import CliRunner
import pytest

from ..LLLM import coder as coder_module
from ..LLLM.coder import (
    CodeCandidate,
    CompileResult,
    Coder,
    CoderResult,
    JudgeScore,
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
    assert '"judging": "<brief comparison, pros and cons>"' in prompt
    assert '"select": "<your choice between A and B>"' in prompt


def test_build_score_prompt_contains_original_task_and_selected_program() -> None:
    candidate = CodeCandidate(
        2,
        "candidate prompt",
        "raw",
        "int main(void) { return 0; }",
    )
    coder = Coder(FakeTextGenerator([]), FakeTextGenerator([]))

    prompt = coder._candidate_generator.build_score_prompt(
        "return success",
        candidate,
    )

    assert "return success" in prompt
    assert "int main(void) { return 0; }" in prompt
    assert '"judging": "<brief analysis, pros and cons>"' in prompt
    assert '"score": <integer from 0 to 100>' in prompt


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


def test_coder_cli_reads_stdin_and_prints_selected_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_candidate = CodeCandidate(
        0,
        "prompt",
        "raw",
        "int main(void) { return 0; }",
    )
    captured: dict[str, object] = {}

    class FakeCliCoder:
        def solve(
            self,
            instruction: str,
            *,
            sample_count: int | None = None,
        ) -> CoderResult:
            captured["instruction"] = instruction
            captured["solve_sample_count"] = sample_count
            return CoderResult(
                task=instruction,
                candidates=(selected_candidate,),
                compile_results=(),
                judge_results=(),
                selected_candidate=selected_candidate,
            )

    def fake_build_cli_coder(**kwargs: object) -> FakeCliCoder:
        captured["kwargs"] = kwargs
        return FakeCliCoder()

    monkeypatch.setattr(coder_module, "_build_cli_coder", fake_build_cli_coder)

    result = CliRunner().invoke(
        coder_module.coder_cli,
        ["--sample-count", "1", "--verbosity", "debug"],
        input="write a tiny program\n",
    )

    assert result.exit_code == 0
    assert result.output == "int main(void) { return 0; }\n"
    assert captured["instruction"] == "write a tiny program\n"
    assert captured["solve_sample_count"] == 1
    assert isinstance(captured["kwargs"], dict)
    assert "sample_count" not in captured["kwargs"]


def test_coder_solve_accepts_per_call_sample_count() -> None:
    code_generator = FakeTextGenerator(
        [
            "int main(void) { return 0; }",
            "int main(void) { return 1; }",
        ]
    )
    coder = Coder(
        code_generator,
        FakeTextGenerator(
            [
                '{"judging": "candidate 0 wins", "select": "A"}',
                '{"judging": "solves the task", "score": 88}',
            ]
        ),
        sample_count=5,
        compile_runner=lambda _command: FakeCompileProcess(returncode=0),
        rng=random.Random(0),
    )

    result = coder.solve("return a status code", sample_count=2)

    assert len(result.candidates) == 2
    assert len(code_generator.tokenizer.prompts) == 2


def test_coder_scores_selected_candidate_after_selection() -> None:
    code_generator = FakeTextGenerator(
        [
            "int main(void) { return 0; }",
            "int main(void) { return 1; }",
        ]
    )
    judge_generator = FakeTextGenerator(
        [
            '{"judging": "candidate 1 wins", "select": "B"}',
            '{"judging": "handles the requested status", "score": 91}',
        ]
    )
    coder = Coder(
        code_generator,
        judge_generator,
        sample_count=2,
        compile_runner=lambda _command: FakeCompileProcess(returncode=0),
    )

    result = coder.solve("return a status code")

    assert result.selected_candidate.index == 1
    assert len(judge_generator.tokenizer.prompts) == 2
    assert "Program A:" in judge_generator.tokenizer.prompts[0]
    assert "Program:" in judge_generator.tokenizer.prompts[1]
    assert "return a status code" in judge_generator.tokenizer.prompts[1]
    assert "int main(void) { return 1; }" in judge_generator.tokenizer.prompts[1]
    assert judge_generator.tokenizer.enable_thinking == [False, False]


def test_score_selected_candidate_returns_parsed_score() -> None:
    judge_generator = FakeTextGenerator(
        ['{"judging": "mostly correct", "score": 73}']
    )
    coder = Coder(
        FakeTextGenerator([]),
        judge_generator,
        judge_max_generated_token=77,
    )
    candidate = CodeCandidate(
        0,
        "prompt",
        "raw",
        "int main(void) { return 0; }",
    )

    judge_score = coder.score_selected_candidate("return success", candidate)

    assert judge_score == JudgeScore(reason="mostly correct", score=73)
    assert judge_generator.tokenizer.enable_thinking == [False]
    assert judge_generator.calls == [
        {
            "max_generated_token": 77,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "include_prompt": False,
        }
    ]


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
    assert judge_generator.tokenizer.enable_thinking == [False, False]
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


def test_coder_tournament_pairs_candidates_by_round() -> None:
    candidates = (
        CodeCandidate(0, "prompt", "raw", "source 0"),
        CodeCandidate(1, "prompt", "raw", "source 1"),
        CodeCandidate(2, "prompt", "raw", "source 2"),
        CodeCandidate(3, "prompt", "raw", "source 3"),
    )
    judge_generator = FakeTextGenerator(
        [
            '{"judging": "candidate 1 wins", "select": "B"}',
            '{"judging": "candidate 2 wins", "select": "A"}',
            '{"judging": "candidate 2 wins final", "select": "B"}',
        ]
    )
    coder = Coder(FakeTextGenerator([]), judge_generator)
    compile_results = tuple(
        CompileResult(
            candidate_index=candidate.index,
            success=True,
            command=("gcc",),
            returncode=0,
            stdout="",
            stderr="",
        )
        for candidate in candidates
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
        (0, 1, 1),
        (2, 3, 2),
        (1, 2, 2),
    ]
    assert selected.index == 2


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


def test_parse_judge_score_accepts_valid_json() -> None:
    assert Coder.parse_judge_score(
        '{"judging": "bad", "score": 0}'
    ) == JudgeScore(reason="bad", score=0)
    assert Coder.parse_judge_score(
        '{"judging": "perfect", "score": 100}'
    ) == JudgeScore(reason="perfect", score=100)
    assert (
        Coder.parse_judge_score(
            '{"judging": "draft", "score": 12}\n'
            '{"judging": "final", "score": 87}'
        )
        == JudgeScore(reason="final", score=87)
    )


def test_parse_judge_score_rejects_invalid_json() -> None:
    assert Coder.parse_judge_score("score: 50") is None
    assert Coder.parse_judge_score('{"judging": "bad", "score": -1}') is None
    assert Coder.parse_judge_score('{"judging": "bad", "score": 101}') is None
    assert Coder.parse_judge_score('{"judging": "bad", "score": true}') is None
    assert Coder.parse_judge_score('{"score": 50}') is None
    assert Coder.parse_judge_score('{"judging": "missing"}') is None
