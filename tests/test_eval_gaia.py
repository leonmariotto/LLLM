from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pytest

from ..LLLM import eval_gaia
from ..LLLM.eval_gaia import (
    GaiaTask,
    evaluate_gaia_agent,
    export_gaia_predictions,
    load_gaia_tasks,
    normalize_gaia_answer,
    score_gaia_answer,
)


class TinyDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.shuffle_seeds: list[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)

    def select(self, selected: range) -> "TinyDataset":
        dataset = TinyDataset([self.rows[index] for index in selected])
        dataset.shuffle_seeds = list(self.shuffle_seeds)
        return dataset

    def shuffle(self, *, seed: int) -> "TinyDataset":
        self.shuffle_seeds.append(seed)
        dataset = TinyDataset(list(reversed(self.rows)))
        dataset.shuffle_seeds = list(self.shuffle_seeds)
        return dataset


def _row(
    task_id: str,
    *,
    question: str = "Question?",
    level: int = 1,
    final_answer: object = "Paris",
    file_name: object = "",
    file_path: object = "",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "Question": question,
        "Level": level,
        "Final answer": final_answer,
        "file_name": file_name,
        "file_path": file_path,
        "Annotator Metadata": {"source": "unit-test", "Tools": "",
                               "How long did this take?": ""},
    }


def _patch_dataset(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> TinyDataset:
        dataset = TinyDataset(rows)
        calls.append({"args": args, "kwargs": kwargs, "dataset": dataset})
        return dataset

    monkeypatch.setattr(eval_gaia, "load_dataset", fake_load_dataset)
    return calls


def test_load_gaia_tasks_selects_all_config_and_resolves_file_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _patch_dataset(
        monkeypatch,
        [
            _row(
                "task-1",
                level=2,
                file_name="document.pdf",
                file_path="2023/validation/document.pdf",
            )
        ],
    )

    tasks = load_gaia_tasks(data_dir=tmp_path, split="validation")

    assert calls[0]["args"] == (str(tmp_path.resolve()), "2023_all")
    assert calls[0]["kwargs"] == {"split": "validation"}
    assert tasks == [
        GaiaTask(
            task_id="task-1",
            question="Question?",
            level=2,
            file_path=(tmp_path / "2023/validation/document.pdf").resolve(),
            file_name="document.pdf",
            metadata={"source": "unit-test", "Tools": "",
                               "How long did this take?": ""},
            expected_answer="Paris",
        )
    ]


@pytest.mark.parametrize(
    ("level", "config_name"),
    [(1, "2023_level1"), (2, "2023_level2"), (3, "2023_level3")],
)
def test_load_gaia_tasks_selects_level_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    level: eval_gaia.GaiaLevel,
    config_name: str,
) -> None:
    calls = _patch_dataset(monkeypatch, [_row("task-1", level=level)])

    load_gaia_tasks(data_dir=tmp_path, level=level)

    assert calls[0]["args"] == (str(tmp_path.resolve()), config_name)


def test_load_gaia_tasks_uses_snapshot_download_when_data_dir_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _patch_dataset(monkeypatch, [_row("task-1")])

    def fake_snapshot_download(*, repo_id: str, repo_type: str) -> str:
        assert repo_id == "gaia-benchmark/GAIA"
        assert repo_type == "dataset"
        return str(tmp_path)

    monkeypatch.setattr(eval_gaia, "snapshot_download", fake_snapshot_download)

    load_gaia_tasks()

    assert calls[0]["args"][0] == str(tmp_path.resolve())


def test_load_gaia_tasks_applies_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1"), _row("task-2")])

    tasks = load_gaia_tasks(data_dir=tmp_path, limit=1)

    assert [task.task_id for task in tasks] == ["task-1"]


def test_load_gaia_tasks_can_shuffle_before_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row("task-1"), _row("task-2")]
    calls = _patch_dataset(monkeypatch, rows)

    tasks = load_gaia_tasks(data_dir=tmp_path, shuffle=True, limit=1)

    assert [task.task_id for task in tasks] == ["task-2"]
    assert calls[0]["dataset"].shuffle_seeds == [0]


def test_evaluate_gaia_agent_scores_validation_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(
        monkeypatch,
        [
            _row("task-1", level=1, final_answer="Paris"),
            _row("task-2", level=2, final_answer="London"),
        ],
    )

    def agent(task: GaiaTask) -> str:
        return "FINAL ANSWER: Paris" if task.task_id == "task-1" else "Rome"

    evaluation = evaluate_gaia_agent(agent, data_dir=tmp_path)

    assert evaluation.total_tasks == 2
    assert evaluation.scored_tasks == 2
    assert evaluation.correct_tasks == 1
    assert evaluation.overall_accuracy == 0.5
    assert evaluation.per_level_accuracy == {1: 1.0, 2: 0.0}
    assert [result.correct for result in evaluation.results] == [True, False]


def test_evaluate_gaia_agent_passes_shuffle_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _patch_dataset(monkeypatch, [_row("task-1"), _row("task-2")])

    evaluation = evaluate_gaia_agent(
        lambda task: task.expected_answer or "",
        data_dir=tmp_path,
        shuffle=True,
        shuffle_seed=7,
        limit=1,
    )

    assert [result.task_id for result in evaluation.results] == ["task-2"]
    assert calls[0]["dataset"].shuffle_seeds == [7]


def test_evaluate_gaia_agent_keeps_test_rows_unscored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1", final_answer=None)])

    evaluation = evaluate_gaia_agent(
        lambda task: "answer",
        data_dir=tmp_path,
        split="test",
    )

    assert evaluation.scored_tasks == 0
    assert evaluation.correct_tasks == 0
    assert evaluation.overall_accuracy is None
    assert evaluation.results[0].correct is None
    assert evaluation.results[0].expected_answer is None


def test_evaluate_gaia_agent_captures_agent_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1", final_answer="Paris")])

    def failing_agent(task: GaiaTask) -> str:
        raise RuntimeError(f"cannot solve {task.task_id}")

    evaluation = evaluate_gaia_agent(failing_agent, data_dir=tmp_path)

    assert evaluation.results[0].prediction == ""
    assert evaluation.results[0].correct is False
    assert evaluation.results[0].error == "cannot solve task-1"


def test_evaluate_gaia_agent_writes_full_jsonl_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1", final_answer="Paris")])
    output_path = tmp_path / "results.jsonl"

    evaluate_gaia_agent(lambda task: "Paris", data_dir=tmp_path, output_path=output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["task_id"] == "task-1"
    assert rows[0]["prediction"] == "Paris"
    assert rows[0]["expected_answer"] == "Paris"
    assert rows[0]["correct"] is True
    assert "elapsed_seconds" in rows[0]


def test_export_gaia_predictions_writes_task_id_and_answer_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_dataset(monkeypatch, [_row("task-1", final_answer="Paris")])
    evaluation = evaluate_gaia_agent(lambda task: "Paris", data_dir=tmp_path)
    output_path = tmp_path / "predictions.jsonl"

    export_gaia_predictions(evaluation, output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [{"answer": "Paris", "task_id": "task-1"}]


def test_normalize_gaia_answer_handles_final_answer_prefix() -> None:
    assert normalize_gaia_answer("  FINAL ANSWER:  New   York.  ") == "new york"


def test_score_gaia_answer_accepts_exact_and_contained_answers() -> None:
    assert score_gaia_answer("paris", "Paris")
    assert score_gaia_answer("The answer is Paris.", "Paris")
    assert not score_gaia_answer("London", "Paris")
