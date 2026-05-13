from collections.abc import Iterator

import torch

from ..LLLM import eval as gpt_eval
from ..LLLM.eval import DatasetAdapter, evaluate_instructions_model


class TinyDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self.rows)

    def select(self, selected: range) -> "TinyDataset":
        return TinyDataset([self.rows[index] for index in selected])


class MockTokenizer:
    def encode(self, input: str) -> list[int]:
        return [2] if input == "Every effort moves the project forward." else [0]

    def decode(self, tok: list[int]) -> str:
        values = {
            1: "ok",
            2: "Every effort moves the project forward.",
        }
        return "".join(values.get(token, "") for token in tok)


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*idx.shape, 3), device=idx.device)
        logits[:, -1, 1] = 1.0
        return logits


def test_evaluate_instructions_model_scores_generated_completion_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gpt_eval,
        "load_dataset",
        lambda *args, **kwargs: TinyDataset(
            [{"prompt": "Every effort moves the project forward.", "expected": "ok"}]
        ),
    )

    adapter = DatasetAdapter(
        dataset_id="local-eval-dataset",
        config=None,
        split="test",
        build_prompt=lambda row: row["prompt"],
        extract_expected=lambda row: row["expected"],
        extract_prediction=lambda text: (
            "prompt leaked"
            if "Every effort moves the project forward." in text
            else text
        ),
        score=lambda prediction, expected: prediction == expected,
    )

    accuracy = evaluate_instructions_model(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        adapter=adapter,
        limit=1,
        max_generated_token=1,
        context_size=8,
    )

    assert accuracy == 1.0
