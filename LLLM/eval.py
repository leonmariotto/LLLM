from dataclasses import dataclass
from typing import Protocol, Callable, Any, List, cast
import re
import torch

from utils import generate_text_simple
from datasets import load_dataset


# Typing class
class TensorModel(Protocol):
    def eval(self) -> Any: ...

    def __call__(self, idx: torch.Tensor) -> torch.Tensor: ...


class Tokenizer(Protocol):
    def encode(self, input: str) -> List[int]: ...

    def decode(self, tok: List[int]) -> str: ...


@dataclass
class DatasetAdapter:
    dataset_id: str
    config: str | None
    split: str

    build_prompt: Callable[[dict], str]
    extract_expected: Callable[[dict], Any]
    extract_prediction: Callable[[str], Any]
    score: Callable[[Any, Any], bool]


def normalize_text(text: str) -> str:
    return text.strip().lower()


def extract_last_number(text: str) -> str:
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def evaluate_instructions_model(
    model: TensorModel,
    tokenizer: Tokenizer,
    adapter: DatasetAdapter,
    limit: int = 20,
):
    dataset_args = [adapter.dataset_id]
    if adapter.config is not None:
        dataset_args.append(adapter.config)

    dataset = load_dataset(*dataset_args, split=adapter.split)
    dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    total = 0

    for row in dataset:
        prompt = adapter.build_prompt(row)
        prompt_tokens = tokenizer.encode(prompt)
        input_idx = torch.tensor([prompt_tokens], dtype=torch.long)
        raw_prediction_idx = generate_text_simple(
            model, input_idx, max_new_tokens=20, context_size=1024
        )
        raw_prediction_tokens = cast(
            list[int], cast(Any, raw_prediction_idx.squeeze(0)).tolist()
        )
        raw_prediction_text = tokenizer.decode(raw_prediction_tokens)

        prediction = adapter.extract_prediction(raw_prediction_text)
        expected = adapter.extract_expected(row)

        ok = adapter.score(prediction, expected)

        correct += int(ok)
        total += 1

        print("=" * 80)
        print("Prompt:", prompt)
        print("Expected:", expected)
        print("Prediction:", prediction)
        print("Raw output:", raw_prediction_text)
        print("Correct:", ok)

    accuracy = correct / total if total else 0

    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")

    return accuracy


gsm8k_adapter = DatasetAdapter(
    dataset_id="gsm8k",
    config="main",
    split="test",
    build_prompt=lambda row: (
        "Solve the following math problem. "
        "Return only the final numeric answer.\n\n"
        f"Problem: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=lambda row: extract_last_number(row["answer"]),
    extract_prediction=lambda text: extract_last_number(text),
    score=lambda prediction, expected: prediction == expected,
)


def boolq_prediction(text: str) -> bool | None:
    text = normalize_text(text)

    if text.startswith("yes") or "answer: yes" in text:
        return True

    if text.startswith("no") or "answer: no" in text:
        return False

    return None


boolq_adapter = DatasetAdapter(
    dataset_id="boolq",
    config=None,
    split="validation",
    build_prompt=lambda row: (
        "Read the passage and answer the question with only yes or no.\n\n"
        f"Passage: {row['passage']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=lambda row: row["answer"],
    extract_prediction=boolq_prediction,
    score=lambda prediction, expected: prediction == expected,
)


def squad_score(prediction: str, expected_answers: list[str]) -> bool:
    prediction = normalize_text(prediction)

    return any(normalize_text(answer) in prediction for answer in expected_answers)


squad_adapter = DatasetAdapter(
    dataset_id="squad",
    config=None,
    split="validation",
    build_prompt=lambda row: (
        "Answer the question using only the context below.\n\n"
        f"Context: {row['context']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    ),
    extract_expected=lambda row: row["answers"]["text"],
    extract_prediction=lambda text: text.strip(),
    score=squad_score,
)
