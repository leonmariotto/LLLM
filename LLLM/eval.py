"""
Models evaluations module.
Support :
    - raw model evaluation with WikiText-based test (measure perplexity).
    - instruction-tuned models evaluation dataset : boolq, gsm8k and squad.
"""

import math
from dataclasses import dataclass
from collections.abc import Mapping
import importlib
from typing import Any, Callable, Protocol, TypeVar, cast
import re
import torch
import torch.nn.functional as F

from .generator import Generator
from .kv_cache import KVCache

# Make pyright happy
load_dataset = cast(
    Callable[..., Any], importlib.import_module("datasets").load_dataset
)


# Typing class
class TensorModel(Protocol):
    """Model contract used by evaluation and text generation."""

    def eval(self) -> Any: ...

    def __call__(self, idx: torch.Tensor) -> torch.Tensor: ...


class CachedTensorModel(Protocol):
    """Model contract used by cached instruction generation."""

    def eval(self) -> Any: ...

    def __call__(
        self, idx: torch.Tensor, *, kv_cache: KVCache | None = None
    ) -> torch.Tensor: ...


class Tokenizer(Protocol):
    """Tokenizer contract for converting between text and token ids."""

    def encode(self, input: str) -> list[int]: ...

    def decode(self, tok: list[int]) -> str: ...

    def get_eos(self) -> int | None: ...


DatasetRow = Mapping[str, Any]
TExpected = TypeVar("TExpected")
TPrediction = TypeVar("TPrediction")


@dataclass
class DatasetAdapter[TExpected, TPrediction]:
    """
    Dataset-specific evaluation hooks.

    Args:
        dataset_id: Hugging Face dataset identifier.
        config: Optional dataset configuration name.
        split: Dataset split to evaluate.
        build_prompt: Converts a dataset row into a model prompt.
        extract_expected: Extracts the expected answer from a dataset row.
        encode_prompt: Optional tokenizer-aware prompt encoder. Use this for
            chat/instruct formats that require special token ids around the text.
    """

    dataset_id: str
    config: str | None
    split: str

    build_prompt: Callable[[DatasetRow], str]
    extract_expected: Callable[[DatasetRow], TExpected]
    extract_prediction: Callable[[str], TPrediction]
    score: Callable[[TPrediction, TExpected], bool]
    encode_prompt: Callable[[Tokenizer, str], list[int]] | None = None


def normalize_text(text: str) -> str:
    """Strip surrounding whitespace and lowercase text for loose matching."""
    return text.strip().lower()


def extract_last_number(text: str) -> str:
    """
    Extract the final integer or decimal number from text.

    Returns an empty string when no number is present.
    """
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def strip_think_blocks(text: str) -> str:
    """
    Remove Qwen-style hidden thinking spans from generated text.

    Complete ``<think>...</think>`` blocks are stripped across lines. If a model
    starts a trailing ``<think>`` block and never closes it, the trailing content
    is dropped.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)


def evaluate_instructions_model(
    model: CachedTensorModel,
    tokenizer: Tokenizer,
    adapter: DatasetAdapter[TExpected, TPrediction],
    limit: int = 20,
    max_generated_token: int = 20,
    cache_length: int = 4096,
) -> float:
    """
    Evaluate an instruction-style model on a Hugging Face dataset.

    Args:
        model: Autoregressive model returning logits for token ids.
        tokenizer: Tokenizer paired with the model.
        adapter: Dataset adapter defining prompt construction and scoring.
        limit: Maximum number of dataset rows to evaluate.
        max_generated_token: Maximum number of completion tokens per row.
        cache_length: KV cache length passed to the generator.

    Returns:
        Accuracy as ``correct / total``. The evaluator prints each prompt,
        expected value, parsed prediction, raw completion, and final accuracy.
    """
    dataset_args = [adapter.dataset_id]
    if adapter.config is not None:
        dataset_args.append(adapter.config)

    dataset = load_dataset(*dataset_args, split=adapter.split)
    dataset = dataset.select(range(min(limit, len(dataset))))

    model.eval()
    generator = Generator(model, tokenizer, cache_length=cache_length)

    correct = 0
    total = 0

    for raw_row in dataset:
        row = cast(DatasetRow, raw_row)
        prompt = adapter.build_prompt(row)
        if adapter.encode_prompt is None:
            raw_prediction_text = generator.generate(
                prompt,
                max_generated_token=max_generated_token,
                include_prompt=False,
            )
        else:
            prompt_tokens = adapter.encode_prompt(tokenizer, prompt)
            raw_prediction_text = generator.generate_from_tokens(
                prompt_tokens,
                max_generated_token=max_generated_token,
                include_prompt=False,
            )

        prediction_text = strip_think_blocks(raw_prediction_text)
        prediction = adapter.extract_prediction(prediction_text)
        expected = adapter.extract_expected(row)

        ok = adapter.score(prediction, expected)

        correct += int(ok)
        total += 1

        print("=" * 80)
        print("Prompt:", prompt)
        print("Expected:", expected)
        print("Prediction:", prediction)
        print("Raw output:", prediction_text)
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
    """
    Parse yes/no completion text for BoolQ.

    Returns ``True`` for yes, ``False`` for no, and ``None`` when unclear.
    """
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
    """Score SQuAD by checking whether any expected answer appears in prediction."""
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


def _model_device(model: TensorModel) -> torch.device:
    """Return the device used by the model in parameters."""
    if not isinstance(model, torch.nn.Module):
        return torch.device("cpu")

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_base_model_perplexity(
    model: TensorModel,
    tokenizer: Tokenizer,
    limit: int = 100,
    context_length: int = 1024,
) -> float:
    """
    Evaluate next-token perplexity on WikiText-2 validation text.

    Args:
        model: Autoregressive model returning logits for token ids.
        tokenizer: Tokenizer paired with the model.
        limit: Maximum number of non-empty dataset rows to evaluate.
        context_length: Maximum token window evaluated from each row.

    Returns:
        Perplexity, computed as ``exp(mean_cross_entropy_loss)`` over target
        tokens. Rows with fewer than two tokens are skipped.
    """
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        # Can use percentage slicing at load time : split="train[:5%]",
        split="validation",
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))

    model.eval()
    device = _model_device(model)
    total_loss = 0.0
    total_tokens = 0

    for raw_row in ds:
        row = cast(DatasetRow, raw_row)
        text = row["text"]
        if not isinstance(text, str):
            continue

        inputs = tokenizer.encode(text.strip())
        if len(inputs) < 2:
            continue
        inputs = inputs[-context_length:]
        input_idx = torch.tensor(
            [inputs],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = model(input_idx)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_idx[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="sum",
        )
        target_token_count = shift_targets.numel()
        total_loss += float(loss.item())
        total_tokens += target_token_count

    if total_tokens == 0:
        raise ValueError("no rows with at least two tokens were available")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity
