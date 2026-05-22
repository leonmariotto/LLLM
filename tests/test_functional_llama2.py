from pathlib import Path
import math
from typing import Literal

import pytest
import torch

from ..LLLM.eval import evaluate_base_model_perplexity
from ..LLLM.fetch import fetch_model_ir
from ..LLLM.generator import Generator
from ..LLLM import llama2 as llama2_module
from ..LLLM.llama2 import Llama2Model, Llama2Tokenizer
from ..LLLM.rope import apply_rope

"""
llama2-15m-tinystories evaluation:
Generation produce coherent text.
The WikiText evaluation return :
Loss: 10.4215
Perplexity: 33574.9Which is very high.
But as long as generation produce coherent text we can assume the weight are correctly loaded.
"""
LLAMA2_TINYSTORIES_REPO_ID = "0rn0/llama2-15m-tinystories"


def _load_remote_llama2_tinystories(
    tmp_path: Path,
) -> tuple[Llama2Model, Llama2Tokenizer]:
    ir = fetch_model_ir(LLAMA2_TINYSTORIES_REPO_ID)
    path = Path(str(ir.metadata["path"]))
    tokenizer = Llama2Tokenizer(str(path / "tokenizer.model"))
    model = Llama2Model(Llama2Model.config_from_ir(ir))
    model.load_ir_weights(ir)
    return model, tokenizer


@pytest.mark.slow
def test_functional_llama2_fetch_load_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = _load_remote_llama2_tinystories(tmp_path)

    prompt = "Once upon a time"
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
    )
    generated_text = generator.generate(
        prompt,
        max_generated_token=20,
        stop_at_eos=False,
    )
    print("Generated text : [" + generated_text + "]\n")

    assert len(generated_text) > len(prompt)


@pytest.mark.slow
def test_functional_llama2_compare_rope_layout_logits_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    This test demonstrate that 0rn0/llama2-15m-tinystories is trained with RoPE
    in interleaved layout.
    """
    model, tokenizer = _load_remote_llama2_tinystories(tmp_path)
    prompt = (
        "Once upon a time, there was a little girl named Lily. "
        "She lived in a small house near the forest with her mother and father. "
        "One sunny morning, Lily found a tiny red box under the old tree."
    )
    continuation = (
        " She opened it carefully and saw a shiny key inside. "
        "Lily smiled because she knew it was the start of an adventure."
    )

    def set_rope_layout(layout: Literal["interleaved", "split-half"]) -> None:
        def apply_selected_rope(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            use_interleaved: bool = True,
        ) -> torch.Tensor:
            return apply_rope(
                x,
                cos,
                sin,
                use_interleaved=layout == "interleaved",
            )

        monkeypatch.setattr(llama2_module, "apply_rope", apply_selected_rope)

    def next_token_logits(layout: Literal["interleaved", "split-half"]) -> torch.Tensor:
        set_rope_layout(layout)
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            return model(input_ids)[:, -1, :]

    def continuation_nll(layout: Literal["interleaved", "split-half"]) -> float:
        set_rope_layout(layout)
        prompt_ids = tokenizer.encode(prompt)
        full_ids = prompt_ids + tokenizer.encode(continuation)
        input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
        targets = torch.tensor([full_ids[1:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids)
        start = max(len(prompt_ids) - 1, 0)
        continuation_logits = logits[:, start:, :]
        continuation_targets = targets[:, start:]
        loss = torch.nn.functional.cross_entropy(
            continuation_logits.reshape(-1, continuation_logits.shape[-1]),
            continuation_targets.reshape(-1),
        )
        return float(loss.item())

    def generated_text(layout: Literal["interleaved", "split-half"]) -> str:
        set_rope_layout(layout)
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
        )
        return generator.generate(
            prompt,
            max_generated_token=80,
            stop_at_eos=False,
        )

    interleaved_logits = next_token_logits("interleaved")
    split_half_logits = next_token_logits("split-half")
    max_abs_diff = torch.max(torch.abs(interleaved_logits - split_half_logits)).item()
    interleaved_top_token = int(torch.argmax(interleaved_logits, dim=-1).item())
    split_half_top_token = int(torch.argmax(split_half_logits, dim=-1).item())
    interleaved_nll = continuation_nll("interleaved")
    split_half_nll = continuation_nll("split-half")
    interleaved_text = generated_text("interleaved")
    split_half_text = generated_text("split-half")

    print(f"RoPE layout logits max abs diff: {max_abs_diff:.6f}")
    print(
        "RoPE layout next-token choices: "
        f"interleaved={interleaved_top_token!r} "
        f"split-half={split_half_top_token!r}"
    )
    print(f"Interleaved continuation NLL: {interleaved_nll:.6f}")
    print(f"Split-half continuation NLL: {split_half_nll:.6f}")
    print("Generated text with interleaved RoPE: [" + interleaved_text + "]\n")
    print("Generated text with split-half RoPE: [" + split_half_text + "]\n")

    assert math.isfinite(interleaved_nll)
    assert math.isfinite(split_half_nll)
    assert max_abs_diff > 0.0
    assert len(interleaved_text) > len(prompt)
    assert len(split_half_text) > len(prompt)


@pytest.mark.slow
def test_functional_llama2_fetch_load_generate_raw_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = _load_remote_llama2_tinystories(tmp_path)

    perplexity = evaluate_base_model_perplexity(
        model=model,
        tokenizer=tokenizer,
        limit=2,
        context_length=64,
    )

    assert math.isfinite(perplexity)
    assert perplexity > 0.0
