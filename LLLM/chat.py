from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, Protocol

import click
from loguru import logger


DEFAULT_CHAT_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_CACHE_LENGTH = 16384
LOG_LEVELS = ("trace", "debug", "info", "success", "warning", "error", "critical")


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


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL)


def _configure_cli_logging(verbosity: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=verbosity.upper())


def _build_qwen3_generator(
    repo_id: str,
    *,
    cache_length: int,
    local_files_only: bool,
) -> TextGenerator:
    from .fetch import fetch_model_ir
    from .generator import Generator
    from .qwen3 import Qwen3Model, Qwen3Tokenizer

    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)

    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def generate_chat_response(
    generator: TextGenerator,
    prompt: str,
    *,
    max_generated_token: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> str:
    tokenizer = getattr(generator, "tokenizer", None)
    encode_prompt = getattr(tokenizer, "encode_instruct_prompt", None)
    if encode_prompt is None:
        raise TypeError("generator tokenizer must implement encode_instruct_prompt")

    prompt_tokens = encode_prompt(prompt, enable_thinking=False)
    response = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        include_prompt=False,
    )
    return strip_think_blocks(response).strip()


@click.command(
    help=(
        "Read a prompt from stdin, send it to a Qwen3 chat model, and print the "
        "assistant response without thinking output."
    )
)
@click.option(
    "--model",
    default=DEFAULT_CHAT_MODEL_REPO_ID,
    show_default=True,
    help="Hugging Face repo id or local path for the Qwen3-compatible chat model.",
)
@click.option(
    "--max-generated-token",
    default=1024,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum tokens to generate for the response.",
)
@click.option(
    "--cache-length",
    default=DEFAULT_CACHE_LENGTH,
    show_default=True,
    type=click.IntRange(min=1),
    help="KV cache length used by the generator.",
)
@click.option(
    "--temperature",
    default=0.0,
    show_default=True,
    type=float,
    help="Sampling temperature. Use 0 for greedy decoding.",
)
@click.option(
    "--top-k",
    default=None,
    type=click.IntRange(min=1),
    help="Restrict sampling to the top K tokens.",
)
@click.option(
    "--top-p",
    default=None,
    type=float,
    help="Restrict sampling to nucleus probability P.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    help="Only use models already present in the local Hugging Face cache.",
)
@click.option(
    "--verbosity",
    default="warning",
    show_default=True,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log verbosity.",
)
def chat_cli(
    model: str,
    max_generated_token: int,
    cache_length: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    local_files_only: bool,
    verbosity: str,
) -> None:
    _configure_cli_logging(verbosity)
    prompt = click.get_text_stream("stdin").read()
    if not prompt.strip():
        raise click.UsageError("expected a chat prompt on stdin")

    generator = _build_qwen3_generator(
        model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    response = generate_chat_response(
        generator,
        prompt,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    click.echo(response)
