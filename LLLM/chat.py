"""
Chat application !

Include a fancy TUI based on textual lib and rich.Text.
All history is reinjected at each request.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable, Protocol, TypedDict, cast

import click
from loguru import logger
from rich.text import Text

from .vector_db import DEFAULT_EMBEDDING_MODEL, VectorDB
from .vector_search import SearchResult, TextEmbedder
from .utils import get_device


DEFAULT_CHAT_MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_CACHE_LENGTH = 16384
LOG_LEVELS = ("trace", "debug", "info", "success", "warning", "error", "critical")


class ChatMessage(TypedDict):
    role: str
    content: str


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


@dataclass(frozen=True)
class ChatGenerationOptions:
    max_generated_token: int
    temperature: float
    top_k: int | None
    top_p: float | None
    enable_thinking: bool


@dataclass(frozen=True)
class ChatRAGOptions:
    score_cutoff: float
    max_entries: int


@dataclass(frozen=True)
class ChatRAGContext:
    vector_db: VectorDB
    options: ChatRAGOptions


@dataclass(frozen=True)
class ChatStatus:
    model_bytes: int
    cache_length: int
    absolute_position: int
    context_bytes: int


def strip_think_blocks(text: str) -> str:
    """Remove complete or unterminated Qwen thinking blocks from model output."""
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
    """Load a Qwen3 model/tokenizer pair and wrap it in the shared generator."""
    from .fetch import fetch_model_ir
    from .generator import Generator
    from .qwen3 import Qwen3Model, Qwen3Tokenizer

    ir = fetch_model_ir(repo_id, local_files_only=local_files_only)
    cfg = Qwen3Model.config_from_ir(ir)
    path = Path(str(ir.metadata["path"]))

    tokenizer = Qwen3Tokenizer(str(path / "tokenizer.json"))
    model = Qwen3Model(cfg)
    model.load_ir_weights(ir)
    device = get_device()
    model.to(device)
    # It appear that the model generate gibberish if dtype set here.
    # model.to(device, dtype=torch.float16)

    return Generator(model=model, tokenizer=tokenizer, cache_length=cache_length)


def _build_rag_embedder(model: str) -> TextEmbedder:
    from .fetch import fetch_embedding_model_ir
    from .sentence_transformer import SentenceTransformerEmbedder

    ir = fetch_embedding_model_ir(model)
    return SentenceTransformerEmbedder.from_ir(ir)


def _build_rag_context(
    *,
    vector_db_path: Path,
    embedding_model: str,
    score_cutoff: float,
    max_entries: int,
) -> ChatRAGContext:
    logger.info(
        "Loading RAG VectorDB: path={}, embedding_model={}, score_cutoff={}, max_entries={}",
        vector_db_path,
        embedding_model,
        score_cutoff,
        max_entries,
    )
    embedder = _build_rag_embedder(embedding_model)
    vector_db = VectorDB.load(vector_db_path, embedder)
    return ChatRAGContext(
        vector_db=vector_db,
        options=ChatRAGOptions(score_cutoff=score_cutoff, max_entries=max_entries),
    )


def _format_rag_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{index}] {result.sequence}" for index, result in enumerate(results, start=1)
    )


def _augment_prompt_with_rag(prompt: str, rag_context: ChatRAGContext | None) -> str:
    if rag_context is None:
        return prompt

    results = rag_context.vector_db.search(
        prompt,
        top_k=rag_context.options.max_entries,
    )
    filtered_results = [
        result for result in results if result.score >= rag_context.options.score_cutoff
    ]
    logger.info(
        "RAG search completed: results={}, accepted={}",
        len(results),
        len(filtered_results),
    )
    for i, result in enumerate(filtered_results):
        logger.debug("RAG result {} (score={}): {}", i, result.score, result.sequence)
    if not filtered_results:
        return prompt

    return (
        "Relevant context:\n"
        f"{_format_rag_context(filtered_results)}\n\n"
        "User question:\n"
        f"{prompt}"
    )


def _augment_messages_with_rag(
    messages: list[ChatMessage],
    rag_context: ChatRAGContext | None,
) -> list[ChatMessage]:
    if rag_context is None or not messages:
        return messages

    augmented_messages: list[ChatMessage] = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    latest_user_index = next(
        (
            index
            for index in range(len(augmented_messages) - 1, -1, -1)
            if augmented_messages[index]["role"] == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return augmented_messages

    latest_user_message = augmented_messages[latest_user_index]
    latest_user_message["content"] = _augment_prompt_with_rag(
        latest_user_message["content"],
        rag_context,
    )
    return augmented_messages


def _encode_chat_messages(
    generator: TextGenerator,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool = True,
) -> list[int]:
    """Encode full chat history with the tokenizer's chat template."""
    tokenizer = getattr(generator, "tokenizer", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise TypeError("generator tokenizer must implement apply_chat_template")

    encoded = apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(encoded, dict):
        raise TypeError("expected tokenized chat template output")
    encoded_dict = cast(dict[str, Any], encoded)
    input_ids = encoded_dict.get("input_ids")
    if not isinstance(input_ids, list):
        raise TypeError("expected input_ids to be a list[int]")
    input_tokens = cast(list[object], input_ids)
    if not all(isinstance(token, int) for token in input_tokens):
        raise TypeError("expected input_ids to be a list[int]")
    return cast(list[int], input_tokens)


def _generate_chat_messages_response(
    generator: TextGenerator,
    messages: list[ChatMessage],
    *,
    max_generated_token: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    enable_thinking: bool = True,
    rag_context: ChatRAGContext | None = None,
) -> str:
    """Generate one assistant response from the complete structured history."""
    messages = _augment_messages_with_rag(messages, rag_context)
    prompt_tokens = _encode_chat_messages(
        generator,
        messages,
        enable_thinking=enable_thinking,
    )
    _raise_if_context_overflows(generator, len(prompt_tokens))
    response = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        include_prompt=False,
    )
    return strip_think_blocks(response).strip()


def generate_chat_response(
    generator: TextGenerator,
    prompt: str,
    *,
    max_generated_token: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    enable_thinking: bool = True,
    rag_context: ChatRAGContext | None = None,
) -> str:
    """Generate a one-shot response from a single user prompt string."""
    prompt = _augment_prompt_with_rag(prompt, rag_context)
    tokenizer = getattr(generator, "tokenizer", None)
    encode_prompt = getattr(tokenizer, "encode_instruct_prompt", None)
    if encode_prompt is None:
        raise TypeError("generator tokenizer must implement encode_instruct_prompt")

    prompt_tokens = encode_prompt(prompt, enable_thinking=enable_thinking)
    response = generator.generate_from_tokens(
        prompt_tokens,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        include_prompt=False,
    )
    return strip_think_blocks(response).strip()


def _raise_if_context_overflows(generator: TextGenerator, token_count: int) -> None:
    """Reject prompts that exceed the model's configured context length."""
    context_length = _model_context_length(generator)
    if context_length is not None and token_count > context_length:
        raise ValueError(
            f"conversation context is {token_count} tokens, "
            f"but the model context length is {context_length}"
        )


def _model_context_length(generator: TextGenerator) -> int | None:
    model = getattr(generator, "model", None)
    value = getattr(model, "context_length", None)
    if isinstance(value, int):
        return value
    return None


def _estimate_model_bytes(generator: TextGenerator) -> int:
    """Estimate loaded model tensor memory from parameters and buffers."""
    model = getattr(generator, "model", None)
    if model is None:
        return 0

    total = 0
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in cast(Iterable[Any], parameters()):
            total += int(parameter.numel()) * int(parameter.element_size())

    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        for buffer in cast(Iterable[Any], buffers()):
            total += int(buffer.numel()) * int(buffer.element_size())

    return total


def _first_model_tensor(generator: TextGenerator) -> Any | None:
    model = getattr(generator, "model", None)
    if model is None:
        return None

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in cast(Iterable[Any], parameters()):
            return parameter

    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        for buffer in cast(Iterable[Any], buffers()):
            return buffer

    return None


def _estimate_context_bytes(
    generator: TextGenerator,
    *,
    cache_length: int,
    absolute_position: int,
) -> int:
    """Estimate retained KV-cache bytes for the current absolute token position."""
    model = getattr(generator, "model", None)
    blocks = getattr(model, "trf_blocks", None)
    if blocks is None:
        return 0

    try:
        n_layers = len(blocks)
        first_block = blocks[0]
    except (TypeError, IndexError):
        return 0

    attention = getattr(first_block, "att", None)
    n_kv_groups = getattr(attention, "num_kv_groups", None)
    head_dim = getattr(attention, "head_dim", None)
    tensor = _first_model_tensor(generator)
    if (
        not isinstance(n_kv_groups, int)
        or not isinstance(head_dim, int)
        or tensor is None
    ):
        return 0

    retained_tokens = min(absolute_position, cache_length)
    dtype_size = int(tensor.element_size())
    return n_layers * 2 * retained_tokens * n_kv_groups * head_dim * dtype_size


def _chat_status(
    generator: TextGenerator,
    messages: list[ChatMessage],
    *,
    cache_length: int,
    enable_thinking: bool = True,
) -> ChatStatus:
    """Build the status-bar metrics from current history and model structure."""
    absolute_position = (
        len(
            _encode_chat_messages(
                generator,
                messages,
                enable_thinking=enable_thinking,
            )
        )
        if messages
        else 0
    )
    return ChatStatus(
        model_bytes=_estimate_model_bytes(generator),
        cache_length=cache_length,
        absolute_position=absolute_position,
        context_bytes=_estimate_context_bytes(
            generator,
            cache_length=cache_length,
            absolute_position=absolute_position,
        ),
    )


def _format_bytes(byte_count: int) -> str:
    if byte_count <= 0:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(byte_count)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _format_status(status: ChatStatus) -> str:
    return (
        f"Model {_format_bytes(status.model_bytes)} | "
        f"history_length/cache_length : "
        f"{status.absolute_position}/{status.cache_length} | "
        f"Context {status.absolute_position} tok abs / "
        f"{_format_bytes(status.context_bytes)} est"
    )


def _stdin_is_interactive() -> bool:
    return bool(click.get_text_stream("stdin").isatty())


def _run_textual_chat_app(
    generator: TextGenerator,
    *,
    cache_length: int,
    options: ChatGenerationOptions,
    rag_context: ChatRAGContext | None = None,
) -> None:
    """Run the full-screen TUI and keep all turns in the model context."""
    from textual import events
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.widgets import RichLog, Static, TextArea

    class ChatInput(TextArea):
        def __init__(self, on_submit: Callable[[str], None]) -> None:
            super().__init__(
                "",
                id="chat-input",
                soft_wrap=True,
                show_line_numbers=False,
                placeholder="Message",
            )
            self._on_submit = on_submit

        def on_key(self, event: events.Key) -> None:
            """Submit on Enter; reserve Shift+Enter/Ctrl+J for multiline input."""
            if event.key in {"shift+enter", "ctrl+j"}:
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                text = self.text
                if text.strip():
                    self.load_text("")
                    self._on_submit(text)

    class ChatApp(App[None]):
        TITLE = "LLLM Chat"
        CSS = """
        Screen {
            layout: vertical;
            background: #101216;
        }

        #status {
            height: 1;
            color: #d7e0ea;
            background: #26313f;
            padding: 0 1;
        }

        #conversation {
            height: 1fr;
            padding: 1 2;
            color: #e6edf3;
            scrollbar-color: #6f7d8c;
            scrollbar-color-hover: #9fb0c2;
        }

        #input-container {
            height: 6;
            border-top: solid #3a4553;
            padding: 0 1;
        }

        #chat-input {
            height: 5;
            border: none;
            background: #151a21;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.messages: list[ChatMessage] = []
            self.status_widget: Static | None = None
            self.conversation_widget: RichLog | None = None
            self.input_widget: ChatInput | None = None

        def compose(self) -> ComposeResult:
            self.status_widget = Static("", id="status")
            self.conversation_widget = RichLog(
                id="conversation",
                auto_scroll=True,
                wrap=True,
                highlight=False,
                markup=False,
                min_width=20,
            )
            self.input_widget = ChatInput(self.submit_message)
            yield self.status_widget
            yield self.conversation_widget
            with Container(id="input-container"):
                yield self.input_widget

        def on_mount(self) -> None:
            if self.input_widget is not None:
                self.input_widget.focus()
            self.refresh_chat()

        def on_resize(self, event: events.Resize) -> None:
            self.scroll_conversation_end()

        def submit_message(self, content: str) -> None:
            self.messages.append({"role": "user", "content": content})
            self._set_input_disabled(True)
            self.refresh_chat()
            self.run_worker(self._generate_response, thread=True, exclusive=True)

        def _generate_response(self) -> None:
            try:
                response = _generate_chat_messages_response(
                    generator,
                    self.messages,
                    max_generated_token=options.max_generated_token,
                    temperature=options.temperature,
                    top_k=options.top_k,
                    top_p=options.top_p,
                    enable_thinking=options.enable_thinking,
                    rag_context=rag_context,
                )
            except Exception as exc:  # pragma: no cover - visible in app.
                self.call_from_thread(self._finish_response, f"Error: {exc}", True)
                return
            self.call_from_thread(self._finish_response, response, False)

        def _finish_response(self, response: str, is_error: bool) -> None:
            role = "system" if is_error else "assistant"
            self.messages.append({"role": role, "content": response})
            self._set_input_disabled(False)
            self.refresh_chat()

        def _set_input_disabled(self, disabled: bool) -> None:
            if self.input_widget is not None:
                self.input_widget.disabled = disabled
                if not disabled:
                    self.input_widget.focus()

        def refresh_chat(self) -> None:
            """Redraw status and transcript, then pin the transcript to the end."""
            if self.status_widget is not None:
                status = _chat_status(
                    generator,
                    self.messages,
                    cache_length=cache_length,
                    enable_thinking=options.enable_thinking,
                )
                self.status_widget.update(_format_status(status))

            if self.conversation_widget is not None:
                self.conversation_widget.clear()
                for message in self.messages:
                    self.conversation_widget.write(
                        _message_renderable(message),
                        scroll_end=True,
                        animate=False,
                    )
                self.scroll_conversation_end()

        def scroll_conversation_end(self) -> None:
            if self.conversation_widget is not None:
                self.conversation_widget.scroll_end(
                    animate=False,
                    immediate=True,
                    force=True,
                )

    ChatApp().run()


def _message_label(role: str) -> str:
    if role == "user":
        return "You: "
    if role == "assistant":
        return "Assistant: "
    return "Status: "


def _message_style(role: str) -> str:
    if role == "user":
        return "cyan"
    if role == "assistant":
        return "green"
    return "yellow"


def _message_renderable(message: ChatMessage) -> Text:
    """Render one chat message with a styled role prefix."""
    label = _message_label(message["role"])
    style = _message_style(message["role"])
    text = Text()
    text.append(label, style=f"bold {style}")
    text.append(message["content"], style=style)
    text.append("\n")
    return text


@click.command(
    help=(
        "Run an interactive Qwen3 chat UI when attached to a TTY, or read one "
        "prompt from stdin and print the assistant response."
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
    default=0.6,
    show_default=True,
    type=float,
    help="Sampling temperature. Use 0 for greedy decoding.",
)
@click.option(
    "--top-k",
    default=20,
    type=click.IntRange(min=1),
    show_default=True,
    help="Restrict sampling to the top K tokens.",
)
@click.option(
    "--top-p",
    default=0.95,
    type=float,
    show_default=True,
    help="Restrict sampling to nucleus probability P.",
)
@click.option(
    "--local-files-only",
    is_flag=True,
    help="Only use models already present in the local Hugging Face cache.",
)
@click.option(
    "--no-think",
    is_flag=True,
    help="Disable Qwen thinking mode in chat prompts.",
)
@click.option(
    "--rag-embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Embedding model repo id or local path used for RAG VectorDB search.",
)
@click.option(
    "--rag-vector-db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a VectorDB JSON file. Enables RAG when provided.",
)
@click.option(
    "--rag-score-cutoff",
    default=0.3,
    show_default=True,
    type=float,
    help="Minimum vector search score required for a RAG entry to be added.",
)
@click.option(
    "--rag-max-entries",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of RAG search results to add to the prompt.",
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
    no_think: bool,
    rag_embedding_model: str,
    rag_vector_db_path: Path | None,
    rag_score_cutoff: float,
    rag_max_entries: int,
    verbosity: str,
) -> None:
    _configure_cli_logging(verbosity)
    generator = _build_qwen3_generator(
        model,
        cache_length=cache_length,
        local_files_only=local_files_only,
    )
    rag_context = (
        _build_rag_context(
            vector_db_path=rag_vector_db_path,
            embedding_model=rag_embedding_model,
            score_cutoff=rag_score_cutoff,
            max_entries=rag_max_entries,
        )
        if rag_vector_db_path is not None
        else None
    )

    if _stdin_is_interactive():
        _run_textual_chat_app(
            generator,
            cache_length=cache_length,
            options=ChatGenerationOptions(
                max_generated_token=max_generated_token,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                enable_thinking=not no_think,
            ),
            rag_context=rag_context,
        )
        return

    prompt = click.get_text_stream("stdin").read()
    if not prompt.strip():
        raise click.UsageError("expected a chat prompt on stdin")

    response = generate_chat_response(
        generator,
        prompt,
        max_generated_token=max_generated_token,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_thinking=not no_think,
        rag_context=rag_context,
    )
    click.echo(response)
