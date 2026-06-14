# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner
import pytest

from ..LLLM import chat as chat_module
from ..LLLM.vector_db import VectorDB
from ..LLLM.vector_search import SearchResult


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.enable_thinking: list[bool] = []
        self.messages: list[list[chat_module.ChatMessage]] = []
        self.chat_enable_thinking: list[bool] = []

    def encode_instruct_prompt(
        self,
        prompt: str,
        *,
        enable_thinking: bool = True,
    ) -> list[int]:
        self.prompts.append(prompt)
        self.enable_thinking.append(enable_thinking)
        return [len(prompt), int(enable_thinking)]

    def apply_chat_template(
        self,
        messages: list[chat_module.ChatMessage],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> dict[str, list[int]] | str:
        self.chat_enable_thinking.append(enable_thinking)
        self.messages.append(
            [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ]
        )
        prompt = "".join(
            f"{message['role']}:{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            prompt += "assistant:"
        if not tokenize:
            return prompt
        return {"input_ids": list(range(len(prompt)))}


class FakeGenerator:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.tokenizer = FakeTokenizer()
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "max_generated_token": max_generated_token,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "include_prompt": include_prompt,
            }
        )
        return self.outputs[len(self.calls) - 1]


class FakeTensor:
    def __init__(self, numel: int, element_size: int) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class FakeAttention:
    num_kv_groups = 2
    head_dim = 4


class FakeBlock:
    att = FakeAttention()


class FakeModel:
    context_length = 64
    trf_blocks = [FakeBlock(), FakeBlock(), FakeBlock()]

    def parameters(self) -> list[FakeTensor]:
        return [FakeTensor(10, 4)]

    def buffers(self) -> list[FakeTensor]:
        return [FakeTensor(5, 4)]


class FakeGeneratorWithModel(FakeGenerator):
    def __init__(self, outputs: Sequence[str]) -> None:
        super().__init__(outputs)
        self.model = FakeModel()
        self.cache_length = 8


class FakeVectorDB:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query_str: str,
        metadata_filter: Sequence[str] | None = None,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query_str": query_str,
                "metadata_filter": metadata_filter,
                "top_k": top_k,
            }
        )
        return self.results[:top_k]


def _fake_rag_context(
    fake_db: FakeVectorDB,
    *,
    score_cutoff: float = 0.3,
    max_entries: int = 5,
) -> chat_module.ChatRAGContext:
    return chat_module.ChatRAGContext(
        vector_db=cast(VectorDB, fake_db),
        options=chat_module.ChatRAGOptions(
            score_cutoff=score_cutoff,
            max_entries=max_entries,
        ),
    )


def test_generate_chat_response_enables_and_strips_thinking_by_default() -> None:
    generator = FakeGenerator(["<think>hidden</think>\nHello"])

    response = chat_module.generate_chat_response(
        generator,
        "Say hello",
        max_generated_token=32,
        temperature=0.0,
        top_k=None,
        top_p=None,
    )

    assert response == "Hello"
    assert generator.tokenizer.prompts == ["Say hello"]
    assert generator.tokenizer.enable_thinking == [True]
    assert generator.calls == [
        {
            "prompt_tokens": [9, 1],
            "max_generated_token": 32,
            "temperature": 0.0,
            "top_k": None,
            "top_p": None,
            "include_prompt": False,
        }
    ]


def test_generate_chat_response_can_disable_thinking() -> None:
    generator = FakeGenerator(["<think>hidden</think>\nHello"])

    response = chat_module.generate_chat_response(
        generator,
        "Say hello",
        max_generated_token=32,
        temperature=0.0,
        top_k=None,
        top_p=None,
        enable_thinking=False,
    )

    assert response == "Hello"
    assert generator.tokenizer.enable_thinking == [False]
    assert generator.calls[0]["prompt_tokens"] == [9, 0]


def test_generate_chat_response_augments_prompt_with_rag_context() -> None:
    generator = FakeGenerator(["answer"])
    fake_db = FakeVectorDB(
        [
            SearchResult(index=0, score=0.9, sequence="relevant chunk"),
            SearchResult(index=1, score=0.2, sequence="weak chunk"),
        ]
    )

    response = chat_module.generate_chat_response(
        generator,
        "question",
        max_generated_token=32,
        temperature=0.0,
        top_k=None,
        top_p=None,
        rag_context=_fake_rag_context(fake_db, score_cutoff=0.3, max_entries=2),
    )

    assert response == "answer"
    assert fake_db.calls == [
        {"query_str": "question", "metadata_filter": None, "top_k": 2}
    ]
    prompt = generator.tokenizer.prompts[0]
    assert "Relevant context:\n[1] relevant chunk" in prompt
    assert "weak chunk" not in prompt
    assert prompt.endswith("User question:\nquestion")


def test_generate_chat_messages_response_keeps_full_history() -> None:
    generator = FakeGenerator(["<think>hidden</think>\nanswer"])
    messages: list[chat_module.ChatMessage] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]

    response = chat_module._generate_chat_messages_response(
        generator,
        messages,
        max_generated_token=16,
        temperature=0.0,
        top_k=None,
        top_p=None,
    )

    assert response == "answer"
    assert generator.tokenizer.messages == [messages]
    assert generator.tokenizer.chat_enable_thinking == [True]
    assert len(generator.calls[0]["prompt_tokens"]) == 49
    assert generator.calls[0]["include_prompt"] is False


def test_generate_chat_messages_response_augments_latest_user_without_mutating_history() -> None:
    generator = FakeGenerator(["answer"])
    fake_db = FakeVectorDB([SearchResult(index=0, score=0.9, sequence="context")])
    messages: list[chat_module.ChatMessage] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]

    response = chat_module._generate_chat_messages_response(
        generator,
        messages,
        max_generated_token=16,
        temperature=0.0,
        top_k=None,
        top_p=None,
        rag_context=_fake_rag_context(fake_db),
    )

    assert response == "answer"
    assert messages[-1]["content"] == "third"
    encoded_messages = generator.tokenizer.messages[0]
    assert encoded_messages[0]["content"] == "first"
    assert encoded_messages[2]["content"].startswith("Relevant context:\n[1] context")
    assert encoded_messages[2]["content"].endswith("User question:\nthird")


def test_chat_status_estimates_model_and_context_memory() -> None:
    generator = FakeGeneratorWithModel([])
    messages: list[chat_module.ChatMessage] = [{"role": "user", "content": "hello"}]

    status = chat_module._chat_status(generator, messages, cache_length=8)

    assert status.model_bytes == 60
    assert status.cache_length == 8
    assert status.absolute_position == 21
    assert status.context_bytes == 3 * 2 * 8 * 2 * 4 * 4


def test_format_status_includes_history_length_over_cache_length() -> None:
    status = chat_module.ChatStatus(
        model_bytes=60,
        cache_length=8,
        absolute_position=21,
        context_bytes=768,
    )

    assert (
        chat_module._format_status(status)
        == "Model 60 B | history_length/cache_length : 21/8 | "
        "Context 21 tok abs / 768 B est"
    )


def test_generate_chat_messages_response_rejects_context_overflow() -> None:
    generator = FakeGeneratorWithModel(["unused"])
    generator.model.context_length = 4
    messages: list[chat_module.ChatMessage] = [{"role": "user", "content": "hello"}]

    with pytest.raises(ValueError, match="conversation context is 21 tokens"):
        chat_module._generate_chat_messages_response(
            generator,
            messages,
            max_generated_token=16,
            temperature=0.0,
            top_k=None,
            top_p=None,
        )

    assert generator.calls == []


def test_chat_cli_reads_stdin_and_prints_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeGenerator(["answer"])
    captured: dict[str, object] = {}

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return generator

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )

    result = CliRunner().invoke(
        chat_module.chat_cli,
        ["--max-generated-token", "8", "--verbosity", "debug"],
        input="question\n",
    )

    assert result.exit_code == 0
    assert result.output == "answer\n"
    assert generator.tokenizer.prompts == ["question\n"]
    assert captured["repo_id"] == chat_module.DEFAULT_CHAT_MODEL_REPO_ID
    assert isinstance(captured["kwargs"], dict)
    assert captured["kwargs"]["cache_length"] == chat_module.DEFAULT_CACHE_LENGTH
    assert generator.tokenizer.enable_thinking == [True]


def test_chat_cli_no_think_disables_stdin_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeGenerator(["answer"])

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        return generator

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )

    result = CliRunner().invoke(
        chat_module.chat_cli,
        ["--no-think"],
        input="question\n",
    )

    assert result.exit_code == 0
    assert result.output == "answer\n"
    assert generator.tokenizer.enable_thinking == [False]


def test_chat_cli_stdin_uses_rag_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(["answer"])
    vector_db_path = tmp_path / "vectors.json"
    vector_db_path.write_text("[]\n", encoding="utf-8")
    fake_db = FakeVectorDB(
        [
            SearchResult(index=0, score=0.8, sequence="kept context"),
            SearchResult(index=1, score=0.1, sequence="dropped context"),
        ]
    )
    captured: dict[str, object] = {}

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        return generator

    def fake_build_rag_context(
        *,
        vector_db_path: Path,
        embedding_model: str,
        score_cutoff: float,
        max_entries: int,
    ) -> chat_module.ChatRAGContext:
        captured["vector_db_path"] = vector_db_path
        captured["embedding_model"] = embedding_model
        captured["score_cutoff"] = score_cutoff
        captured["max_entries"] = max_entries
        return _fake_rag_context(
            fake_db,
            score_cutoff=score_cutoff,
            max_entries=max_entries,
        )

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )
    monkeypatch.setattr(chat_module, "_build_rag_context", fake_build_rag_context)

    result = CliRunner().invoke(
        chat_module.chat_cli,
        [
            "--rag-vector-db-path",
            str(vector_db_path),
            "--rag-embedding-model",
            "embedding-model",
            "--rag-score-cutoff",
            "0.3",
            "--rag-max-entries",
            "2",
        ],
        input="question\n",
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "vector_db_path": vector_db_path,
        "embedding_model": "embedding-model",
        "score_cutoff": 0.3,
        "max_entries": 2,
    }
    assert fake_db.calls == [
        {"query_str": "question\n", "metadata_filter": None, "top_k": 2}
    ]
    prompt = generator.tokenizer.prompts[0]
    assert "kept context" in prompt
    assert "dropped context" not in prompt


def test_chat_cli_launches_textual_app_for_interactive_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeGenerator(["unused"])
    captured: dict[str, object] = {}

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return generator

    def fake_run_textual_chat_app(
        app_generator: chat_module.TextGenerator,
        *,
        cache_length: int,
        options: chat_module.ChatGenerationOptions,
        rag_context: chat_module.ChatRAGContext | None = None,
    ) -> None:
        captured["app_generator"] = app_generator
        captured["cache_length"] = cache_length
        captured["options"] = options
        captured["rag_context"] = rag_context

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )
    monkeypatch.setattr(chat_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        chat_module,
        "_run_textual_chat_app",
        fake_run_textual_chat_app,
    )

    result = CliRunner().invoke(
        chat_module.chat_cli,
        ["--max-generated-token", "8", "--verbosity", "debug"],
    )

    assert result.exit_code == 0
    assert result.output == ""
    assert captured["repo_id"] == chat_module.DEFAULT_CHAT_MODEL_REPO_ID
    assert isinstance(captured["kwargs"], dict)
    assert captured["kwargs"]["cache_length"] == chat_module.DEFAULT_CACHE_LENGTH
    assert captured["app_generator"] is generator
    assert captured["cache_length"] == chat_module.DEFAULT_CACHE_LENGTH
    assert captured["rag_context"] is None
    options = captured["options"]
    assert isinstance(options, chat_module.ChatGenerationOptions)
    assert options.max_generated_token == 8
    assert options.enable_thinking is True


def test_chat_cli_interactive_passes_rag_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(["unused"])
    vector_db_path = tmp_path / "vectors.json"
    vector_db_path.write_text("[]\n", encoding="utf-8")
    fake_db = FakeVectorDB([])
    rag_context = _fake_rag_context(fake_db, score_cutoff=0.4, max_entries=3)
    captured: dict[str, object] = {}

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        return generator

    def fake_build_rag_context(
        *,
        vector_db_path: Path,
        embedding_model: str,
        score_cutoff: float,
        max_entries: int,
    ) -> chat_module.ChatRAGContext:
        captured["vector_db_path"] = vector_db_path
        captured["embedding_model"] = embedding_model
        captured["score_cutoff"] = score_cutoff
        captured["max_entries"] = max_entries
        return rag_context

    def fake_run_textual_chat_app(
        app_generator: chat_module.TextGenerator,
        *,
        cache_length: int,
        options: chat_module.ChatGenerationOptions,
        rag_context: chat_module.ChatRAGContext | None = None,
    ) -> None:
        captured["app_generator"] = app_generator
        captured["rag_context"] = rag_context

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )
    monkeypatch.setattr(chat_module, "_build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(chat_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        chat_module,
        "_run_textual_chat_app",
        fake_run_textual_chat_app,
    )

    result = CliRunner().invoke(
        chat_module.chat_cli,
        [
            "--rag-vector-db-path",
            str(vector_db_path),
            "--rag-embedding-model",
            "embedding-model",
            "--rag-score-cutoff",
            "0.4",
            "--rag-max-entries",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["app_generator"] is generator
    assert captured["rag_context"] is rag_context
    assert captured["vector_db_path"] == vector_db_path
    assert captured["embedding_model"] == "embedding-model"
    assert captured["score_cutoff"] == 0.4
    assert captured["max_entries"] == 3


def test_chat_cli_no_think_reaches_textual_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = FakeGenerator(["unused"])
    captured: dict[str, object] = {}

    def fake_build_qwen3_generator(
        repo_id: str,
        **kwargs: object,
    ) -> FakeGenerator:
        return generator

    def fake_run_textual_chat_app(
        app_generator: chat_module.TextGenerator,
        *,
        cache_length: int,
        options: chat_module.ChatGenerationOptions,
        rag_context: chat_module.ChatRAGContext | None = None,
    ) -> None:
        captured["options"] = options
        captured["rag_context"] = rag_context

    monkeypatch.setattr(
        chat_module,
        "_build_qwen3_generator",
        fake_build_qwen3_generator,
    )
    monkeypatch.setattr(chat_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        chat_module,
        "_run_textual_chat_app",
        fake_run_textual_chat_app,
    )

    result = CliRunner().invoke(chat_module.chat_cli, ["--no-think"])

    assert result.exit_code == 0
    options = captured["options"]
    assert isinstance(options, chat_module.ChatGenerationOptions)
    assert options.enable_thinking is False
    assert captured["rag_context"] is None
