# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import Any

from click.testing import CliRunner
import pytest

from ..LLLM import chat as chat_module


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
    ) -> None:
        captured["app_generator"] = app_generator
        captured["cache_length"] = cache_length
        captured["options"] = options

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
    options = captured["options"]
    assert isinstance(options, chat_module.ChatGenerationOptions)
    assert options.max_generated_token == 8
    assert options.enable_thinking is True


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
    ) -> None:
        captured["options"] = options

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
