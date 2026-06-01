"""
Tool-aware generation loop built on top of a token-based text generator.

The loop is independent of model-specific tool syntax.  Its tokenizer protocol
is responsible for rendering tools into a chat prompt and parsing assistant
completions into structured tool calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict, cast

from loguru import logger

# Use code execution or programming tools for running code, checking syntax,
# inspecting files, tests, data processing, or reproducible technical work.

SYSTEM_PROMPT_V1 = """You are LLLM, a tool-capable assistant.

- When the user's intent is clear, execute immediately without confirmation.
- Only when intent is unclear, ask minimal questions to clarify.
- Use tools proactively, without asking permission, when a tool is needed to
  answer accurately or complete the task.
- Do not claim that you used a tool unless a tool call was actually made.
- Use a search or retrieval tool for current, changing, obscure,
  source-backed, or user-specified external information.
- Use a search or retrieval tool when the user asks to look up, search, browse,
  verify, cite, open a URL, inspect a page, or use a named source.
- Use a compute tool for arithmetic, unit conversion, formulas, precision
  math, or any calculation where mental math may be error-prone.
- If a tool result is incomplete or only identifies a source, continue with the
  next appropriate tool call, such as opening a search result, before answering, do
  not stop the work until it's finish.
- Do not use search tool for timeless information, fundamental concepts, definitions, or
  well-established technical facts.
- Do not use tools for simple language edits, brainstorming, summaries of text
  already provided by the user, or straightforward reasoning from given facts.
- Do not use a tool when the user explicitly asks you not to.
- Your internal knowledge may be incomplete or outdated.
- You cannot access external pages, files, or runtime state unless they are in
  the conversation or obtained through an available tool.
- Be direct, concise, and useful.
- Put the answer first, then brief supporting details when needed.
- Use Markdown for lists, tables, and code blocks when it improves clarity.
- Distinguish facts from assumptions. Cite or name sources when tool results
  provide them.
- Do not stop immediatly when a tool return. Keep trying until the work is done
  or you're confident that you can't find the needed informations.
- Keep work until work is done.

Examples:
- User: "What is binary search?" Assistant: answer directly without tools.
- User: "What's the latest Python version?" Assistant: use search/retrieval,
  then answer with the current version and source.
- User: "Calculate the surface area of a sphere with diameter 22.2 cm."
  Assistant: use compute, then provide the numeric result and formula.
- User: "Is Venezuela was a participant of the 2026 Winter Olympic Games"
  Assistant: use wikisearch tool to find the corresponding wikipedia pages, then
    search 'Venezuela' in this page.
- User: "Write a script to rename these files." Assistant: provide or edit code;
  use file/code tools if repository or local files must be inspected.
- User: "Should I use a tool?" Assistant: if the task matches a trigger, use the
  tool immediately rather than asking permission.
"""


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by an assistant completion."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AssistantOutput:
    """Parsed assistant completion with optional tool calls."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class ToolMessage(TypedDict):
    """Structured chat message understood by tool-capable tokenizers."""

    role: str
    content: str
    tool_calls: NotRequired[list[ToolCall]]


ToolExecutor = Callable[[dict[str, object]], str]


@dataclass(frozen=True)
class Tool:
    """A function schema exposed to the model and its local implementation."""

    schema: dict[str, object]
    execute: ToolExecutor


class ToolTokenizer(Protocol):
    """Tokenizer operations required by the model-agnostic tool loop."""

    def apply_chat_template(
        self,
        messages: Sequence[ToolMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]] | str: ...

    def parse_assistant_output(self, completion: str) -> AssistantOutput: ...


class TextGenerator(Protocol):
    """Underlying completion generator used for each assistant turn."""

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


class GeneratorWithTool:
    """Run generation and local tool execution until an assistant answer is ready."""

    def __init__(
        self,
        generator: TextGenerator,
        tools: Sequence[Tool],
        *,
        max_tool_rounds: int = 8,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        self.generator = generator
        self.tokenizer = cast(ToolTokenizer, generator.tokenizer)
        self.tools = tuple(tools)
        self.max_tool_rounds = max_tool_rounds
        self._tools_by_name = self._index_tools(self.tools)

    def generate(
        self,
        messages: Sequence[ToolMessage],
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        cache_length: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """Return a final assistant response after any required tool rounds."""
        history = self._copy_messages(messages)
        tool_rounds = 0
        logger.info(
            "Tool generation started with {} tools and max_tool_rounds={}",
            len(self.tools),
            self.max_tool_rounds,
        )

        while True:
            prompt_tokens = self._encode_history(history)
            logger.info(
                "Generating assistant turn for tool round {} with {} history messages",
                tool_rounds,
                len(history),
            )
            completion = self.generator.generate_from_tokens(
                prompt_tokens,
                stop_at_eos=stop_at_eos,
                max_generated_token=max_generated_token,
                cache_length=cache_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                include_prompt=False,
            )
            logger.debug(
                "Generated assistant completion on tool round {}:\n{}",
                tool_rounds,
                completion,
            )
            try:
                output = self.tokenizer.parse_assistant_output(completion)
            except ValueError as error:
                logger.info(
                    "Assistant output could not be parsed as tool calls on round {}: {}",
                    tool_rounds,
                    error,
                )
                self._check_tool_round_limit(tool_rounds)
                history.append({"role": "assistant", "content": completion})
                history.append(
                    {
                        "role": "tool",
                        "content": f"Tool error: invalid tool call output: {error}",
                    }
                )
                tool_rounds += 1
                continue

            if not output.tool_calls:
                logger.info(
                    "Tool generation completed after {} tool rounds",
                    tool_rounds,
                )
                return output.content

            logger.debug(
                "Parsed assistant content on tool round {}:\n{}",
                tool_rounds,
                output.content,
            )
            for index, tool_call in enumerate(output.tool_calls):
                logger.debug(
                    "Parsed tool call block {} on round {}: name={} arguments={}",
                    index,
                    tool_rounds,
                    tool_call.name,
                    tool_call.arguments,
                )
            logger.info(
                "Assistant requested {} tool calls on round {}: {}",
                len(output.tool_calls),
                tool_rounds,
                [tool_call.name for tool_call in output.tool_calls],
            )
            self._check_tool_round_limit(tool_rounds)
            history.append(
                {
                    "role": "assistant",
                    "content": output.content,
                    "tool_calls": list(output.tool_calls),
                }
            )
            for tool_call in output.tool_calls:
                tool_response = self._execute_tool_call(tool_call)
                logger.info(
                    "Tool {} response:\n{}",
                    tool_call.name,
                    tool_response,
                )
                history.append(
                    {
                        "role": "tool",
                        "content": tool_response,
                    }
                )
            tool_rounds += 1

    def _encode_history(self, messages: list[ToolMessage]) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=[tool.schema for tool in self.tools],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(encoded, dict):
            raise TypeError("expected tokenized chat template output")
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise TypeError("expected input_ids to be a list[int]")
        return input_ids

    @staticmethod
    def _copy_messages(messages: Sequence[ToolMessage]) -> list[ToolMessage]:
        history: list[ToolMessage] = []
        for message in messages:
            copied: ToolMessage = {
                "role": message["role"],
                "content": message["content"],
            }
            if "tool_calls" in message:
                copied["tool_calls"] = list(message["tool_calls"])
            history.append(copied)
        return history

    @classmethod
    def _index_tools(cls, tools: Sequence[Tool]) -> dict[str, Tool]:
        indexed: dict[str, Tool] = {}
        for tool in tools:
            name = cls._tool_name(tool.schema)
            if name in indexed:
                raise ValueError(f"duplicate tool name {name!r}")
            indexed[name] = tool
        return indexed

    @staticmethod
    def _tool_name(schema: dict[str, object]) -> str:
        function = schema.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool schema must include a function object")
        function_dict = cast(dict[str, object], function)
        name = function_dict.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool schema function must include a non-empty name")
        return name

    def _execute_tool_call(self, tool_call: ToolCall) -> str:
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            logger.info("Tool call rejected for unknown tool {}", tool_call.name)
            return f"Tool error: unknown tool {tool_call.name!r}"
        logger.info("Executing tool {}", tool_call.name)
        try:
            result = tool.execute(tool_call.arguments)
        except Exception as error:
            logger.info("Tool {} failed: {}", tool_call.name, error)
            return f"Tool error: {tool_call.name!r} failed: {error}"
        logger.info("Tool {} completed", tool_call.name)
        return result

    def _check_tool_round_limit(self, tool_rounds: int) -> None:
        if tool_rounds >= self.max_tool_rounds:
            logger.info(
                "Tool generation stopped after reaching max_tool_rounds={}",
                self.max_tool_rounds,
            )
            raise RuntimeError(f"maximum tool rounds exceeded ({self.max_tool_rounds})")
