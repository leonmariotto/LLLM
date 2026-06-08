"""
Agent implementation.

Use LlmClient as backend, handle tool execution.
Do context management.

All the agent runtime is stored in a single entity ExecutionContext.
Execution context contain a list of Event, which can be ToolResult,
ToolCall, or Message from user, system or assistant.

This context is used to forge a LLMRequest, at this point we can do
the context management thing: include some, exclude some info of the
context.

The actual LLM call occure within LLMClient, which take a LLMRequest
and output a LlmResponse.

The LlmResponse is then parsed, if it contain ToolCall execute them.
We check that the LlmResponse contain no final_answer: depending on
configuration at init, final_answer may be provided by a tool call.

TODO this is really minimal, and I should get a lot of improvement by reading
at the implementation of scratch_agent.
TODO a lot of thing is overcomplicate here and could be simplified.
Need to re-review it.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .agent_context import ContentItem, ExecutionContext, Message
from .agent_context import ToolCall as AgentToolCall
from .agent_context import ToolResult
from .agent_llm import (
    LlmClient,
    LlmRequest,
    LlmResponse,
)
from .generator import ToolCall as GeneratorToolCall
from .generator_with_tool import Tool


class Agent:
    """
    Own conversation context, call the model, execute tools, and return text.
    Initialized with an LLM client and a tool list.
    """

    def __init__(
        self,
        llm: LlmClient,
        tools: Sequence[Tool],
        *,
        instructions: str | Sequence[str] | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        self.llm = llm
        self.tools = tuple(tools)
        self._tools_by_name = self._index_tools(self.tools)
        self.instructions = self._normalize_instructions(
            instructions
        )  # TODO not needed to have a list here. Instructions should be str.
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        prompt: str,
        *,
        context: ExecutionContext | None = None,
    ) -> str:
        """Run the agent until the model returns an assistant answer."""

        # Create execution context.
        execution_context = context if context is not None else ExecutionContext()

        # Add user input as the first event.
        execution_context.add_user_message(prompt)

        # TODO try catch??
        # Execute steps until completion or max steps reached
        tool_rounds = 0
        while True:  # TODO replace this while True by a real condition
            request = LlmRequest(
                instructions=self.instructions,
                content=execution_context.items(),
                tool_schemas=[tool.schema for tool in self.tools],
            )
            response = self.llm.complete(request)
            if (
                response.error_message is not None
                and not response.messages
                and not response.tool_calls
            ):
                raise RuntimeError(response.error_message)

            step = execution_context.current_step
            assistant_items = self._response_items(response, step=step)
            execution_context.add_agent_items("assistant", assistant_items)

            tool_calls = [
                item for item in assistant_items if isinstance(item, AgentToolCall)
            ]
            # If not tool call its a final answer....
            # TODO do a final_answer tool call !
            if not tool_calls:
                final_answer = self._last_assistant_text(assistant_items)
                execution_context.final_result = final_answer
                return final_answer

            self._check_tool_round_limit(tool_rounds)
            tool_results = [
                self._execute_tool_call(tool_call) for tool_call in tool_calls
            ]
            execution_context.add_agent_items("tool", tool_results)
            execution_context.increment_step()
            tool_rounds += 1

    @staticmethod
    def _normalize_instructions(
        instructions: str | Sequence[str] | None,
    ) -> list[str]:
        if instructions is None:
            return []
        if isinstance(instructions, str):
            return [instructions]
        return list(instructions)

    @staticmethod
    def _response_items(response: LlmResponse, *, step: int) -> list[ContentItem]:
        """Convert an LLM response into agent event items with local tool IDs."""
        items: list[ContentItem] = list(response.messages)
        items.extend(
            Agent._agent_tool_call(tool_call, step=step, index=index)
            for index, tool_call in enumerate(response.tool_calls)
        )
        return items

    @staticmethod
    def _agent_tool_call(
        tool_call: GeneratorToolCall,
        *,
        step: int,
        index: int,
    ) -> AgentToolCall:
        return AgentToolCall(
            tool_call_id=f"call_{step}_{index}",
            name=tool_call.name,
            arguments=dict(tool_call.arguments),
        )

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

    def _execute_tool_call(self, tool_call: AgentToolCall) -> ToolResult:
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"unknown tool {tool_call.name!r}"],
            )

        try:
            result = tool.execute(cast(dict[str, object], tool_call.arguments))
        except Exception as error:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"{tool_call.name!r} failed: {error}"],
            )

        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            status="success",
            content=[result],
        )

    @staticmethod
    def _last_assistant_text(items: Sequence[ContentItem]) -> str:
        for item in reversed(items):
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return ""

    def _check_tool_round_limit(self, tool_rounds: int) -> None:
        if tool_rounds >= self.max_tool_rounds:
            raise RuntimeError(f"maximum tool rounds exceeded ({self.max_tool_rounds})")
