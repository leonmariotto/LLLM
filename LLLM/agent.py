"""
Agent implementation.

Use LlmClient as backend, handle tool execution.
Do context management.

All the agent runtime is stored in a single entity ExecutionContext.
Execution context contain a list of Event, which can be AgentToolResult,
ToolCall, or Message from user, system or assistant.

This context is used to forge a LLMRequest, at this point we can do
the context management thing: include some, exclude some info of the
context.

The actual LLM call occure within LLMClient, which take a LLMRequest
and output a LlmResponse.

The LlmResponse is then parsed, if it contain ToolCall execute them.
We check that the LlmResponse contain no final_answer: depending on
configuration at init, final_answer may be provided by a tool call.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .agent_context import ContentItem, ExecutionContext, Message
from .tool_common import ToolCall
from .agent_context import AgentToolResult
from .agent_context import AgentResult
from .agent_llm import (
    LlmClient,
    LlmRequest,
    LlmResponse,
)
from .tool_common import Tool

from loguru import logger


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
        instruction: str = "",
        instructions: str | Sequence[str] | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        if instruction and instructions is not None:
            raise ValueError("use either instruction or instructions, not both")
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        self.llm = llm
        self.tools = tuple(tools)
        self._tools_by_name = self._index_tools(self.tools)
        self.system_instructions = self._normalize_instructions(
            instructions if instructions is not None else instruction
        )
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        prompt: str,
        *,
        context: ExecutionContext | None = None,
    ) -> str:
        """
        Run the agent until the model returns an assistant answer.

        @param prompt: user input
        @param context: optional caller initialized execution context. If None it's
            init here.
        @return agent final answer.
        """

        # Create execution context.
        execution_context = context if context is not None else ExecutionContext()

        # Add user input as the first event.
        execution_context.add_user_message(prompt)

        # Set up code execution environment if needed
        # TODO

        # while execution_context.final_result is None:
        #     result = self.step(execution_context)
        #     self.step(execution_context)
        #
        #     # Check if the last event is a final response
        #     if context.events:
        #         last_event = context.events[-1]
        #         if self._is_final_response(last_event):
        #             context.final_result = self._extract_final_result(last_event)
        #     if final_answer is not None:
        #         return final_answer
        #
        # return AgentResult(output=context.final_result, context=context)
        while execution_context.final_result is None:
            final_answer = self.step(execution_context)
            if final_answer is not None:
                return final_answer

        return str(execution_context.final_result)

    def step(self, context: ExecutionContext) -> AgentResult | None:
        """
        Perform one ReAct think-act cycle.

        @param context: execution context to update.
        @return final assistant answer when the run is complete, otherwise None.
        """
        # Prepare llm request
        request = LlmRequest(
            instructions=self.system_instructions,
            content=context.items(),
            tool_schemas=[tool.schema for tool in self.tools],
        )
        response = self.think(request)

        if response.error_message is not None and not response.content:
            raise RuntimeError(response.error_message)

        assistant_items = self._response_items(
            response,
            step=context.current_step,
        )
        context.add_agent_items("assistant", assistant_items)

        tool_calls = [item for item in assistant_items if isinstance(item, ToolCall)]
        # If not tool call its a final answer....
        # TODO do a final_answer tool call !
        if not tool_calls:
            final_answer = ""
            if len(assistant_items) < 1:
                logger.error("No assistant item and no tool call\n")
            else:
                item = assistant_items[-1]
                if isinstance(item, Message):
                    final_answer = item.content

            context.final_result = final_answer
            return final_answer

        if context.current_step >= self.max_tool_rounds:
            raise RuntimeError(f"maximum tool rounds exceeded ({self.max_tool_rounds})")
        self.act(context, tool_calls)
        return None

    def think(self, request: LlmRequest) -> LlmResponse:
        """
        Ask the LLM for the next assistant message or tool call.
        Prepare the request using current ExecutionContext.
        Context management happen here !

        @param request: request.
        @return parsed LLM response.
        """
        return self.llm.complete(request)

    def act(
        self,
        context: ExecutionContext,
        tool_calls: ToolCall | Sequence[ToolCall],
    ) -> list[AgentToolResult]:
        """
        Execute tool calls and append their results to the context.

        @param context: execution context to update.
        @param tool_calls: tool calls emitted by the last think phase.
        @return stored tool results.
        """
        pending_tool_calls = (
            [tool_calls] if isinstance(tool_calls, ToolCall) else tool_calls
        )
        tool_results = [
            self._execute_tool_call(tool_call) for tool_call in pending_tool_calls
        ]
        context.add_agent_items("tool", tool_results)
        context.increment_step()
        return tool_results

    @staticmethod
    def _normalize_instructions(instructions: str | Sequence[str]) -> list[str]:
        if isinstance(instructions, str):
            return [instructions] if instructions else []
        return [instruction for instruction in instructions if instruction]

    @staticmethod
    def _response_items(response: LlmResponse, *, step: int) -> list[ContentItem]:
        """Convert an LLM response into agent event items with local tool IDs."""
        items: list[ContentItem] = []
        tool_call_index = 0
        for item in response.content:
            if isinstance(item, ToolCall):
                items.append(
                    item.model_copy(
                        update={"tool_call_id": f"call_{step}_{tool_call_index}"}
                    )
                )
                tool_call_index += 1
            else:
                items.append(item)
        return items

    @classmethod
    def _index_tools(cls, tools: Sequence[Tool]) -> dict[str, Tool]:
        """
        Build a tool dictionary for easy access (by name).
        """
        indexed: dict[str, Tool] = {}
        for tool in tools:
            name = cls._tool_name(tool.schema)
            if name in indexed:
                raise ValueError(f"duplicate tool name {name!r}")
            indexed[name] = tool
        return indexed

    @staticmethod
    def _tool_name(schema: dict[str, object]) -> str:
        """
        Retrieve a tool name from a tool description dict.
        """
        function = schema.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool schema must include a function object")
        function_dict = cast(dict[str, object], function)
        name = function_dict.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool schema function must include a non-empty name")
        return name

    def _execute_tool_call(self, tool_call: ToolCall) -> AgentToolResult:
        """
        Lookup in tool dictionary and execute tool.
        Return AgentToolResult.
        """
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            return AgentToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"unknown tool {tool_call.name!r}"],
            )

        try:
            result = tool.execute(cast(dict[str, object], tool_call.arguments))
        except Exception as error:
            return AgentToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"{tool_call.name!r} failed: {error}"],
            )

        return AgentToolResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            status="success",
            content=[result],
        )
