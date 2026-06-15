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

from .agent_context import ExecutionContext, Message
from .tool_common import ToolCall
from .agent_context import AgentToolResult
from .agent_context import AgentResult
from .agent_context import Event
from .agent_llm import (
    LlmClient,
    LlmRequest,
    LlmResponse,
)
from .tool_common import Tool

from loguru import logger

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
- If a tool result does not contain the requested answer, call another tool.
- Do not ask the user to look it up.
- Do not say you can help later.
- Keep using available tools until you have the answer or the tools clearly fail.


Examples:
- User: "What is binary search?" Assistant: answer directly without tools.
- User: "What's the latest Python version?" Assistant: use search/retrieval,
  then answer with the current version and source.
- User: "Calculate the surface area of a sphere with diameter 22.2 cm."
  Assistant: use compute, then provide the numeric result and formula.
- User: "Is Venezuela was a participant of the 2026 Winter Olympic Games"
  Assistant: use wiki tool to find the corresponding wikipedia pages, then
    search 'Venezuela' in this page.
- User: "Write a script to rename these files." Assistant: provide or edit code;
  use file/code tools if repository or local files must be inspected.
"""


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
        max_step: int = 8,
    ) -> None:
        if max_step < 0:
            raise ValueError("max_step must be non-negative")
        self.llm = llm
        self.tools = tuple(tools)
        self._tools_by_name = self._index_tools(self.tools)
        self.system_instructions = [instruction] if instruction else []
        self.max_step = max_step

    def run(
        self,
        prompt: str,
        *,
        context: ExecutionContext | None = None,
    ) -> AgentResult:
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

        while (
            execution_context.final_result is None
            and execution_context.current_step < self.max_step
        ):
            _ = self.step(execution_context)

            # Check if the last event is a final response
            if execution_context.events:
                last_event = execution_context.events[-1]
                if self._is_final_response(last_event):
                    execution_context.final_result = self._extract_final_result(
                        last_event
                    )
        if execution_context.current_step >= self.max_step:
            logger.warning("reached max_step, return final_result=None")

        return AgentResult(
            output=execution_context.final_result, context=execution_context
        )

    def _is_final_response(self, event: Event) -> bool:
        """
        Check if this event contains a final response.
        Return true if no ToolCall nor AgentToolResult in event contents.
        """
        # TODO check final_answer tool call at this point.
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, AgentToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str:
        """
        Extract the final result from an event.
        Return the first assistant message in the event contents.
        """
        # TODO extract the output of final_answer tool
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return "Woops!!"

    def step(self, context: ExecutionContext) -> None:
        """
        Perform one ReAct think-act cycle.

        @param context: execution context to update.
        @return None.
        """
        # Prepare what to send to the LLM
        request = LlmRequest(
            instructions=self.system_instructions,
            content=context.items(),
            tool_schemas=[tool.schema for tool in self.tools],
        )

        # Get LLM's decision
        response = self.think(request)

        # TODO pretty print LlmResposne
        logger.debug("LlmResponse=[{}]", response)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author="agent",
            content=response.content,
        )
        context.add_event(response_event)

        tool_calls = [item for item in response.content if isinstance(item, ToolCall)]
        if tool_calls:
            _ = self.act(context, tool_calls)
        context.increment_step()
        return None

    def think(self, request: LlmRequest) -> LlmResponse:
        """
        Ask the LLM for the next assistant message or tool call.

        @param request: request.
        @return parsed LLM response.
        """
        return self.llm.complete(request)

    def act(
        self,
        context: ExecutionContext,
        tool_calls: Sequence[ToolCall],
    ) -> None:
        """
        Execute tool calls and append their results to the context.

        @param context: execution context to update.
        @param tool_calls: tool calls emitted by the last think phase.
        @return stored tool results.
        """
        tool_results = [self._execute_tool_call(tool_call) for tool_call in tool_calls]
        tool_event = Event(
            execution_id=context.execution_id,
            author="tool",
            content=tool_results,
        )
        context.add_event(tool_event)
        return None

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
            result = tool.execute(tool_call.arguments)
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
