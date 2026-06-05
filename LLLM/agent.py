"""
Agent implementation.

Use directly Generator, handle tool execution.
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

"""

# class Agent:
