"""
Agent execution context central storage, and its internal types.
"""

from typing import Any, Dict, List, Optional, Literal, Union
import uuid
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime


# TODO system prompt is a message ?
class Message(BaseModel):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: Dict[str, Any]


# TODO strict typing here.
# TODO the ai-agent-from-scratch BaseTool seems not needed, I rather do without it,
# or maybe only have a common BaseTool class, but no decorator.
# TODO BaseTool inherit ABC, whats ABC ??
class ToolResult(BaseModel):
    """
    Result from tool execution.
    This is not returned by tools, its built by agent before feeding the
    ExecutionContext.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: List[Any]


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1
