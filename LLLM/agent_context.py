"""
Agent execution context central storage, and its internal types.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal


class Message(BaseModel):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class AgentToolCall(BaseModel):
    """
    Agent request to execute a tool.
    The only difference with Generator ToolCall is the tool_call_id, used by
    agent to differenciate tool calls.
    LLM's output ToolCall, agent give it an id and it become AgentToolCall.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


class AgentToolResult(BaseModel):
    """
    Result from tool execution.
    Built by agent before feeding the ExecutionContext.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list[Any]


ContentItem = Message | AgentToolCall | AgentToolResult


def _empty_content() -> list[ContentItem]:
    return []


def _empty_events() -> list["Event"]:
    return []


def _empty_state() -> dict[str, Any]:
    return {}


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: list[ContentItem] = Field(default_factory=_empty_content)


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=_empty_events)
    current_step: int = 0
    state: dict[str, Any] = field(default_factory=_empty_state)
    final_result: str | BaseModel | None = None

    def add_event(self, event: Event) -> None:
        """Append an event to the execution history."""
        self.events.append(event)

    def add_user_message(self, content: str) -> Message:
        """Record a user message and return the stored item."""
        message = Message(role="user", content=content)
        self.add_event(
            Event(
                execution_id=self.execution_id,
                author="user",
                content=[message],
            )
        )
        return message

    def add_agent_items(
        self,
        author: str,
        items: Sequence[ContentItem],
    ) -> None:
        """Record one assistant/tool event when there is content to store."""
        if not items:
            return
        self.add_event(
            Event(
                execution_id=self.execution_id,
                author=author,
                content=list(items),
            )
        )

    def items(self) -> list[ContentItem]:
        """Return all content items in event order."""
        return [item for event in self.events for item in event.content]

    def messages(self) -> list[Message]:
        """Return all text messages in event order."""
        return [item for item in self.items() if isinstance(item, Message)]

    def increment_step(self) -> None:
        """Move to the next execution step."""
        self.current_step += 1
