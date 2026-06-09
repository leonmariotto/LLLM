from typing import Any, Literal
from dataclasses import dataclass
from collections.abc import Callable
from pydantic import BaseModel


ToolExecutor = Callable[[dict[str, object]], str]


@dataclass(frozen=True)
class Tool:
    """A function schema exposed to the model and its local implementation."""

    schema: dict[str, object]
    execute: ToolExecutor


class ToolCall(BaseModel):
    """A parsed assistant request to call one tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = ""
    name: str
    arguments: dict[str, object]

    def __init__(self, *args: Any, **data: Any) -> None:
        if args:
            if len(args) != 2 or "name" in data or "arguments" in data:
                raise TypeError(
                    "ToolCall accepts either ToolCall(name, arguments) or keyword "
                    "arguments"
                )
            data["name"] = args[0]
            data["arguments"] = args[1]
        super().__init__(**data)
