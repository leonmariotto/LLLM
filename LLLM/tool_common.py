from typing import TypeAlias
from dataclasses import dataclass
from collections.abc import Callable

from .generator import (
    ChatMessage,
)

ToolMessage: TypeAlias = ChatMessage

ToolExecutor = Callable[[dict[str, object]], str]


@dataclass(frozen=True)
class Tool:
    """A function schema exposed to the model and its local implementation."""

    schema: dict[str, object]
    execute: ToolExecutor
