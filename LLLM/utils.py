from collections.abc import Mapping, Sequence
import json
from typing import cast

from loguru import logger
import torch

from .generator import AssistantOutput, ToolCall


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def get_device_str() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"

    return device


def render_plain_chat_template(
    messages: Sequence[Mapping[str, object]],
    *,
    tools: Sequence[dict[str, object]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a generic text chat format for tokenizers without native chat."""
    logger.warning("Tokenizer lack chat format, using plain chat template")
    prompt = ""
    if tools is not None:
        prompt += "system: Available tools:\n"
        for tool in tools:
            prompt += json.dumps(tool, ensure_ascii=False) + "\n"
        prompt += (
            "\nTo call a tool, output one line per call in this exact format:\n"
            'tool_call: {"name":"function_name","arguments":{"key":"value"}}\n\n'
        )

    for message in messages:
        role = _message_role(message)
        content = _message_content(message)
        prompt += f"{role}: {content}\n"
        raw_calls = message.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, list):
                raise TypeError("assistant tool_calls must be a list[ToolCall]")
            raw_call_items = cast(list[object], raw_calls)
            if not all(isinstance(call, ToolCall) for call in raw_call_items):
                raise TypeError("assistant tool_calls must be a list[ToolCall]")
            for call in cast(list[ToolCall], raw_call_items):
                prompt += (
                    "tool_call: "
                    f"{json.dumps({'name': call.name, 'arguments': call.arguments}, ensure_ascii=False)}\n"
                )

    if add_generation_prompt:
        prompt += "assistant: "
    return prompt


def parse_plain_assistant_output(completion: str) -> AssistantOutput:
    """Parse plain text output from tokenizers with no native tool syntax."""
    content_lines: list[str] = []
    tool_calls: list[ToolCall] = []
    for line in completion.splitlines():
        stripped = line.strip()
        if not stripped.startswith("tool_call:"):
            content_lines.append(line)
            continue

        payload = stripped[len("tool_call:") :].strip()
        try:
            raw_call = cast(object, json.loads(payload))
        except json.JSONDecodeError as error:
            raise ValueError("tool call must contain valid JSON") from error
        if not isinstance(raw_call, dict):
            raise ValueError("tool call must be a JSON object")
        call = cast(dict[str, object], raw_call)
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        tool_calls.append(
            ToolCall(name=name, arguments=cast(dict[str, object], arguments))
        )

    return AssistantOutput(
        content="\n".join(content_lines).strip(),
        tool_calls=tuple(tool_calls),
    )


def _message_role(message: Mapping[str, object]) -> str:
    role = message.get("role")
    if not isinstance(role, str):
        raise TypeError("message role must be a string")
    return role


def _message_content(message: Mapping[str, object]) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("message content must be a string")
    return content
