"""Shared types for the LLM layer.

These dataclasses are the contract between the model client and the agent loop.
The interface is intentionally model-agnostic — swap the backend without
touching the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single conversation turn."""
    role: str                           # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None     # set when role == "tool"
    name: str | None = None             # tool name when role == "tool"
    tool_calls: list[dict] | None = None  # set when role == "assistant" and a tool was called

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class GenerationResult:
    """The result of one model call."""
    content: str                                # Clean text (think blocks stripped)
    thinking: str = ""                          # Raw <think>...</think> content
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"                 # "stop" | "tool_calls" | "length"
    usage: "Any | None" = None                  # TokenUsage if available from API response
