"""Tool registry.

Tools are plain Python callables wrapped in a Tool descriptor that carries:
  - the OpenAI-compatible function schema (for the model)
  - keyword tags (for the router to score relevance without an LLM call)

Usage:
    from birdclaw.tools.registry import registry, Tool

    registry.register(Tool(
        name="my_tool",
        description="Does something useful.",
        input_schema={"properties": {"arg": {"type": "string"}}, "required": ["arg"]},
        handler=my_handler,
        tags=["keyword1", "keyword2"],
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]   # {"properties": {...}, "required": [...]}
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)  # for tool router scoring

    def to_openai_schema(self) -> dict[str, Any]:
        """Full schema — includes all parameters and their descriptions."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.input_schema.get("properties", {}),
                    "required": self.input_schema.get("required", []),
                },
            },
        }

    def to_compact_schema(self) -> dict[str, Any]:
        """Minimal schema for small models — required parameters only, no per-property descriptions.

        Reduces token cost by ~60% vs full schema. Use this in the agent loop
        when passing tools to the model. Keep the tool-level description (the
        model still needs to know what the tool does) but strip per-property
        descriptions and drop optional parameters entirely.
        """
        required = self.input_schema.get("required", [])
        all_props = self.input_schema.get("properties", {})

        # Keep only required params; strip their descriptions
        compact_props = {
            name: {k: v for k, v in prop.items() if k != "description"}
            for name, prop in all_props.items()
            if name in required
        }

        # Cap description to first sentence or 60 chars — small models
        # fail when tool schemas are verbose.
        desc = self.description
        first_sentence_end = desc.find(". ")
        if first_sentence_end != -1 and first_sentence_end < 80:
            desc = desc[: first_sentence_end + 1]
        desc = desc[:80]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": compact_props,
                    "required": required,
                },
            },
        }


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("[registry] registered  name=%s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())


# Module-level singleton
registry = Registry()
