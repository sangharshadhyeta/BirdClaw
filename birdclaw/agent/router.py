"""Tool router.

Selects the most relevant tools for a given query without making an LLM call.
Scores each registered tool by keyword overlap between its tags and the
current query + last few conversation turns, then returns the top N.

This keeps the model's tool list small (≤ max_tools_per_turn) so the 4B
model doesn't get confused by a wall of schemas it doesn't need.
"""

from __future__ import annotations

import logging
import re

from birdclaw.config import settings
from birdclaw.llm.types import Message
from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)

# Regex for extracting file paths from query text
_PATH_RE = re.compile(r"[\w./~-]+\.(?:py|rs|ts|js|toml|json|yaml|yml|md|txt|sh|cfg|ini)")


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens from text."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _query_tokens(query: str, history: list[Message], lookback: int = 3) -> set[str]:
    """Combine query and last N history turns into a token set."""
    combined = query
    for msg in history[-lookback:]:
        combined += " " + msg.content
    return _tokenise(combined)


def _has_file_path(query: str) -> bool:
    return bool(_PATH_RE.search(query))


def _looks_like_question(query: str) -> bool:
    q = query.strip().lower()
    return q.startswith(("what", "who", "where", "when", "why", "how", "is ", "are ", "does ", "can ")) or q.endswith("?")


def select(
    query: str,
    history: list[Message] | None = None,
    max_n: int | None = None,
) -> list[Tool]:
    """Return the most relevant tools for this query, capped at max_n.

    Control tools (think, answer) are NOT included — the loop adds them.
    """
    if max_n is None:
        max_n = settings.max_tools_per_turn

    all_tools = registry.all_tools()
    if not all_tools:
        return []

    tokens = _query_tokens(query, history or [])
    has_path = _has_file_path(query)
    is_question = _looks_like_question(query)

    # Score each tool by tag overlap
    scored: list[tuple[int, Tool]] = []
    for tool in all_tools:
        score = len(tokens & set(tool.tags))
        scored.append((score, tool))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Priority boosts — bump relevant tools to the front before capping
    # so they survive the max_n cut rather than being added after it.
    priority_names: list[str] = []
    if has_path:
        priority_names += ["write_file", "read_file"]
    if is_question:
        priority_names += ["web_search"]

    # Rebuild scored with priority tools first (deduplicated)
    priority_tools = [registry.get(n) for n in priority_names if registry.get(n)]
    priority_set = {t.name for t in priority_tools if t}
    rest = [t for _, t in scored if t.name not in priority_set]
    ordered = [t for t in priority_tools if t] + rest

    # Hard cap — never exceed max_n regardless of overrides
    result = ordered[:max_n]
    logger.debug("[router] select  query=%r  tools=%s", query[:40], [t.name for t in result])
    return result
