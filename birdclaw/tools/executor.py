"""Tool executor.

Calls registered tools and returns JSON-serialised results.

Hook layer (tools.hooks):
  Pre-hook runs before every tool call. It can deny execution or rewrite
  the tool arguments (updatedInput). Post-hook runs after every successful
  call; post_failure hook runs when the tool raises an exception.

Cache layer (memory.tool_cache):
  Before every tool call, checks the session graph for a valid cached result.
  If found and verified fresh, returns the cache hit immediately — no repeated
  user approvals for the same read-only operation.

  After a successful write (write_file, edit_file), invalidates any cached
  reads that cover the affected path so stale content is never served.

Observation bounding:
  Tool results exceeding _OBS_MAX_CHARS are saved to a temp file.
  The model receives a pointer + preview instead of the full output.
  This prevents context bloat from large bash outputs or web fetches.

All errors are caught and returned as {"error": "..."} so the model always
gets a valid tool result and the loop never crashes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from typing import Any

from birdclaw.llm.types import ToolCall
from birdclaw.tools.hooks import hook_runner
from birdclaw.tools.registry import registry

logger = logging.getLogger(__name__)

# Tools whose successful execution must invalidate path-based cache entries
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})

# Tools exempt from observation bounding — their output is always short/structured
_UNBOUNDED_TOOLS = frozenset({"answer", "think", "bash_poll", "bash_write", "bash_kill"})

# Max chars injected directly into the model's context per tool result.
# Anything larger is written to a temp file; the model gets a pointer + preview.
_OBS_MAX_CHARS = 800

# O4: Tool-specific result summarizers — run before bounding so models see
#     actionable content instead of raw dumps
_BASH_TOOLS   = frozenset({"bash"})
_SEARCH_TOOLS = frozenset({"web_search"})


def _summarise_bash(result_str: str) -> str:
    """Keep only error lines + last 80 chars of stdout for bash results.

    Full stdout is still saved to tmp via _bound_observation; this gives
    the model a focused view that highlights failures.
    """
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, ValueError):
        return result_str
    if not isinstance(data, dict):
        return result_str

    parts: list[str] = []
    stderr = str(data.get("stderr") or "").strip()
    stdout = str(data.get("stdout") or "").strip()
    rc = data.get("return_code", 0)

    if rc not in (0, None):
        parts.append(f"exit_code: {rc}")

    if stderr:
        # Keep all stderr — it's almost always the important signal
        parts.append(f"stderr: {stderr[:300]}")

    if stdout:
        # Keep only non-blank lines that look like errors/warnings, plus the tail
        error_lines = [
            l for l in stdout.splitlines()
            if re.search(r"\b(error|fail|exception|traceback|warning|fatal)\b", l, re.I)
        ]
        tail = stdout[-80:]
        if error_lines:
            parts.append("stdout (errors): " + " | ".join(error_lines[:5]))
        if tail and (not error_lines or tail not in "\n".join(error_lines)):
            parts.append(f"stdout (tail): {tail}")
    elif not parts:
        parts.append("(no output)")

    summary = json.dumps({k: v for k, v in data.items() if k not in ("stdout", "stderr")}
                         | {"_summary": " // ".join(parts)},
                         ensure_ascii=False)
    return summary


def _summarise_search(result_str: str, top_n: int = 2, query: str = "") -> str:
    """Keep the top-N most query-relevant search results to reduce context bloat.

    When a query is provided, results are scored by keyword overlap so the
    most relevant ones are kept — not just the first N by position.
    """
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, ValueError):
        return result_str
    if not isinstance(data, (dict, list)):
        return result_str

    results = data if isinstance(data, list) else data.get("results", [])
    if not isinstance(results, list) or len(results) <= top_n:
        return result_str

    if query:
        from birdclaw.llm.pruner import _tokenise
        goal_tokens = _tokenise(query)
        def _score(r: dict) -> int:
            text = f"{r.get('title', '')} {r.get('content', '')}".lower()
            return len(goal_tokens & _tokenise(text))
        results = sorted(results, key=_score, reverse=True)

    trimmed = results[:top_n]
    return json.dumps(trimmed, ensure_ascii=False)


def _summarise_result(tool_name: str, result_str: str, arguments: dict | None = None) -> str:
    """O4: Apply tool-specific summarization before bounding."""
    if tool_name in _BASH_TOOLS:
        return _summarise_bash(result_str)
    if tool_name in _SEARCH_TOOLS:
        query = (arguments or {}).get("query", "")
        return _summarise_search(result_str, query=query)
    return result_str


def _bound_observation(tool_name: str, result_str: str) -> str:
    """Cap tool output at _OBS_MAX_CHARS; overflow → temp file + preview pointer.

    Small models lose coherence when a single observation dominates their context
    window. Cap to 800 chars (≈200 tokens) and store the rest in /tmp so the model
    can read_file the full output only if it actually needs it.
    """
    if tool_name in _UNBOUNDED_TOOLS:
        return result_str
    if len(result_str) <= _OBS_MAX_CHARS:
        return result_str

    tmp_dir = tempfile.gettempdir()
    fname = f"bc_tool_{uuid.uuid4().hex[:8]}.txt"
    tmp_path = os.path.join(tmp_dir, fname)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(result_str)
        preview = result_str[:_OBS_MAX_CHARS]
        return (
            f'{{"truncated": true, "full_output": "{tmp_path}", '
            f'"preview": {json.dumps(preview)}}}'
        )
    except OSError:
        # If we can't write the temp file, fall back to truncation in-place
        return result_str[:_OBS_MAX_CHARS] + f'\n[... truncated at {_OBS_MAX_CHARS} chars]'


def _path_from_args(arguments: dict) -> str | None:
    return arguments.get("path")


def execute(tool_call: ToolCall) -> str:
    """Execute a tool call. Returns a JSON string (result or error).

    Runs pre/post hooks, checks the tool cache, then calls the tool handler.
    """
    from birdclaw.memory.tool_cache import get_cached, invalidate_path, store

    tool = registry.get(tool_call.name)
    if tool is None:
        msg = f"Unknown tool: {tool_call.name!r}. Available: {registry.names()}"
        logger.warning(msg)
        return json.dumps({"error": msg})

    # ── Pre-hook ──────────────────────────────────────────────────────────────
    pre = hook_runner.pre_tool_use(tool_call.name, tool_call.arguments)
    if pre.is_denied():
        msg = pre.primary_message() or f"Tool '{tool_call.name}' denied by pre-hook"
        logger.info("pre-hook denied %s: %s", tool_call.name, msg)
        return json.dumps({"error": msg})
    if pre.updated_input is not None:
        logger.debug("pre-hook rewrote args for %s", tool_call.name)
        tool_call.arguments = pre.updated_input

    # ── Cache read ────────────────────────────────────────────────────────────
    cached = get_cached(tool_call.name, tool_call.arguments)
    if cached is not None:
        logger.info("cache hit: %s", tool_call.name)
        return cached

    # ── Live execution ────────────────────────────────────────────────────────
    error_str: str | None = None
    result_str: str | None = None
    try:
        logger.info("tool=%s args=%s", tool_call.name, tool_call.arguments)
        result: Any = tool.handler(**tool_call.arguments)
        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    except TypeError as e:
        error_str = f"Tool {tool_call.name!r} called with wrong arguments: {e}"
        logger.warning(error_str)
    except Exception as e:
        logger.exception("Tool %s raised unexpectedly", tool_call.name)
        error_str = str(e)

    # ── Post-hook ─────────────────────────────────────────────────────────────
    if error_str is not None:
        hook_runner.post_failure(tool_call.name, tool_call.arguments, error_str)
        return json.dumps({"error": error_str})

    hook_runner.post_tool_use(tool_call.name, tool_call.arguments, result_str)

    # ── Cache write ───────────────────────────────────────────────────────────
    store(tool_call.name, tool_call.arguments, result_str)

    # ── Cache invalidation after writes ───────────────────────────────────────
    if tool_call.name in _WRITE_TOOLS:
        path = _path_from_args(tool_call.arguments)
        if path:
            invalidate_path(path)

    # ── O4: Tool-specific summarization + observation bounding ───────────────
    result_str = _summarise_result(tool_call.name, result_str, tool_call.arguments)
    return _bound_observation(tool_call.name, result_str)
