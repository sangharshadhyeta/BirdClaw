"""Tool call cache — memoization over the graph store.

Before every tool call, the executor checks here. If a valid cached result
exists, it is returned immediately — no user approval needed for the repeated
operation, no redundant network/disk work.

Cacheable tools (read-only, deterministic enough):
    read_file, glob_search, grep_search, web_fetch, web_search

Never cached (side-effectful):
    bash, write_file, edit_file — always run live.

Cache node in graph:
    type = "tool_result"
    name = "<tool_name>:<args_hash>"   (e.g. "read_file:a3f9...")
    summary = "<tool_name>(<canonical_args>)"
    extra attrs: result (str), args_hash (str), tool_name (str)
    last_seen = timestamp of last successful run

Verification (cheap, no LLM, no user approval):
    read_file / glob_search / grep_search  — os.stat() mtime <= last_seen
    web_fetch / web_search                 — age < WEB_CACHE_TTL_SECONDS

If verification fails, the tool runs live and the cache node is updated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from birdclaw.memory.graph import session_graph

logger = logging.getLogger(__name__)

# Tools that are safe to cache (read-only, no side effects)
CACHEABLE_TOOLS = frozenset({
    "read_file",
    "glob_search",
    "grep_search",
    "web_fetch",
    "web_search",
})

# Web results stay fresh for 1 hour
WEB_CACHE_TTL_SECONDS = 3600

# File-based tools — check mtime against cache timestamp
_FILE_TOOLS = frozenset({"read_file", "glob_search", "grep_search"})
_WEB_TOOLS = frozenset({"web_fetch", "web_search"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args_hash(tool_name: str, arguments: dict) -> str:
    """Stable SHA-256 hash of tool name + canonical args JSON."""
    canonical = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _node_name(tool_name: str, args_hash: str) -> str:
    return f"{tool_name}:{args_hash}"


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_file_tool(tool_name: str, arguments: dict, cached_ts: str) -> bool:
    """Return True if all relevant files are unmodified since cached_ts."""
    ts = _parse_ts(cached_ts)
    if ts is None:
        return False

    paths_to_check: list[str] = []

    if tool_name == "read_file":
        p = arguments.get("path")
        if p:
            paths_to_check.append(p)
    elif tool_name in ("glob_search", "grep_search"):
        p = arguments.get("path")
        if p:
            paths_to_check.append(p)

    if not paths_to_check:
        # No paths to verify — assume stale to be safe
        return False

    for path_str in paths_to_check:
        try:
            mtime = os.stat(path_str).st_mtime
            file_ts = datetime.fromtimestamp(mtime, tz=timezone.utc)
            if file_ts > ts:
                logger.debug("cache stale: %s modified after %s", path_str, cached_ts)
                return False
        except OSError:
            return False

    return True


def _verify_web_tool(cached_ts: str) -> bool:
    """Return True if the cached web result is within TTL."""
    ts = _parse_ts(cached_ts)
    if ts is None:
        return False
    age = (_now_utc() - ts).total_seconds()
    return age < WEB_CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached(tool_name: str, arguments: dict) -> str | None:
    """Check the session graph for a valid cached result.

    Returns the cached result string if valid, None otherwise.
    """
    if tool_name not in CACHEABLE_TOOLS:
        return None

    h = _args_hash(tool_name, arguments)
    node = session_graph.get_node(_node_name(tool_name, h))
    if node is None:
        return None

    cached_ts = node.get("last_seen", "")
    result = node.get("result", "")
    if not result:
        return None

    # Verify freshness
    if tool_name in _FILE_TOOLS:
        valid = _verify_file_tool(tool_name, arguments, cached_ts)
    elif tool_name in _WEB_TOOLS:
        valid = _verify_web_tool(cached_ts)
    else:
        valid = False

    if valid:
        logger.debug("cache hit: %s %s", tool_name, h)
        return result

    logger.debug("cache stale: %s %s", tool_name, h)
    return None


def store(tool_name: str, arguments: dict, result: str) -> None:
    """Store a tool result in the session graph cache."""
    if tool_name not in CACHEABLE_TOOLS:
        return

    h = _args_hash(tool_name, arguments)
    name = _node_name(tool_name, h)
    canonical_args = json.dumps(arguments, sort_keys=True)

    session_graph.upsert_node(
        name=name,
        node_type="tool_result",
        summary=f"{tool_name}({canonical_args[:80]})",
        sources=[],
        result=result,
        args_hash=h,
        tool_name=tool_name,
    )
    logger.debug("cached: %s %s (%d chars)", tool_name, h, len(result))


def invalidate(tool_name: str, arguments: dict) -> None:
    """Explicitly remove a cached entry (e.g. after a write that affects a path)."""
    h = _args_hash(tool_name, arguments)
    removed = session_graph.remove_node(_node_name(tool_name, h))
    if removed:
        logger.debug("invalidated cache: %s %s", tool_name, h)


def invalidate_path(path: str) -> None:
    """Invalidate all file-tool cache entries that cover a given path.

    Called by write_file and edit_file after a successful write so that
    subsequent read_file calls on the same path get fresh content.
    """
    to_remove = []
    for node in session_graph.nodes_by_type("tool_result"):
        if node.get("tool_name") not in _FILE_TOOLS:
            continue
        summary = node.get("summary", "")
        if path in summary:
            to_remove.append(node["key"])

    for key in to_remove:
        session_graph.remove_node(key)
        logger.debug("invalidated cache key %s (path=%s)", key, path)
