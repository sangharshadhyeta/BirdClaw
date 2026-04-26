"""MCP naming utilities — Python port of mcp.rs.

normalize_name_for_mcp  →  sanitise server/tool names to [a-zA-Z0-9_-]
mcp_tool_prefix         →  "mcp__<server>__"
mcp_tool_name           →  "mcp__<server>__<tool>"

These names appear as tool names in the LLM's tool list, so they must be
stable and unambiguous even when server names contain spaces or punctuation.

Reference: claw-code-parity/rust/crates/runtime/src/mcp.rs
"""

from __future__ import annotations

import re


def normalize_name_for_mcp(name: str) -> str:
    """Replace non-alphanumeric/underscore/hyphen chars with underscores.

    Mirrors Rust normalize_name_for_mcp() exactly:
      - 'my server' → 'my_server'
      - 'github.com' → 'github_com'
      - 'tool name!' → 'tool_name_'
    """
    normalized = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Collapse consecutive underscores (only for claude.ai prefixed names in Rust,
    # but we apply universally for cleanliness)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "server"


def mcp_tool_prefix(server_name: str) -> str:
    """Return the tool name prefix for a server: 'mcp__<normalized_name>__'"""
    return f"mcp__{normalize_name_for_mcp(server_name)}__"


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Return the fully qualified tool name visible to the LLM.

    Example: mcp_tool_name('my server', 'read_file') → 'mcp__my_server__read_file'
    """
    return f"{mcp_tool_prefix(server_name)}{normalize_name_for_mcp(tool_name)}"


def server_name_from_tool(qualified_name: str) -> str | None:
    """Extract server name from a qualified tool name, or None if not an MCP tool."""
    if not qualified_name.startswith("mcp__"):
        return None
    parts = qualified_name[5:].split("__", 1)
    return parts[0] if parts else None
