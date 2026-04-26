"""Graph MCP server — exposes the knowledge graph as an MCP stdio server.

Implements the MCP JSON-RPC 2.0 stdio transport (server side).
External agents (other BirdClaw instances, Claude Code, etc.) can connect
to this process and call graph_search, graph_get, graph_add, graph_relate.

Usage:
    python main.py graph-server          # reads stdin, writes stdout
    python -m birdclaw.tools.mcp.graph_server

The server is intentionally single-threaded and stateless between requests
(each call resolves against the current on-disk graph state).

JSON-RPC 2.0 protocol:
    → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
    ← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}
    → {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    ← {"jsonrpc":"2.0","id":2,"result":{"tools":[...]}}
    → {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"graph_search","arguments":{"query":"foo"}}}
    ← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}]}}

Content messages (id=null) are sent to stderr to avoid polluting the stdio
transport:
    notifications → stderr logging only
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from birdclaw.tools.registry import registry

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "birdclaw-graph"
_SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _err(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _send(obj: dict) -> None:
    """Write one JSON object to stdout followed by a newline."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

_GRAPH_TOOL_NAMES = {"graph_search", "graph_get", "graph_add", "graph_relate"}
_TASK_TOOL_NAMES  = {"search_tasks", "get_task_output"}


_ALL_MCP_TOOL_NAMES = _GRAPH_TOOL_NAMES | _TASK_TOOL_NAMES


def _mcp_tools() -> list[dict]:
    """Return MCP-formatted tool descriptors for graph + task query tools."""
    result = []
    for tool in registry.all_tools():
        if tool.name not in _ALL_MCP_TOOL_NAMES:
            continue
        s = tool.to_openai_schema()
        fn = s["function"]
        result.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn["parameters"],
        })
    return result


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

def _handle_initialize(id: Any, _params: dict) -> dict:
    return _ok(id, {
        "protocolVersion": _PROTOCOL_VERSION,
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
        "capabilities": {"tools": {}},
    })


def _handle_tools_list(id: Any) -> dict:
    return _ok(id, {"tools": _mcp_tools()})


def _handle_tools_call(id: Any, params: dict) -> dict:
    name = params.get("name", "")
    arguments = params.get("arguments", {})

    if name not in _ALL_MCP_TOOL_NAMES:
        return _err(id, -32601, f"Tool not found: {name!r}")

    tool = registry.get(name)
    if tool is None:
        return _err(id, -32601, f"Tool not registered: {name!r}")

    try:
        text_result = tool.handler(**arguments)
    except TypeError as e:
        return _err(id, -32602, f"Invalid arguments for {name!r}: {e}")
    except Exception as e:
        logger.exception("tool %s raised", name)
        return _err(id, -32603, f"Tool execution error: {e}")

    return _ok(id, {
        "content": [{"type": "text", "text": text_result}],
        "isError": False,
    })


def _handle_ping(id: Any) -> dict:
    return _ok(id, {})


# ---------------------------------------------------------------------------
# Main stdio loop
# ---------------------------------------------------------------------------

def serve() -> None:
    """Read JSON-RPC requests from stdin, write responses to stdout.

    Runs until stdin is closed (EOF) or an unrecoverable error occurs.
    Import graph tools so they register themselves in the registry.
    """
    import birdclaw.tools.graph_tools  # noqa — registers graph tools
    import birdclaw.tools.task_tools   # noqa — registers search_tasks / get_task_output

    logger.debug("%s starting — awaiting JSON-RPC requests on stdin", _SERVER_NAME)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            _send(_err(None, -32700, f"Parse error: {e}"))
            continue

        req_id = req.get("id")  # None for notifications
        method = req.get("method", "")
        params = req.get("params") or {}

        if method == "initialize":
            _send(_handle_initialize(req_id, params))

        elif method == "initialized":
            # Client notification — no response required
            pass

        elif method == "tools/list":
            _send(_handle_tools_list(req_id))

        elif method == "tools/call":
            _send(_handle_tools_call(req_id, params))

        elif method == "ping":
            _send(_handle_ping(req_id))

        elif req_id is not None:
            # Unknown method with an id — must return an error
            _send(_err(req_id, -32601, f"Method not found: {method!r}"))

        # Notifications (no id) for unknown methods are silently ignored per spec


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # keep stdout clean for the transport
    )
    serve()
