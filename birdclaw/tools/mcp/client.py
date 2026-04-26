"""McpClient — JSON-RPC 2.0 over stdio subprocess.

Python port of mcp_stdio.rs: spawns an MCP server process, performs the
initialize handshake, lists tools/resources, and calls tools.

Transport: stdio only (SSE/WebSocket/OAuth are future ports).

Concurrency model:
    Runs in a dedicated asyncio event loop on a background daemon thread.
    Public API is fully synchronous — callers use connect(), list_tools(),
    call_tool(), list_resources(), read_resource() without async/await.
    Internally uses asyncio.run_coroutine_threadsafe() to dispatch into
    the background loop.

Reference: claw-code-parity/rust/crates/runtime/src/mcp_stdio.rs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Timeouts (ms) — match Rust constants
_INIT_TIMEOUT_MS      = 10_000
_LIST_TOOLS_TIMEOUT_MS = 30_000
_CALL_TOOL_TIMEOUT_MS  = 60_000
_LIST_RES_TIMEOUT_MS   = 30_000

MCP_PROTOCOL_VERSION = "2024-11-05"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class McpError(Exception):
    """Raised when an MCP server returns a JSON-RPC error or times out."""


class McpTransportError(McpError):
    """Raised on subprocess I/O failures."""


class McpTimeoutError(McpError):
    """Raised when a request exceeds its timeout."""


class McpJsonRpcError(McpError):
    """Raised when the server returns a JSON-RPC error object."""
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code    = code
        self.message = message
        self.data    = data


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class McpToolInfo:
    name:         str
    description:  str                = ""
    input_schema: dict | None        = None

    def to_registry_schema(self) -> dict:
        """Convert to the schema format used by birdclaw's tool registry."""
        return self.input_schema or {"type": "object", "properties": {}, "required": []}


@dataclass
class McpResourceInfo:
    uri:         str
    name:        str        = ""
    description: str        = ""
    mime_type:   str | None = None


@dataclass
class McpServerInfo:
    name:    str = ""
    version: str = ""


@dataclass
class McpToolCallResult:
    content:            list[dict] = field(default_factory=list)
    structured_content: Any        = None
    is_error:           bool       = False

    def text(self) -> str:
        """Extract text content from all content blocks."""
        parts = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else json.dumps(self.content)


# ---------------------------------------------------------------------------
# Background event loop (module-level singleton)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            t = threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-event-loop")
            t.start()
            # Give the loop a moment to start
            time.sleep(0.05)
    return _loop


def _run_sync(coro, timeout_s: float = 30.0):
    """Run a coroutine on the background loop, block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout_s)


# ---------------------------------------------------------------------------
# McpClient
# ---------------------------------------------------------------------------

class McpClient:
    """Manages one MCP server subprocess and speaks JSON-RPC 2.0 over stdio.

    Usage:
        client = McpClient("filesystem", "uvx", ["mcp-server-filesystem", "/tmp"])
        client.connect()
        tools  = client.list_tools()
        result = client.call_tool("read_file", {"path": "/tmp/hello.txt"})
        client.close()
    """

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tool_call_timeout_ms: int = _CALL_TOOL_TIMEOUT_MS,
    ) -> None:
        self.server_name          = server_name
        self.command              = command
        self.args                 = args or []
        self.env                  = env or {}
        self.tool_call_timeout_ms = tool_call_timeout_ms

        self._proc:    asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._req_id:  int  = 0
        self._lock:    asyncio.Lock | None = None
        self.server_info: McpServerInfo = McpServerInfo()
        self._tools:     list[McpToolInfo]     = []
        self._resources: list[McpResourceInfo] = []

    # ── Public sync API ───────────────────────────────────────────────────────

    def connect(self) -> McpServerInfo:
        return _run_sync(self._connect(), timeout_s=_INIT_TIMEOUT_MS / 1000 + 5)

    def list_tools(self) -> list[McpToolInfo]:
        return _run_sync(self._list_tools(), timeout_s=_LIST_TOOLS_TIMEOUT_MS / 1000 + 5)

    def call_tool(self, name: str, arguments: dict | None = None) -> McpToolCallResult:
        timeout_s = self.tool_call_timeout_ms / 1000 + 5
        return _run_sync(self._call_tool(name, arguments), timeout_s=timeout_s)

    def list_resources(self) -> list[McpResourceInfo]:
        return _run_sync(self._list_resources(), timeout_s=_LIST_RES_TIMEOUT_MS / 1000 + 5)

    def read_resource(self, uri: str) -> list[dict]:
        return _run_sync(self._read_resource(uri), timeout_s=_LIST_RES_TIMEOUT_MS / 1000 + 5)

    def close(self) -> None:
        _run_sync(self._close(), timeout_s=5)

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    @property
    def resources(self) -> list[McpResourceInfo]:
        return list(self._resources)

    # ── Async internals ───────────────────────────────────────────────────────

    async def _connect(self) -> McpServerInfo:
        if self._lock is None:
            self._lock = asyncio.Lock()

        merged_env = {**os.environ, **self.env}
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        logger.debug("MCP server %r spawned (pid=%d)", self.server_name, self._proc.pid)

        # Start background reader
        loop = asyncio.get_event_loop()
        self._reader_task = loop.create_task(self._read_loop())

        # initialize handshake
        result = await asyncio.wait_for(
            self._request("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "BirdClaw", "version": "0.1.0"},
            }),
            timeout=_INIT_TIMEOUT_MS / 1000,
        )
        info = result.get("serverInfo", {})
        self.server_info = McpServerInfo(
            name=info.get("name", self.server_name),
            version=info.get("version", ""),
        )

        # Send initialized notification (no response expected)
        await self._notify("notifications/initialized")
        logger.debug(
            "MCP server %r connected: %s %s",
            self.server_name, self.server_info.name, self.server_info.version,
        )
        return self.server_info

    async def _list_tools(self) -> list[McpToolInfo]:
        tools: list[McpToolInfo] = []
        cursor: str | None = None
        while True:
            params: dict = {}
            if cursor:
                params["cursor"] = cursor
            result = await asyncio.wait_for(
                self._request("tools/list", params or None),
                timeout=_LIST_TOOLS_TIMEOUT_MS / 1000,
            )
            for t in result.get("tools", []):
                tools.append(McpToolInfo(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema"),
                ))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self._tools = tools
        logger.debug("MCP server %r: %d tools listed", self.server_name, len(tools))
        return tools

    async def _call_tool(self, name: str, arguments: dict | None) -> McpToolCallResult:
        params: dict = {"name": name}
        if arguments:
            params["arguments"] = arguments
        result = await asyncio.wait_for(
            self._request("tools/call", params),
            timeout=self.tool_call_timeout_ms / 1000,
        )
        return McpToolCallResult(
            content=result.get("content", []),
            structured_content=result.get("structuredContent"),
            is_error=bool(result.get("isError", False)),
        )

    async def _list_resources(self) -> list[McpResourceInfo]:
        resources: list[McpResourceInfo] = []
        cursor: str | None = None
        while True:
            params: dict = {}
            if cursor:
                params["cursor"] = cursor
            try:
                result = await asyncio.wait_for(
                    self._request("resources/list", params or None),
                    timeout=_LIST_RES_TIMEOUT_MS / 1000,
                )
            except McpJsonRpcError as e:
                # Some servers don't implement resources — gracefully return empty
                if e.code == -32601:  # Method not found
                    break
                raise
            for r in result.get("resources", []):
                resources.append(McpResourceInfo(
                    uri=r["uri"],
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType"),
                ))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self._resources = resources
        return resources

    async def _read_resource(self, uri: str) -> list[dict]:
        result = await asyncio.wait_for(
            self._request("resources/read", {"uri": uri}),
            timeout=_LIST_RES_TIMEOUT_MS / 1000,
        )
        return result.get("contents", [])

    async def _close(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
        logger.debug("MCP server %r closed", self.server_name)

    # ── JSON-RPC helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _request(self, method: str, params: Any = None) -> Any:
        """Send a JSON-RPC request and await the response."""
        if self._proc is None or self._proc.stdin is None:
            raise McpTransportError(f"server {self.server_name!r} is not connected")
        req_id = self._next_id()
        msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()
        return await future

    async def _notify(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if self._proc is None or self._proc.stdin is None:
            return
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        """Background task: read lines from server stdout and resolve futures."""
        assert self._proc and self._proc.stdout
        try:
            async for raw_line in self._proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("MCP server %r: malformed JSON: %s", self.server_name, line[:200])
                    continue

                req_id = msg.get("id")
                if req_id is None:
                    # Notification from server — ignore for now
                    logger.debug("MCP server %r notification: %s", self.server_name, msg.get("method"))
                    continue

                future = self._pending.pop(req_id, None)
                if future is None:
                    logger.warning("MCP server %r: unexpected response id %s", self.server_name, req_id)
                    continue

                if not future.done():
                    if err := msg.get("error"):
                        future.set_exception(McpJsonRpcError(
                            code=err.get("code", -1),
                            message=err.get("message", "unknown error"),
                            data=err.get("data"),
                        ))
                    else:
                        future.set_result(msg.get("result"))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("MCP server %r read loop error: %s", self.server_name, e)
        finally:
            # Fail all pending requests
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(McpTransportError(
                        f"MCP server {self.server_name!r} connection closed"
                    ))
            self._pending.clear()
