"""McpServerManager + McpToolBridge — Python port of mcp_tool_bridge.rs.

McpServerManager:
    Manages multiple McpClient instances (one per configured server).
    Loads server configs from config.toml or BC_MCP_SERVERS env var.
    Thread-safe: all mutations serialised through threading.Lock.

McpToolBridge:
    Registers each MCP server's tools into BirdClaw's tool registry
    so the agent loop can call them like any other tool.
    Tool names use mcp_tool_name() convention: mcp__<server>__<tool>

Config:
    ~/.birdclaw/config.toml:
        [mcp_servers.filesystem]
        command = "uvx"
        args    = ["mcp-server-filesystem", "/home/user/workspace"]

        [mcp_servers.sqlite]
        command = "uvx"
        args    = ["mcp-server-sqlite", "--db-path", "/tmp/data.db"]
        env     = { SQLITE_PATH = "/tmp/data.db" }

    Or via env var (simple format, no env vars per server):
        BC_MCP_SERVERS="filesystem:uvx mcp-server-filesystem /tmp,sqlite:uvx mcp-server-sqlite"

Reference: claw-code-parity/rust/crates/runtime/src/mcp_tool_bridge.rs
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Literal

from birdclaw.tools.mcp.client import McpClient, McpError, McpToolInfo, McpResourceInfo
from birdclaw.tools.mcp.naming import mcp_tool_name, normalize_name_for_mcp
from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)

McpConnectionStatus = Literal["disconnected", "connecting", "connected", "error"]


# ---------------------------------------------------------------------------
# Server state (mirrors McpServerState in mcp_tool_bridge.rs)
# ---------------------------------------------------------------------------

@dataclass
class McpServerState:
    server_name:   str
    status:        McpConnectionStatus      = "disconnected"
    tools:         list[McpToolInfo]        = field(default_factory=list)
    resources:     list[McpResourceInfo]    = field(default_factory=list)
    server_info:   str                      = ""
    error_message: str                      = ""


# ---------------------------------------------------------------------------
# Server config (from config.toml or env var)
# ---------------------------------------------------------------------------

@dataclass
class McpStdioConfig:
    command: str
    args:    list[str]       = field(default_factory=list)
    env:     dict[str, str]  = field(default_factory=dict)
    tool_call_timeout_ms: int = 60_000


# ---------------------------------------------------------------------------
# McpServerManager
# ---------------------------------------------------------------------------

class McpServerManager:
    """Thread-safe registry of MCP server connections.

    Usage:
        manager = McpServerManager()
        manager.add_server("filesystem", McpStdioConfig("uvx", ["mcp-server-filesystem", "/tmp"]))
        manager.connect_all()
        tools   = manager.all_tools()
        result  = manager.call_tool("mcp__filesystem__read_file", {"path": "/tmp/hi.txt"})
        manager.disconnect_all()
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._clients:  dict[str, McpClient]      = {}
        self._states:   dict[str, McpServerState] = {}

    # ── Config loading ────────────────────────────────────────────────────────

    def load_from_config(self) -> None:
        """Load server configs from config.toml [mcp_servers] and BC_MCP_SERVERS env var."""
        configs = self._configs_from_toml()
        configs.update(self._configs_from_env())
        for name, cfg in configs.items():
            self.add_server(name, cfg)
        if configs:
            self.connect_all()
        else:
            logger.debug("no MCP servers configured")

    def _configs_from_toml(self) -> dict[str, McpStdioConfig]:
        from birdclaw.config import settings
        path = settings.data_dir.parent / "config.toml"
        if not path.exists():
            return {}
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                logger.warning("tomllib/tomli not available — cannot load MCP config from TOML")
                return {}
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            servers = data.get("mcp_servers", {})
            result = {}
            for name, cfg in servers.items():
                if not isinstance(cfg, dict) or "command" not in cfg:
                    continue
                result[name] = McpStdioConfig(
                    command=cfg["command"],
                    args=list(cfg.get("args", [])),
                    env=dict(cfg.get("env", {})),
                    tool_call_timeout_ms=int(cfg.get("tool_call_timeout_ms", 60_000)),
                )
            return result
        except Exception as e:
            logger.warning("failed to load MCP config from %s: %s", path, e)
            return {}

    def _configs_from_env(self) -> dict[str, McpStdioConfig]:
        """Parse BC_MCP_SERVERS="name1:cmd arg1 arg2,name2:cmd" env var."""
        raw = os.environ.get("BC_MCP_SERVERS", "").strip()
        if not raw:
            return {}
        result = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            name, rest = entry.split(":", 1)
            parts = rest.strip().split()
            if not parts:
                continue
            result[name.strip()] = McpStdioConfig(command=parts[0], args=parts[1:])
        return result

    # ── Server management ─────────────────────────────────────────────────────

    def add_server(self, name: str, cfg: McpStdioConfig) -> None:
        with self._lock:
            self._clients[name] = McpClient(
                server_name=name,
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
                tool_call_timeout_ms=cfg.tool_call_timeout_ms,
            )
            self._states[name] = McpServerState(server_name=name, status="disconnected")

    def connect_all(self) -> None:
        """Connect all registered servers (in parallel threads)."""
        import concurrent.futures
        with self._lock:
            names = list(self._clients.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(names), 8)) as pool:
            futures = {pool.submit(self._connect_one, n): n for n in names}
            for f in concurrent.futures.as_completed(futures):
                name = futures[f]
                try:
                    f.result()
                except Exception as e:
                    logger.warning("MCP server %r failed to connect: %s", name, e)

    def _connect_one(self, name: str) -> None:
        with self._lock:
            client = self._clients.get(name)
            state  = self._states.get(name)
        if client is None or state is None:
            return
        state.status = "connecting"
        try:
            info   = client.connect()
            tools  = client.list_tools()
            resources = client.list_resources()
            with self._lock:
                state.status      = "connected"
                state.tools       = tools
                state.resources   = resources
                state.server_info = f"{info.name} {info.version}".strip()
                state.error_message = ""
            logger.info(
                "MCP server %r connected: %d tools, %d resources",
                name, len(tools), len(resources),
            )
        except Exception as e:
            with self._lock:
                state.status        = "error"
                state.error_message = str(e)
            raise

    def disconnect_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_servers(self) -> list[McpServerState]:
        with self._lock:
            return list(self._states.values())

    def all_tools(self) -> list[tuple[str, McpToolInfo]]:
        """Return (server_name, tool_info) pairs for all connected servers."""
        with self._lock:
            result = []
            for name, state in self._states.items():
                if state.status == "connected":
                    for tool in state.tools:
                        result.append((name, tool))
            return result

    def server_for_tool(self, qualified_name: str) -> McpClient | None:
        """Find the client that owns a qualified tool name."""
        with self._lock:
            for server_name, state in self._states.items():
                if state.status != "connected":
                    continue
                for tool in state.tools:
                    if mcp_tool_name(server_name, tool.name) == qualified_name:
                        return self._clients[server_name]
        return None

    # ── Tool calling ──────────────────────────────────────────────────────────

    def call_tool(self, qualified_name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool by its qualified name. Returns JSON string."""
        client = self.server_for_tool(qualified_name)
        if client is None:
            return json.dumps({"error": f"MCP tool not found: {qualified_name}"})
        # Find raw tool name
        raw_name = qualified_name.split("__", 2)[-1] if "__" in qualified_name else qualified_name
        # Reverse-normalise: find original name from tool list
        for tool in client.tools:
            if normalize_name_for_mcp(tool.name) == raw_name or tool.name == raw_name:
                raw_name = tool.name
                break
        try:
            result = client.call_tool(raw_name, arguments)
            if result.is_error:
                return json.dumps({"error": result.text(), "is_error": True})
            return json.dumps({
                "content": result.content,
                "structured_content": result.structured_content,
            })
        except McpError as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# McpToolBridge — registers MCP tools into BirdClaw's tool registry
# ---------------------------------------------------------------------------

class McpToolBridge:
    """Registers tools from all connected MCP servers into the tool registry.

    Called after McpServerManager.connect_all(). Each tool gets:
    - name:        mcp__<server>__<tool>
    - description: from MCP server
    - schema:      from MCP server inputSchema
    - handler:     closure that calls manager.call_tool()
    - tags:        ["mcp", server_name]
    """

    def __init__(self, manager: McpServerManager) -> None:
        self._manager = manager
        self._registered: set[str] = set()

    def register_all(self) -> int:
        """Register all connected MCP tools. Returns count of newly registered tools."""
        count = 0
        for server_name, tool_info in self._manager.all_tools():
            qualified = mcp_tool_name(server_name, tool_info.name)
            if qualified in self._registered:
                continue
            self._register_one(server_name, qualified, tool_info)
            self._registered.add(qualified)
            count += 1
        if count:
            logger.info("MCP bridge: registered %d tools", count)
        return count

    def _register_one(self, server_name: str, qualified: str, tool: McpToolInfo) -> None:
        description = tool.description or f"MCP tool from {server_name}"
        schema = tool.to_registry_schema()
        manager = self._manager

        def handler(**kwargs) -> str:
            return manager.call_tool(qualified, kwargs or None)

        registry.register(Tool(
            name=qualified,
            description=description,
            input_schema=schema,
            handler=handler,
            tags=["mcp", server_name, normalize_name_for_mcp(server_name)],
        ))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

mcp_manager = McpServerManager()
mcp_bridge  = McpToolBridge(mcp_manager)
