"""MCP client — Python port of claw-code-parity/rust/crates/runtime/src/mcp_*.rs

Public API:
    from birdclaw.tools.mcp import mcp_manager
    mcp_manager.load_from_config()          # connect servers from config.toml
    mcp_manager.list_servers()              # → list[McpServerState]
    mcp_manager.call_tool(name, arguments)  # → str (JSON result)
"""

from birdclaw.tools.mcp.manager import mcp_manager, McpServerState

__all__ = ["mcp_manager", "McpServerState"]
