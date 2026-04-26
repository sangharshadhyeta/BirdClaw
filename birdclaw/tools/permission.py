"""Permission enforcer.

Wraps the full bash validation pipeline from bash_validation.py and adds
file-write workspace boundary checks.

Modes:
    read_only          — only safe read-only commands; no file writes
    workspace_write    — file writes inside workspace_roots only (default)
    danger_full_access — unrestricted (still warns on fork bombs etc.)
    prompt             — read-only ops pass; mutating ops block in approval queue
    allow              — dev alias for danger_full_access
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from birdclaw.config import settings
from birdclaw.tools.bash_validation import (
    Allow, Block, Warn,
    validate_command,
    classify_command, CommandIntent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tool minimum permission mode
# ---------------------------------------------------------------------------

# Maps tool name → minimum mode required to invoke it.
# Tools not listed here default to "workspace_write".
# Modes ranked by permissiveness: read_only < workspace_write < danger_full_access
_TOOL_MIN_MODE: dict[str, str] = {
    # Pure read / reasoning — always allowed
    "think":          "read_only",
    "answer":         "read_only",
    "graph_search":   "read_only",
    "web_search":     "read_only",
    "web_fetch":      "read_only",
    "read_file":      "read_only",
    "list_dir":       "read_only",
    "code_search":    "read_only",

    # Write ops — need at least workspace_write
    "bash":           "workspace_write",
    "write_file":     "workspace_write",
    "edit_file":      "workspace_write",
    "delete_file":    "workspace_write",
    "graph_add":      "workspace_write",
    "graph_relate":   "workspace_write",
    "graph_delete":   "workspace_write",

    # Self-modification — need danger_full_access
    "self_edit":      "danger_full_access",
    "self_exec":      "danger_full_access",
}

_MODE_RANK: dict[str, int] = {
    "read_only":          0,
    "workspace_write":    1,
    "prompt":             1,   # treated like workspace_write for ranking
    "danger_full_access": 2,
    "allow":              2,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class EnforcementResult:
    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    @classmethod
    def ok(cls) -> "EnforcementResult":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> "EnforcementResult":
        return cls(allowed=False, reason=reason)

    @classmethod
    def warn(cls, message: str) -> "EnforcementResult":
        """Warns are allowed but the message is surfaced to the caller."""
        return cls(allowed=True, reason=message)

    def __bool__(self) -> bool:
        return self.allowed

    def __str__(self) -> str:
        if self.allowed:
            return f"allowed" + (f" (warn: {self.reason})" if self.reason else "")
        return f"permission denied: {self.reason}"


# ---------------------------------------------------------------------------
# Permission enforcer
# ---------------------------------------------------------------------------

class PermissionEnforcer:
    def __init__(self, mode: str | None = None) -> None:
        self.mode = mode or settings.permission_mode

    def check_bash(self, command: str) -> EnforcementResult:
        """Run the full bash validation pipeline against the current mode.

        danger_full_access / allow: still runs destructive checks (warns only).
        prompt: read-only commands pass immediately; mutating commands block the
                calling agent thread in the approval queue until the user decides.
        """
        workspace = settings.workspace_roots[0] if settings.workspace_roots else Path.cwd()

        if self.mode in ("danger_full_access", "allow"):
            from birdclaw.tools.bash_validation import check_destructive
            result = check_destructive(command)
            if isinstance(result, Warn):
                return EnforcementResult.warn(result.message)
            return EnforcementResult.ok()

        if self.mode == "prompt":
            intent = classify_command(command)
            if intent in (CommandIntent.ReadOnly, CommandIntent.Network):
                return EnforcementResult.ok()
            return self._request_approval("bash", command[:200])

        result = validate_command(command, self.mode, workspace)

        if isinstance(result, Allow):
            pass  # still need to check write paths below
        elif isinstance(result, Warn):
            pass
        elif isinstance(result, Block):
            return EnforcementResult.deny(result.reason)

        # Workspace boundary: check any absolute paths being written via redirection/tee.
        if self.mode == "workspace_write":
            for write_path in _extract_bash_write_paths(command):
                path_result = self.check_file_write(write_path)
                if not path_result.allowed:
                    return path_result

        if isinstance(result, Warn):
            return EnforcementResult.warn(result.message)
        return EnforcementResult.ok()

    def check_file_write(self, path: Path) -> EnforcementResult:
        """Check whether writing to path is allowed under current mode."""
        if self.mode in ("danger_full_access", "allow"):
            return EnforcementResult.ok()

        if self.mode == "prompt":
            return self._request_approval("file_write", str(path))

        if self.mode == "read_only":
            return EnforcementResult.deny(
                "file writes are not allowed in read_only mode"
            )

        if self.mode == "workspace_write":
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path.absolute()

            # data_dir (~/.birdclaw) is always writable — the agent owns its own data store
            try:
                data_resolved = settings.data_dir.resolve()
                if resolved == data_resolved or resolved.is_relative_to(data_resolved):
                    return EnforcementResult.ok()
            except OSError:
                pass

            # src_dir (birdclaw package source) — writable only when BC_SELF_MODIFY=1
            try:
                src_resolved = settings.src_dir.resolve()
                if resolved == src_resolved or resolved.is_relative_to(src_resolved):
                    if settings.self_modify:
                        return EnforcementResult.ok()
                    return EnforcementResult.deny(
                        "writes to birdclaw source require BC_SELF_MODIFY=1"
                    )
            except OSError:
                pass

            for root in settings.workspace_roots:
                try:
                    root_resolved = root.resolve()
                except OSError:
                    root_resolved = root.absolute()
                if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                    return EnforcementResult.ok()

            roots_str = ", ".join(str(r) for r in settings.workspace_roots)
            return EnforcementResult.deny(
                f"path {path} is outside workspace roots ({roots_str})"
            )

        return EnforcementResult.ok()

    def check_tool(self, tool_name: str) -> EnforcementResult:
        """Check whether the named tool may be invoked under the current mode.

        Uses _TOOL_MIN_MODE to determine the minimum required mode.
        In prompt mode, write-level tools are routed to the approval queue.
        """
        min_mode = _TOOL_MIN_MODE.get(tool_name, "workspace_write")
        required_rank = _MODE_RANK.get(min_mode, 1)
        current_rank  = _MODE_RANK.get(self.mode, 1)

        if self.mode in ("danger_full_access", "allow"):
            return EnforcementResult.ok()

        if self.mode == "prompt":
            if required_rank == 0:
                return EnforcementResult.ok()
            return self._request_approval(tool_name, f"invoke tool {tool_name}")

        if current_rank < required_rank:
            return EnforcementResult.deny(
                f"tool '{tool_name}' requires mode '{min_mode}' but current mode is '{self.mode}'"
            )
        return EnforcementResult.ok()

    def is_bash_cacheable(self, command: str) -> bool:
        """Return True if this bash command is read-only and safe to cache."""
        intent = classify_command(command)
        return intent in (
            CommandIntent.ReadOnly,
            CommandIntent.Network,
        )

    # ── Approval queue integration ────────────────────────────────────────────

    def _request_approval(self, tool_name: str, description: str) -> EnforcementResult:
        """Push to the approval queue and block the calling agent thread.

        Returns ok() if approved, deny() if denied or timed out.
        Falls back to deny if no task context is available (e.g. interactive use).
        """
        from birdclaw.agent.approvals import approval_queue
        from birdclaw.tools.context_vars import get_task_id, get_agent_id

        task_id  = get_task_id()  or "interactive"
        agent_id = get_agent_id() or "unknown"

        logger.info(
            "permission: prompt mode — queuing approval for task %s: %s %s",
            task_id[:8], tool_name, description[:60],
        )

        decision = approval_queue.request(
            task_id=task_id,
            agent_id=agent_id,
            tool_name=tool_name,
            description=description,
        )

        if decision in ("allow_once", "allow_always"):
            return EnforcementResult.ok()
        return EnforcementResult.deny(
            f"user denied approval for {tool_name}: {description[:60]}"
        )


# ---------------------------------------------------------------------------
# Bash write-path extraction
# ---------------------------------------------------------------------------

# Matches output redirections (> /path or >> /path) and tee targets.
# Intentionally conservative: only matches absolute paths to avoid false positives.
_REDIRECT_RE = re.compile(r"(?:>>?|tee\s+(?:-a\s+)?)\s*(\/[^\s;|&\"']+)")


def _extract_bash_write_paths(command: str) -> list[Path]:
    """Return absolute paths that a bash command would write to."""
    return [Path(m) for m in _REDIRECT_RE.findall(command)]


# Module-level singleton
enforcer = PermissionEnforcer()
