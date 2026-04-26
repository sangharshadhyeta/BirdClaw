"""Approval queue — agent threads request user permission; UI/soul resolves.

When the permission enforcer runs in `prompt` mode and encounters a
non-read-only operation, it calls approval_queue.request(), which:

    1. Registers the request with a unique approval_id
    2. Blocks the agent thread (threading.Event.wait)
    3. Returns the user's decision when resolved

The UI or soul layer calls approval_queue.resolve() to unblock the agent.

Decisions:
    allow_once    — permit this specific call; future calls ask again
    allow_always  — permit this tool+command permanently (stored in memory)
    deny          — reject; agent receives EnforcementResult.deny()

Usage:
    # From a tool (blocks agent thread):
    decision = approval_queue.request(task_id, agent_id, "bash", "rm foo.txt")

    # From TUI or soul (unblocks agent thread):
    approval_queue.resolve("abc123", "allow_once")
    approval_queue.resolve("abc123", "deny")

    # List what's waiting:
    pending = approval_queue.list_pending()
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ApprovalDecision = Literal["allow_once", "allow_always", "deny"]

_DEFAULT_TIMEOUT = 120.0   # seconds before auto-deny

# ---------------------------------------------------------------------------
# Destructive-command heuristic (ported from permission_enforcer.rs)
# ---------------------------------------------------------------------------

# Bash patterns that are potentially destructive — require user confirmation.
_DESTRUCTIVE_BASH_PATTERNS = re.compile(
    r"\brm\s+-[^\s]*r|\brm\s+-[^\s]*f|"   # rm -rf, rm -f
    r"\bdd\b|"                              # disk dump
    r"\bmkfs\b|"                            # format filesystem
    r"\bshred\b|"                           # secure delete
    r"\bkill\b|\bkillall\b|"               # process termination
    r"\bchmod\s+[0-7]*7[0-7]{2}\b|"        # chmod 777 / world-writable
    r"\bchown\b|"                           # ownership change
    r"\biptables\b|\bnftables\b|"           # firewall rules
    r"\bsystemctl\s+(stop|disable|mask)\b|" # service control
    r"\bdrop\s+table|delete\s+from\b|"      # SQL destructive ops
    r"\bgit\s+(push\s+--force|reset\s+--hard|clean\s+-[^\s]*f)\b",
    re.IGNORECASE,
)

# Tools that are always safe (reads, web queries)
_ALWAYS_SAFE_TOOLS = frozenset({"web_search", "web_fetch", "read_file", "think", "glob_search", "grep_search", "find_symbol"})


def _is_destructive(tool_name: str, description: str) -> bool:
    """Return True if this operation requires explicit user approval.

    Non-destructive operations (web, file reads, workspace writes without
    dangerous patterns) are auto-approved with a flash notification.
    Mirrors the dangerous-command heuristic in permission_enforcer.rs.
    """
    if tool_name in _ALWAYS_SAFE_TOOLS:
        return False
    if tool_name == "bash":
        return bool(_DESTRUCTIVE_BASH_PATTERNS.search(description))
    # write_file / edit_file to paths outside workspace: caller enforces boundary;
    # we trust the workspace validator — so these are non-destructive by default.
    return False


@dataclass
class ApprovalRequest:
    approval_id: str
    task_id:     str
    agent_id:    str
    tool_name:   str
    description: str       # human-readable: "bash: rm foo.txt"
    created_at:  float = field(default_factory=time.time)
    expires_at:  float = 0.0

    def __post_init__(self) -> None:
        if not self.expires_at:
            self.expires_at = self.created_at + _DEFAULT_TIMEOUT

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def short_id(self) -> str:
        return self.approval_id[:6]

    def summary(self) -> str:
        tid = self.task_id[:8]
        return (
            f"[{self.short_id()}] task:{tid} · {self.tool_name}: "
            f"{self.description[:80]}"
        )

    def to_dict(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "task_id":     self.task_id,
            "agent_id":    self.agent_id,
            "tool_name":   self.tool_name,
            "description": self.description,
            "expires_in":  max(0, int(self.expires_at - time.time())),
        }


# ---------------------------------------------------------------------------
# Approval queue
# ---------------------------------------------------------------------------

class ApprovalQueue:
    """Thread-safe registry of pending approval requests.

    Agent threads block in request().
    UI / soul threads call resolve() or list_pending().
    """

    def __init__(self) -> None:
        self._lock:    threading.Lock               = threading.Lock()
        self._pending: dict[str, ApprovalRequest]   = {}
        self._events:  dict[str, threading.Event]   = {}
        self._results: dict[str, ApprovalDecision]  = {}
        self._allowed: set[str]                     = set()   # allow_always keys
        self._load_allowed()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _allowed_path() -> Path:
        from birdclaw.config import settings
        return settings.data_dir / "allowed_permissions.json"

    def _load_allowed(self) -> None:
        """Load persisted allow_always decisions from disk."""
        p = self._allowed_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._allowed = set(data.get("allowed", []))
            logger.debug("loaded %d allow_always rules", len(self._allowed))
        except Exception as e:
            logger.warning("could not load allowed_permissions.json: %s", e)

    def _save_allowed(self) -> None:
        """Persist allow_always decisions to disk (called inside lock)."""
        try:
            p = self._allowed_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({"allowed": sorted(self._allowed)}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("could not save allowed_permissions.json: %s", e)

    # ── Agent-side ────────────────────────────────────────────────────────────

    def request(
        self,
        task_id:     str,
        agent_id:    str,
        tool_name:   str,
        description: str,
        timeout:     float = _DEFAULT_TIMEOUT,
    ) -> ApprovalDecision:
        """Register an approval request and block until resolved or timed out.

        Returns the user's decision. Callers must honour `deny` by aborting
        the operation and returning an error observation to the agent loop.
        """
        key = _allow_key(tool_name, description)
        with self._lock:
            already_allowed = key in self._allowed
        if already_allowed:
            logger.debug("approvals: allow_always hit for %s", description[:60])
            return "allow_once"

        # Non-destructive operations: auto-approve immediately and emit a TUI
        # notification. This prevents blocking the agent on routine tool use
        # (workspace reads/writes, web searches, etc.).
        if not _is_destructive(tool_name, description):
            logger.info("approvals: auto-approve non-destructive %s — %s", tool_name, description[:60])
            # Signal the TUI to flash a notification (best-effort — no blocking)
            try:
                from birdclaw.gateway.events import emit_approval_flash
                emit_approval_flash(task_id, tool_name, description)
            except Exception:
                pass
            return "allow_once"

        approval_id = uuid.uuid4().hex[:12]
        req = ApprovalRequest(
            approval_id=approval_id,
            task_id=task_id,
            agent_id=agent_id,
            tool_name=tool_name,
            description=description,
            expires_at=time.time() + timeout,
        )
        evt = threading.Event()

        with self._lock:
            self._pending[approval_id] = req
            self._events[approval_id]  = evt

        logger.info("approvals: waiting %s — %s", req.short_id(), req.summary())

        triggered = evt.wait(timeout=timeout + 2.0)   # small buffer

        with self._lock:
            decision = self._results.pop(approval_id, None)
            self._pending.pop(approval_id, None)
            self._events.pop(approval_id, None)

        if not triggered or decision is None:
            logger.warning("approvals: timeout → deny %s", approval_id[:6])
            return "deny"

        if decision == "allow_always":
            with self._lock:
                self._allowed.add(key)
                self._save_allowed()
            logger.info("approvals: allow_always persisted for %s %s", tool_name, description[:40])

        return decision

    # ── UI / soul-side ────────────────────────────────────────────────────────

    def resolve(self, approval_id: str, decision: ApprovalDecision) -> bool:
        """Resolve a pending approval by ID or prefix. Returns True if found."""
        with self._lock:
            matched = next(
                (aid for aid in self._pending if aid.startswith(approval_id)),
                None,
            )
            if matched is None:
                return False
            self._results[matched] = decision
            evt = self._events.get(matched)

        if evt:
            evt.set()
        logger.info("approvals: resolved %s → %s", matched[:6], decision)
        return True

    def list_pending(self) -> list[ApprovalRequest]:
        """Return all non-expired pending requests."""
        with self._lock:
            return [r for r in self._pending.values() if not r.is_expired()]

    def get(self, approval_id: str) -> ApprovalRequest | None:
        """Fetch a pending request by ID or prefix."""
        with self._lock:
            for aid, req in self._pending.items():
                if aid.startswith(approval_id):
                    return req
        return None

    def deny_all_for_task(self, task_id: str) -> int:
        """Deny all pending approvals for a task (called on interrupt). Returns count."""
        denied = 0
        with self._lock:
            for aid, req in list(self._pending.items()):
                if req.task_id == task_id:
                    self._results[aid] = "deny"
                    evt = self._events.get(aid)
                    if evt:
                        evt.set()
                    denied += 1
        logger.info("approvals: denied %d request(s) for task %s", denied, task_id[:8])
        return denied


# ---------------------------------------------------------------------------
# Module singleton + helpers
# ---------------------------------------------------------------------------

def _allow_key(tool_name: str, description: str) -> str:
    return f"{tool_name}:{description}"


approval_queue = ApprovalQueue()
