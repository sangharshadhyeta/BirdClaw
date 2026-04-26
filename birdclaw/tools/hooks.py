"""Pre/post tool use hooks.

Full port of claw-code-parity/rust/crates/runtime/src/hooks.rs.

Hooks are shell commands configured via environment variables or config.
They run before/after every tool call. They can:
    - Allow the tool call (exit 0, no JSON or JSON with continue=true)
    - Deny the tool call (exit 2, or JSON with continue=false / decision=block)
    - Rewrite the tool arguments before execution (hookSpecificOutput.updatedInput)
    - Grant/deny permission override (hookSpecificOutput.permissionDecision)
    - Fail silently (exit 1 — logged, execution continues)

Configuration (env vars or ~/.birdclaw/config.toml):
    BC_HOOKS_PRE_TOOL_USE   — comma-separated shell commands
    BC_HOOKS_POST_TOOL_USE  — comma-separated shell commands
    BC_HOOKS_POST_FAILURE   — comma-separated shell commands

Each hook receives a JSON payload on stdin:
    {
        "hook_event_name": "PreToolUse",
        "tool_name": "bash",
        "tool_input": {...},          # parsed args
        "tool_input_json": "...",     # raw JSON string
        "tool_output": "...",         # PostToolUse only
        "tool_result_is_error": false
    }

Hook response (stdout, optional JSON):
    {
        "continue": false,            # deny (alternative to exit 2)
        "decision": "block",          # deny
        "systemMessage": "reason",    # message to surface
        "reason": "reason",           # alternative message field
        "hookSpecificOutput": {
            "additionalContext": "...",
            "permissionDecision": "allow" | "deny" | "ask",
            "permissionDecisionReason": "...",
            "updatedInput": {...}      # PreToolUse only — replaces args
        }
    }

Exit codes:
    0 — allow (check JSON for deny flag)
    2 — deny
    other — failure (logged, does not block)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HOOK_TIMEOUT_S = 30.0  # max time per hook command
_POLL_INTERVAL_S = 0.02  # 20ms poll loop (matches Rust)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HookEvent(Enum):
    PreToolUse       = "PreToolUse"
    PostToolUse      = "PostToolUse"
    PostToolUseFailure = "PostToolUseFailure"


# ---------------------------------------------------------------------------
# Progress reporting (mirrors HookProgressReporter trait + HookProgressEvent)
# ---------------------------------------------------------------------------

@dataclass
class HookProgressEvent:
    kind:      str        # "started" | "completed" | "cancelled"
    event:     "HookEvent"
    tool_name: str
    command:   str

    @classmethod
    def started(cls, event: "HookEvent", tool_name: str, command: str) -> "HookProgressEvent":
        return cls("started", event, tool_name, command)

    @classmethod
    def completed(cls, event: "HookEvent", tool_name: str, command: str) -> "HookProgressEvent":
        return cls("completed", event, tool_name, command)

    @classmethod
    def cancelled(cls, event: "HookEvent", tool_name: str, command: str) -> "HookProgressEvent":
        return cls("cancelled", event, tool_name, command)


class HookProgressReporter(ABC):
    @abstractmethod
    def on_event(self, event: HookProgressEvent) -> None: ...


class PermissionDecision(Enum):
    Allow = auto()
    Deny  = auto()
    Ask   = auto()


@dataclass
class HookRunResult:
    allowed: bool
    failed: bool = False
    cancelled: bool = False
    messages: list[str] = field(default_factory=list)
    permission_decision: PermissionDecision | None = None
    permission_reason: str = ""
    updated_input: dict | None = None  # PreToolUse only — replaces tool arguments

    @classmethod
    def allow(cls, messages: list[str] | None = None) -> "HookRunResult":
        return cls(allowed=True, messages=messages or [])

    def is_denied(self) -> bool:
        return not self.allowed

    def is_failed(self) -> bool:
        return self.failed

    def is_cancelled(self) -> bool:
        return self.cancelled

    def primary_message(self) -> str:
        return self.messages[0] if self.messages else ""


# ---------------------------------------------------------------------------
# Abort signal (thread-safe)
# ---------------------------------------------------------------------------

class HookAbortSignal:
    def __init__(self) -> None:
        self._aborted = threading.Event()

    def abort(self) -> None:
        self._aborted.set()

    def is_aborted(self) -> bool:
        return self._aborted.is_set()


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _build_payload(
    event: HookEvent,
    tool_name: str,
    tool_input: dict,
    tool_output: str | None,
    is_error: bool,
) -> str:
    tool_input_json = json.dumps(tool_input, ensure_ascii=False)

    if event == HookEvent.PostToolUseFailure:
        data = {
            "hook_event_name": event.value,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_input_json": tool_input_json,
            "tool_error": tool_output,
            "tool_result_is_error": True,
        }
    else:
        data = {
            "hook_event_name": event.value,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_input_json": tool_input_json,
            "tool_output": tool_output,
            "tool_result_is_error": is_error,
        }
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Hook output parsing
# ---------------------------------------------------------------------------

@dataclass
class _ParsedOutput:
    messages: list[str] = field(default_factory=list)
    deny: bool = False
    permission_decision: PermissionDecision | None = None
    permission_reason: str = ""
    updated_input: dict | None = None


def _parse_hook_output(stdout: str) -> _ParsedOutput:
    parsed = _ParsedOutput()

    if not stdout.strip():
        return parsed

    try:
        root = json.loads(stdout)
        if not isinstance(root, dict):
            raise ValueError("not a dict")
    except (json.JSONDecodeError, ValueError):
        parsed.messages.append(stdout.strip())
        return parsed

    if msg := root.get("systemMessage"):
        parsed.messages.append(str(msg))
    if reason := root.get("reason"):
        parsed.messages.append(str(reason))

    if root.get("continue") is False or root.get("decision") == "block":
        parsed.deny = True

    if specific := root.get("hookSpecificOutput"):
        if ctx := specific.get("additionalContext"):
            parsed.messages.append(str(ctx))
        decision_str = specific.get("permissionDecision", "")
        parsed.permission_decision = {
            "allow": PermissionDecision.Allow,
            "deny":  PermissionDecision.Deny,
            "ask":   PermissionDecision.Ask,
        }.get(decision_str)
        if pr := specific.get("permissionDecisionReason"):
            parsed.permission_reason = str(pr)
        if ui := specific.get("updatedInput"):
            parsed.updated_input = ui if isinstance(ui, dict) else None

    if not parsed.messages:
        parsed.messages.append(stdout.strip())

    return parsed


def _merge(result: HookRunResult, parsed: _ParsedOutput) -> None:
    result.messages.extend(parsed.messages)
    if parsed.permission_decision is not None:
        result.permission_decision = parsed.permission_decision
    if parsed.permission_reason:
        result.permission_reason = parsed.permission_reason
    if parsed.updated_input is not None:
        result.updated_input = parsed.updated_input


# ---------------------------------------------------------------------------
# Single command runner
# ---------------------------------------------------------------------------

def _run_command(
    command: str,
    event: HookEvent,
    tool_name: str,
    payload: str,
    abort_signal: HookAbortSignal | None,
) -> tuple[str, str, int | None]:
    """Run one hook command. Returns (stdout, stderr, exit_code|None)."""
    env = dict(os.environ)
    env["HOOK_EVENT"] = event.value
    env["HOOK_TOOL_NAME"] = tool_name

    try:
        proc = subprocess.Popen(
            ["sh", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except OSError as e:
        return "", str(e), -1

    # Write payload to stdin then close
    try:
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.close()
    except OSError:
        pass

    # Poll until done or aborted
    deadline = time.monotonic() + _HOOK_TIMEOUT_S
    while True:
        if abort_signal and abort_signal.is_aborted():
            try:
                proc.kill()
                proc.wait()
            except OSError:
                pass
            return "", "aborted", None  # None = cancelled

        ret = proc.poll()
        if ret is not None:
            stdout = proc.stdout.read().decode("utf-8", errors="replace").strip()
            stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
            return stdout, stderr, ret

        if time.monotonic() > deadline:
            try:
                proc.kill()
                proc.wait()
            except OSError:
                pass
            return "", f"hook timed out after {_HOOK_TIMEOUT_S}s", -1

        time.sleep(_POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Hook runner
# ---------------------------------------------------------------------------

class HookRunner:
    """Runs configured pre/post hook commands around tool execution."""

    def __init__(
        self,
        pre_tool_use: list[str] | None = None,
        post_tool_use: list[str] | None = None,
        post_failure: list[str] | None = None,
    ) -> None:
        self._pre = pre_tool_use or []
        self._post = post_tool_use or []
        self._fail = post_failure or []

    @classmethod
    def from_env(cls) -> "HookRunner":
        """Load hook commands from environment variables."""
        def _parse(env_var: str) -> list[str]:
            raw = os.environ.get(env_var, "").strip()
            if not raw:
                return []
            return [c.strip() for c in raw.split(",") if c.strip()]

        return cls(
            pre_tool_use=_parse("BC_HOOKS_PRE_TOOL_USE"),
            post_tool_use=_parse("BC_HOOKS_POST_TOOL_USE"),
            post_failure=_parse("BC_HOOKS_POST_FAILURE"),
        )

    def _run_commands(
        self,
        event: HookEvent,
        commands: list[str],
        tool_name: str,
        tool_input: dict,
        tool_output: str | None,
        is_error: bool,
        abort_signal: HookAbortSignal | None,
        reporter: Optional[HookProgressReporter] = None,
    ) -> HookRunResult:
        if not commands:
            return HookRunResult.allow()

        if abort_signal and abort_signal.is_aborted():
            return HookRunResult(
                allowed=True,
                cancelled=True,
                messages=[f"{event.value} hook cancelled before execution"],
            )

        payload = _build_payload(event, tool_name, tool_input, tool_output, is_error)
        result = HookRunResult.allow()

        for command in commands:
            if reporter:
                reporter.on_event(HookProgressEvent.started(event, tool_name, command))

            stdout, stderr, exit_code = _run_command(
                command, event, tool_name, payload, abort_signal
            )

            if exit_code is None:
                # Cancelled
                if reporter:
                    reporter.on_event(HookProgressEvent.cancelled(event, tool_name, command))
                result.cancelled = True
                result.messages.append(
                    f"{event.value} hook '{command}' cancelled while handling '{tool_name}'"
                )
                return result

            parsed = _parse_hook_output(stdout)

            if exit_code == 0:
                if reporter:
                    reporter.on_event(HookProgressEvent.completed(event, tool_name, command))
                if parsed.deny:
                    _merge(result, parsed)
                    result.allowed = False
                    return result
                _merge(result, parsed)

            elif exit_code == 2:
                if reporter:
                    reporter.on_event(HookProgressEvent.completed(event, tool_name, command))
                # Explicit deny
                if not parsed.messages:
                    parsed.messages.append(
                        f"{event.value} hook denied tool '{tool_name}'"
                    )
                _merge(result, parsed)
                result.allowed = False
                return result

            else:
                if reporter:
                    reporter.on_event(HookProgressEvent.completed(event, tool_name, command))
                # Non-zero, non-2 — failure, stop chain
                msg = f"Hook '{command}' exited with status {exit_code}"
                if parsed.messages:
                    msg += ": " + parsed.messages[0]
                elif stderr:
                    msg += ": " + stderr
                result.messages.append(msg)
                result.failed = True
                logger.warning("hook failure: %s", msg)
                return result

        return result

    # Public API

    def pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict,
        abort_signal: HookAbortSignal | None = None,
        reporter: Optional[HookProgressReporter] = None,
    ) -> HookRunResult:
        return self._run_commands(
            HookEvent.PreToolUse, self._pre,
            tool_name, tool_input, None, False, abort_signal, reporter,
        )

    def post_tool_use(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        is_error: bool = False,
        abort_signal: HookAbortSignal | None = None,
        reporter: Optional[HookProgressReporter] = None,
    ) -> HookRunResult:
        return self._run_commands(
            HookEvent.PostToolUse, self._post,
            tool_name, tool_input, tool_output, is_error, abort_signal, reporter,
        )

    def post_failure(
        self,
        tool_name: str,
        tool_input: dict,
        error: str,
        abort_signal: HookAbortSignal | None = None,
        reporter: Optional[HookProgressReporter] = None,
    ) -> HookRunResult:
        return self._run_commands(
            HookEvent.PostToolUseFailure, self._fail,
            tool_name, tool_input, error, True, abort_signal, reporter,
        )

    def has_pre_hooks(self) -> bool:
        return bool(self._pre)

    def has_post_hooks(self) -> bool:
        return bool(self._post) or bool(self._fail)


# Module-level singleton loaded from env
hook_runner = HookRunner.from_env()
