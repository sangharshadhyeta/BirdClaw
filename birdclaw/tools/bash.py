"""Bash execution tool — Python port of claw-code-parity/rust/crates/runtime/src/bash.rs.

Key behaviours matched from Rust:
- Login shell: sh -lc <command>
- Structured output: stdout, stderr, interrupted, return_code_interpretation
- 16 KB truncation at valid UTF-8 char boundaries
- Background mode: returns session_id + PID; output buffered for polling
- Permission check before execution

Async process registry (Port 4):
    bash_run(cmd, background=True)   → {"session_id": ..., "pid": N, "status": "running"}
    bash_poll(session_id)            → {"status": ..., "stdout_tail": ..., "exit_code": ...}
    bash_write(session_id, text)     → {"ok": true}
    bash_kill(session_id)            → {"ok": true}
"""

from __future__ import annotations

import json
import os
import re as _re
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal

import logging

from birdclaw.tools.permission import enforcer
from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_BYTES = 16_384          # 16 KB — matches Rust TRUNCATED_OUTPUT_SIZE
_DEFAULT_TIMEOUT_MS = 30_000
_PROCESS_TTL_S = 1800               # 30 min — clean up finished sessions

# ---------------------------------------------------------------------------
# Secret scrubbing — strip high-entropy secrets from output before returning
# ---------------------------------------------------------------------------

# Patterns: each is (compiled_re, keep_prefix_group | None)
# If keep_prefix_group is set, group(1) is preserved and only the suffix is redacted.
_SECRET_PATTERNS: list[tuple[_re.Pattern, int | None]] = [
    # AWS key ID — standalone token
    (_re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None),
    # KEY=value pairs — keep the key name, redact the value
    (_re.compile(r"(aws_secret_access_key\s*=\s*)\S+", _re.IGNORECASE), 1),
    (_re.compile(r"(aws_session_token\s*=\s*)\S+", _re.IGNORECASE), 1),
    (_re.compile(r"((?:GITHUB|GH)_TOKEN\s*=\s*)\S+", _re.IGNORECASE), 1),
    (_re.compile(r"(OPENAI_API_KEY\s*=\s*)\S+", _re.IGNORECASE), 1),
    (_re.compile(r"(ANTHROPIC_API_KEY\s*=\s*)\S+", _re.IGNORECASE), 1),
    (_re.compile(r"((?:API|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY)\s*=\s*)\S{8,}", _re.IGNORECASE), 1),
    # Bearer tokens in HTTP headers
    (_re.compile(r"(Authorization:\s*Bearer\s+)\S+", _re.IGNORECASE), 1),
    # Generic 64-char hex strings (looks like SHA-256 key material)
    (_re.compile(r"\b[0-9a-f]{64}\b", _re.IGNORECASE), None),
]


def _scrub_secrets(text: str) -> str:
    """Replace recognisable secret patterns with [REDACTED]."""
    for pat, prefix_group in _SECRET_PATTERNS:
        if prefix_group is None:
            text = pat.sub("[REDACTED]", text)
        else:
            text = pat.sub(lambda m, g=prefix_group: m.group(g) + "[REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Rate limiting — max 20 bash calls/min per session (task_id)
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_rate_counts: dict[str, list[float]] = {}   # task_id → list of call timestamps
_RATE_LIMIT = 120   # max bash calls/min per task — runaway loop guard only (local LLM = no API cost)
_RATE_WINDOW = 60.0  # seconds


def _check_rate_limit(task_id: str) -> bool:
    """Return True if call is allowed, False if rate limit exceeded."""
    now = time.time()
    with _rate_lock:
        calls = _rate_counts.get(task_id, [])
        calls = [t for t in calls if now - t < _RATE_WINDOW]
        if len(calls) >= _RATE_LIMIT:
            _rate_counts[task_id] = calls
            return False
        calls.append(now)
        if calls:
            _rate_counts[task_id] = calls
        else:
            _rate_counts.pop(task_id, None)
        return True


# ---------------------------------------------------------------------------
# Command classification — programmatic safety analysis before execution
# ---------------------------------------------------------------------------

_READ_ONLY_CMDS = frozenset({
    "cat", "ls", "ll", "la", "echo", "pwd", "whoami", "id", "date", "uname",
    "which", "type", "file", "stat", "wc", "head", "tail", "less", "more",
    "grep", "find", "locate", "df", "du", "free", "ps", "top", "htop",
    "env", "printenv", "set", "history", "man", "help", "python", "python3",
    "pip", "pip3", "conda", "git", "diff", "md5sum", "sha256sum",
})

_WRITE_CMDS = frozenset({
    "touch", "mkdir", "cp", "mv", "ln", "chmod", "chown", "chgrp",
    "tee", "truncate", "write", "install", "make", "cmake", "pip install",
    "conda install", "apt", "apt-get", "yum", "dnf", "brew", "npm", "yarn",
})

_DESTRUCTIVE_PATTERNS = [
    _re.compile(r"\brm\s+(-[a-z]*r[a-z]*|-[a-z]*f[a-z]*|--recursive|--force)"),
    _re.compile(r"\brm\s+/"),
    _re.compile(r":\(\)\{.*\}.*:"),        # fork bomb
    _re.compile(r"\bmkfs\b"),
    _re.compile(r"\bdd\s+.*of=/dev/"),
    _re.compile(r"\bshred\b"),
    _re.compile(r"\bwipe\b"),
    _re.compile(r"chmod\s+-?R\s+000"),
]

_NETWORK_CMDS = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "ftp", "sftp", "nc", "ncat",
    "nmap", "ping", "traceroute", "dig", "nslookup", "telnet",
})

_PROCESS_CMDS = frozenset({
    "kill", "killall", "pkill", "systemctl", "service", "nohup", "screen",
    "tmux", "at", "cron", "crontab", "bg", "fg", "jobs", "disown",
})


def classify_command(cmd: str) -> str:
    """Classify a bash command's intent without executing it.

    Returns one of: "read_only" | "write" | "destructive" | "network" | "process" | "unknown"

    Used to:
    - Surface meaningful reason codes in permission prompts
    - Block destructive patterns in restricted modes
    - Log what class of action the model is taking
    """
    stripped = cmd.strip()
    first_word = stripped.split()[0].lstrip("!").lower() if stripped.split() else ""

    for pat in _DESTRUCTIVE_PATTERNS:
        if pat.search(stripped):
            return "destructive"

    if first_word in _NETWORK_CMDS:
        return "network"

    if first_word in _PROCESS_CMDS:
        return "process"

    # Check for output redirection (write to file) — e.g. "echo x > file"
    if _re.search(r"\s>+\s", stripped) or _re.search(r"\|\s*tee\b", stripped):
        return "write"

    if first_word in _WRITE_CMDS:
        return "write"

    if first_word in _READ_ONLY_CMDS:
        return "read_only"

    return "unknown"


# ---------------------------------------------------------------------------
# Output dataclass (synchronous execution)
# ---------------------------------------------------------------------------

@dataclass
class BashCommandOutput:
    stdout: str
    stderr: str
    interrupted: bool
    return_code_interpretation: str  # "exit_code:N" or "timeout"
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: int = 0
    sandbox_status: dict | None = None   # SandboxStatus.to_dict() if sandbox was evaluated

    def to_dict(self) -> dict:
        d = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "interrupted": self.interrupted,
            "return_code_interpretation": self.return_code_interpretation,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
        }
        if self.sandbox_status is not None:
            d["sandbox_status"] = self.sandbox_status
        return d


# ---------------------------------------------------------------------------
# Async process — background execution with buffered stdout/stderr
# ---------------------------------------------------------------------------

@dataclass
class BashSession:
    session_id: str
    pid: int
    command: str
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "pid": self.pid,
            "command": self.command,
            "started_at": self.started_at,
            "status": "running",
        }


class _AsyncProcess:
    """Wraps a Popen with thread-buffered stdout/stderr and a stdin pipe."""

    def __init__(self, session_id: str, proc: subprocess.Popen, command: str) -> None:
        self.session_id = session_id
        self.command = command
        self.started_at = time.time()
        self.ended_at: float = 0.0
        self.exit_code: int | None = None
        self._status: Literal["running", "completed", "failed", "killed"] = "running"
        self._lock = threading.Lock()
        self._proc = proc

        # Ring buffers — keep last 16 KB of output
        self._stdout_buf: Deque[str] = deque(maxlen=200)
        self._stderr_buf: Deque[str] = deque(maxlen=200)

        # Reader threads
        self._t_out = threading.Thread(target=self._read_stream,
                                       args=(proc.stdout, self._stdout_buf), daemon=True)
        self._t_err = threading.Thread(target=self._read_stream,
                                       args=(proc.stderr, self._stderr_buf), daemon=True)
        self._t_wait = threading.Thread(target=self._wait, daemon=True)
        self._t_out.start()
        self._t_err.start()
        self._t_wait.start()

    def _read_stream(self, stream, buf: Deque[str]) -> None:
        try:
            for line in stream:
                with self._lock:
                    buf.append(line)
        except Exception:
            pass

    def _wait(self) -> None:
        code = self._proc.wait()
        with self._lock:
            self.exit_code = code
            self.ended_at = time.time()
            self._status = "completed" if code == 0 else "failed"

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def poll_dict(self) -> dict:
        with self._lock:
            stdout_tail = _scrub_secrets(_truncate_to_char_boundary(
                "".join(self._stdout_buf).encode("utf-8", errors="replace"), _MAX_OUTPUT_BYTES
            ))
            stderr_tail = _scrub_secrets(_truncate_to_char_boundary(
                "".join(self._stderr_buf).encode("utf-8", errors="replace"), _MAX_OUTPUT_BYTES
            ))
            duration_ms = int(
                ((self.ended_at or time.time()) - self.started_at) * 1000
            )
            return {
                "session_id": self.session_id,
                "status": self._status,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "exit_code": self.exit_code,
                "duration_ms": duration_ms,
                "started_at": self.started_at,
                "ended_at": self.ended_at or None,
            }

    def write_stdin(self, text: str) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.write(text)
            self._proc.stdin.flush()

    def kill(self) -> None:
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        with self._lock:
            self._status = "killed"
            self.ended_at = time.time()

    def is_expired(self) -> bool:
        """True if finished and past the TTL window (safe to evict)."""
        with self._lock:
            if self._status == "running":
                return False
            return (time.time() - (self.ended_at or self.started_at)) > _PROCESS_TTL_S


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

class _ProcessRegistry:
    """Thread-safe registry of active async bash sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _AsyncProcess] = {}

    def register(self, proc: subprocess.Popen, command: str) -> BashSession:
        sid = str(uuid.uuid4())[:16]
        ap = _AsyncProcess(sid, proc, command)
        with self._lock:
            self._evict_expired()
            self._sessions[sid] = ap
        return BashSession(session_id=sid, pid=proc.pid, command=command,
                           started_at=ap.started_at)

    def get(self, session_id: str) -> _AsyncProcess | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _evict_expired(self) -> None:
        expired = [k for k, v in self._sessions.items() if v.is_expired()]
        for k in expired:
            del self._sessions[k]


_registry = _ProcessRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_to_char_boundary(data: bytes, limit: int) -> str:
    """Decode up to `limit` bytes, backing off to a valid UTF-8 boundary."""
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    chunk = data[:limit]
    while chunk and (chunk[-1] & 0xC0) == 0x80:
        chunk = chunk[:-1]
    return chunk.decode("utf-8", errors="replace") + "\n[truncated]"


# ---------------------------------------------------------------------------
# Core execution — synchronous
# ---------------------------------------------------------------------------

def run_bash(
    command: str,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    background: bool = False,
    stdin: str | None = None,
) -> str:
    """Execute a shell command and return a JSON-encoded BashCommandOutput.

    Args:
        command:    Shell command to run (passed to `sh -lc`).
        timeout_ms: Milliseconds before the process is killed (default 30s).
        background: If True, start process in the async registry and return session_id.
        stdin:      Optional text to feed to the process as standard input.

    Returns:
        JSON string — BashCommandOutput fields, or BashSession fields if background=True.
    """
    # Classify command intent before permission check — gives the enforcer
    # and the model a meaningful reason code instead of generic "denied".
    intent = classify_command(command)
    logger.debug("bash intent=%s cmd=%s", intent, command[:80])

    result = enforcer.check_bash(command)
    if not result.allowed:
        return json.dumps({"error": str(result), "intent": intent})

    # Rate limiting — guard against runaway loops
    from birdclaw.tools.context_vars import get_task_id
    _task_id = get_task_id() or "default"
    if not _check_rate_limit(_task_id):
        return json.dumps({"error": f"rate limit: max {_RATE_LIMIT} bash calls/min exceeded"})

    # Resolve sandbox for this command
    from birdclaw.tools.sandbox import build_sandbox_command, resolve_sandbox_status, sandbox_config_from_settings
    cwd = Path.cwd()
    sandbox_cfg = sandbox_config_from_settings()
    sandbox_st  = resolve_sandbox_status(sandbox_cfg, cwd)
    sc          = build_sandbox_command(command, cwd, sandbox_st)
    sandbox_dict = sandbox_st.to_dict()

    if sc:
        cmd = [sc.program] + sc.args
        proc_env = {**os.environ, **dict(sc.env)}
    else:
        if sys.platform == "win32":
            cmd = ["cmd", "/c", command]
        else:
            cmd = ["sh", "-lc", command]
        proc_env = None  # inherit from parent

    _popen_kwargs: dict = dict(
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
    )
    if sys.platform == "win32":
        _popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        _popen_kwargs["start_new_session"] = True  # own process group for clean kill

    if background:
        proc = subprocess.Popen(cmd, **_popen_kwargs)
        session = _registry.register(proc, command)
        return json.dumps({**session.to_dict(), "sandbox_status": sandbox_dict})

    stdin_bytes = stdin.encode("utf-8") if stdin is not None else None
    timeout_s = timeout_ms / 1000.0
    interrupted = False
    started_at = time.time()

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=timeout_s,
            env=proc_env,
        )
        stdout_raw = proc.stdout
        stderr_raw = proc.stderr
        return_code_interp = f"exit_code:{proc.returncode}"
    except subprocess.TimeoutExpired as e:
        stdout_raw = e.stdout or b""
        stderr_raw = e.stderr or b""
        return_code_interp = "timeout"
        interrupted = True

    ended_at = time.time()
    output = BashCommandOutput(
        stdout=_scrub_secrets(_truncate_to_char_boundary(stdout_raw, _MAX_OUTPUT_BYTES)),
        stderr=_scrub_secrets(_truncate_to_char_boundary(stderr_raw, _MAX_OUTPUT_BYTES)),
        interrupted=interrupted,
        return_code_interpretation=return_code_interp,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=int((ended_at - started_at) * 1000),
        sandbox_status=sandbox_dict,
    )
    return json.dumps(output.to_dict())


# ---------------------------------------------------------------------------
# Async process tools
# ---------------------------------------------------------------------------

def bash_poll(session_id: str) -> str:
    """Poll the status and output tail of a background bash session.

    Returns JSON with: status, stdout_tail, stderr_tail, exit_code, duration_ms.
    """
    ap = _registry.get(session_id)
    if ap is None:
        return json.dumps({"error": f"session not found: {session_id}"})
    return json.dumps(ap.poll_dict())


def bash_write(session_id: str, text: str) -> str:
    """Write text to the stdin of a running background bash session.

    Useful for answering interactive prompts in long-running processes.
    """
    ap = _registry.get(session_id)
    if ap is None:
        return json.dumps({"error": f"session not found: {session_id}"})
    if ap.status != "running":
        return json.dumps({"error": f"session {session_id} is not running (status={ap.status})"})
    ap.write_stdin(text)
    return json.dumps({"ok": True, "session_id": session_id})


def bash_kill(session_id: str) -> str:
    """Send SIGTERM to a background bash session's process group."""
    ap = _registry.get(session_id)
    if ap is None:
        return json.dumps({"error": f"session not found: {session_id}"})
    ap.kill()
    return json.dumps({"ok": True, "session_id": session_id, "status": "killed"})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_BASH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Timeout in milliseconds (default 30000). Ignored when background=true.",
        },
        "background": {
            "type": "boolean",
            "description": (
                "If true, run in background and return a session_id. "
                "Use bash_poll to check output, bash_write to send stdin, bash_kill to stop."
            ),
        },
        "stdin": {
            "type": "string",
            "description": "Text to send to the process as stdin (synchronous mode only).",
        },
    },
    "required": ["command"],
}

_POLL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "session_id": {
            "type": "string",
            "description": "Session ID returned by bash with background=true.",
        },
    },
    "required": ["session_id"],
}

_WRITE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "Background session ID."},
        "text": {"type": "string", "description": "Text to write to stdin."},
    },
    "required": ["session_id", "text"],
}

_KILL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "Background session ID to terminate."},
    },
    "required": ["session_id"],
}

registry.register(Tool(
    name="bash",
    description=(
        "Run a shell command. Returns structured output with stdout, stderr, "
        "and return code. Use background=true for long-running processes; "
        "then use bash_poll / bash_write / bash_kill to manage them."
    ),
    input_schema=_BASH_SCHEMA,
    handler=run_bash,
    tags=[
        "run", "exec", "shell", "bash", "sh",
        "build", "test", "install", "compile",
        "script", "command", "terminal",
        "git", "pip", "npm", "cargo", "make",
        "python", "node",
    ],
))

registry.register(Tool(
    name="bash_poll",
    description=(
        "Poll the status and buffered output of a background bash session. "
        "Returns status (running/completed/failed/killed), stdout_tail, stderr_tail, exit_code."
    ),
    input_schema=_POLL_SCHEMA,
    handler=bash_poll,
    tags=["background", "poll", "status", "async", "process"],
))

registry.register(Tool(
    name="bash_write",
    description="Write text to the stdin of a running background bash session (e.g. answer a prompt).",
    input_schema=_WRITE_SCHEMA,
    handler=bash_write,
    tags=["background", "stdin", "interactive", "async", "process"],
))

registry.register(Tool(
    name="bash_kill",
    description="Send SIGTERM to a background bash session's process group.",
    input_schema=_KILL_SCHEMA,
    handler=bash_kill,
    tags=["background", "kill", "stop", "async", "process"],
))
