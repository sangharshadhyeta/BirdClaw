"""Daemon-side TUI socket server.

Each TUI process connects to ~/.birdclaw/tui.sock.  The daemon assigns a
unique session ID (tui:<hex8>), registers a TUISocketChannel, and handles
three disconnect cases:

  Clean quit  — TUI sends {"type": "quit"}  → tasks killed immediately
  Drop        — socket closes without quit  → 60 s grace window; tasks
                killed only if TUI does not reconnect in time
  Reconnect   — TUI sends {"type": "hello", "session_id": "tui:..."}
                → pending kill timer cancelled, same session resumes

For channels with owns_tasks=False (future Telegram etc.) no teardown
is ever triggered — tasks keep running regardless of disconnect.

Protocol: newline-delimited JSON over AF_UNIX SOCK_STREAM.

  TUI → daemon:
    {"type": "hello"}                              — new session
    {"type": "hello", "session_id": "tui:abc123"} — resume session
    {"type": "msg",   "content": "..."}            — user message
    {"type": "quit"}                               — clean exit

  Daemon → TUI (first message):
    {"type": "session", "session_id": "tui:abc123"} — assigned / confirmed

  Daemon → TUI (ongoing):
    OutgoingMessage dict (session_id, content, msg_type, task_id, metadata)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_SOCK_PATH    = Path.home() / ".birdclaw" / "tui.sock"
_GRACE_SECS   = 60   # seconds before killing tasks after an unclean disconnect


def socket_path() -> Path:
    return _SOCK_PATH


# ---------------------------------------------------------------------------
# Pending-kill registry  (session_id → threading.Timer)
# ---------------------------------------------------------------------------

_pending_kills: dict[str, threading.Timer] = {}
_pending_kills_lock = threading.Lock()


def _schedule_kill(session_id: str) -> None:
    """Start a grace-period timer.  Replaces any existing timer for the session."""
    with _pending_kills_lock:
        existing = _pending_kills.pop(session_id, None)
        if existing:
            existing.cancel()
        t = threading.Timer(_GRACE_SECS, _kill_session_tasks, args=(session_id,))
        t.daemon = True
        _pending_kills[session_id] = t
        t.start()
    logger.info("tui_socket: drop — %s grace timer started (%ds)", session_id, _GRACE_SECS)


def _cancel_kill(session_id: str) -> None:
    """Cancel a pending kill timer (called on reconnect)."""
    with _pending_kills_lock:
        t = _pending_kills.pop(session_id, None)
    if t:
        t.cancel()
        logger.info("tui_socket: kill timer cancelled for %s (reconnected)", session_id)


def _kill_session_tasks(session_id: str) -> None:
    """Interrupt every agent running a task owned by this session."""
    with _pending_kills_lock:
        _pending_kills.pop(session_id, None)
    try:
        from birdclaw.agent.orchestrator import orchestrator
        from birdclaw.gateway.gateway import gateway

        session = gateway._session_mgr.get(session_id)
        if session is None:
            return
        for task_id in list(session.task_ids):
            if orchestrator.interrupt_by_task(task_id):
                logger.info("tui_socket: interrupted task %s (session %s)", task_id[:8], session_id)
    except Exception as e:
        logger.warning("tui_socket: error killing tasks for %s: %s", session_id, e)


# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

def _send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode())


def _handle(conn: socket.socket) -> None:
    """Handle one TUI connection for its full lifetime."""
    from birdclaw.channels.tui_socket_channel import TUISocketChannel
    from birdclaw.gateway.channel import IncomingMessage
    from birdclaw.gateway.gateway import gateway

    # ── Handshake ─────────────────────────────────────────────────────────────
    # Read the hello line (with a short timeout so a badly-behaved client
    # does not stall the handler thread forever).
    conn.settimeout(5.0)
    hello_line = b""
    try:
        while b"\n" not in hello_line:
            chunk = conn.recv(256)
            if not chunk:
                conn.close()
                return
            hello_line += chunk
    except (OSError, TimeoutError):
        # No hello — treat as a new session anyway
        pass
    finally:
        conn.settimeout(None)

    session_id: str | None = None
    hello_obj: dict = {}
    try:
        hello_obj = json.loads(hello_line.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    requested_sid = hello_obj.get("session_id", "")
    launch_cwd    = hello_obj.get("cwd", "")
    if requested_sid and gateway._session_mgr.get(requested_sid) is not None:
        # Resume an existing session
        session_id = requested_sid
        _cancel_kill(session_id)
        logger.info("tui_socket: resumed session %s  cwd=%s", session_id, launch_cwd or "(none)")
    else:
        session_id = f"tui:{secrets.token_hex(4)}"
        logger.info("tui_socket: new session %s  cwd=%s", session_id, launch_cwd or "(none)")

    # Store the client's launch cwd on the session so the agent loop writes there
    if launch_cwd:
        sess = gateway._session_mgr.get_or_create(session_id, "local")
        sess.launch_cwd = launch_cwd

    # Tell the TUI which session_id it owns
    try:
        _send(conn, {"type": "session", "session_id": session_id})
    except OSError as e:
        logger.warning("tui_socket: failed to send session assignment: %s", e)
        conn.close()
        return

    # Register (or re-register) the channel so gateway routes to this socket
    channel = TUISocketChannel(session_id, conn)
    gateway.register(channel)

    # ── Message loop ──────────────────────────────────────────────────────────
    clean_quit = False
    buf = b""
    try:
        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("tui_socket: bad JSON from %s", session_id)
                    continue

                msg_type = obj.get("type", "msg")

                if msg_type == "quit":
                    clean_quit = True
                    break
                if msg_type == "ping":
                    continue
                if msg_type != "msg":
                    continue

                content = obj.get("content", "").strip()
                if not content:
                    continue

                gateway.submit(IncomingMessage(
                    channel_id=session_id,
                    user_id="local",
                    content=content,
                ))
            if clean_quit:
                break
    finally:
        try:
            conn.close()
        except OSError:
            pass

    if channel.owns_tasks:
        if clean_quit:
            logger.info("tui_socket: clean quit — killing tasks for %s immediately", session_id)
            _kill_session_tasks(session_id)
        else:
            logger.info("tui_socket: drop — starting grace timer for %s", session_id)
            _schedule_kill(session_id)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def start(daemon: bool = True) -> threading.Thread:
    """Start the TUI socket server in a background thread."""
    _SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        _SOCK_PATH.unlink()
    except FileNotFoundError:
        pass

    def _serve() -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(_SOCK_PATH))
            srv.listen(8)
            try:
                os.chmod(str(_SOCK_PATH), 0o600)
            except OSError:
                pass
            logger.info("tui_socket: listening on %s", _SOCK_PATH)
            while True:
                try:
                    conn, _ = srv.accept()
                    t = threading.Thread(
                        target=_handle,
                        args=(conn,),
                        daemon=True,
                        name="tui-sock-handler",
                    )
                    t.start()
                except Exception as e:
                    logger.warning("tui_socket: accept error: %s", e)
        finally:
            srv.close()
            try:
                _SOCK_PATH.unlink()
            except FileNotFoundError:
                pass

    t = threading.Thread(target=_serve, daemon=daemon, name="tui-socket-server")
    t.start()
    return t
