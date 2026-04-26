"""Unix socket bridge for long-running tests → live TUI.

When the TUI is running, tests can route prompts through it so execution
is visible in all three panels.  Tests that detect the socket path exists
send a JSON request; the server submits to the in-process gateway, waits
for the task to reach a terminal state, then returns metrics.

Socket path:  ~/.birdclaw/test.sock
Protocol:     one JSON object per connection (newline-terminated)

Request  {"id": str, "prompt": str, "workspace": str | null}
Response {"id": str, "task_id": str, "planned": int, "completed": int,
          "steps": int, "answer": str, "duration_s": float, "error": str | null}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SOCK_PATH    = Path.home() / ".birdclaw" / "test.sock"
_POLL_INTERVAL = 0.5   # seconds between task-status polls
_TIMEOUT       = 1800  # maximum seconds to wait for a task


def socket_path() -> Path:
    return _SOCK_PATH


def _handle(conn: socket.socket) -> None:
    """Handle one test connection: submit prompt, wait, return metrics."""
    try:
        raw = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            raw += chunk
            if b"\n" in raw:
                break

        try:
            req = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            _send(conn, {"error": f"bad JSON: {e}"})
            return

        req_id    = req.get("id", "")
        prompt    = req.get("prompt", "")
        workspace = req.get("workspace", "")

        if not prompt:
            _send(conn, {"id": req_id, "error": "empty prompt"})
            return

        # Add workspace to workspace_roots so the agent can write there.
        from birdclaw.config import settings as _settings
        orig_roots = list(_settings.workspace_roots)
        if workspace:
            ws_path = Path(workspace)
            if ws_path not in _settings.workspace_roots:
                _settings.workspace_roots = [ws_path] + orig_roots

        t0 = time.time()
        task_id = _submit_and_wait_for_task_id(prompt, session_key=req_id, launch_cwd=workspace)

        if task_id is None:
            # Soul answered directly instead of creating a task — force-create one
            # so tests always get full agent execution with measurable phases/steps.
            try:
                from birdclaw.agent.soul_loop import _force_create_task
                soul_resp = _force_create_task(prompt, session_id=req_id)
                task_id = soul_resp.task_id or None
            except Exception as _e:
                logger.warning("test_socket: force-create fallback failed: %s", _e)
            if not task_id:
                _settings.workspace_roots = orig_roots
                _send(conn, {"id": req_id, "error": "soul did not create a task"})
                return

        # Poll until the task reaches a terminal state (restore roots after)
        try:
            task = _poll_until_done(task_id)
        finally:
            _settings.workspace_roots = orig_roots
        duration = time.time() - t0

        planned   = len(task.phases) if task else 0
        completed = task.completed_phase_count if task else 0
        answer    = (task.output or "") if task else ""
        error     = None if (task and task.status == "completed") else (
            f"task {task.status}" if task else "task lost"
        )

        _send(conn, {
            "id":         req_id,
            "task_id":    task_id,
            "planned":    planned,
            "completed":  completed,
            "steps":      _count_steps(task),
            "answer":     answer,
            "duration_s": round(duration, 1),
            "error":      error,
        })

    except Exception as e:
        logger.exception("test_socket: handler error")
        try:
            _send(conn, {"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode())


def _submit_and_wait_for_task_id(prompt: str, session_key: str = "test", launch_cwd: str = "") -> str | None:
    """Submit prompt through the soul and wait for a task to be created.

    Returns the task_id when the soul routes to create_task, or None if the
    soul answered directly (run_command / answer) without spawning a task.
    """
    from birdclaw.gateway.gateway import gateway
    from birdclaw.gateway.channel import Channel, IncomingMessage, OutgoingMessage
    from birdclaw.memory.tasks import task_registry

    existing_ids: set[str] = {t.task_id for t in task_registry.list()}
    result_box: list[str | None] = [None]
    event = threading.Event()

    class _CaptureChannel(Channel):
        @property
        def channel_id(self) -> str:
            return "test_socket"

        def deliver(self, msg: OutgoingMessage) -> None:
            if msg.task_id and msg.task_id not in existing_ids and result_box[0] is None:
                result_box[0] = msg.task_id
                event.set()
            elif msg.msg_type in ("reply", "command_result") and result_box[0] is None:
                # Soul answered directly — no task spawned
                result_box[0] = ""
                event.set()

    ch = _CaptureChannel()
    gateway.register(ch)

    session = gateway._session_mgr.get_or_create("test_socket", session_key)
    if launch_cwd:
        session.launch_cwd = launch_cwd

    gateway.submit(IncomingMessage(
        channel_id="test_socket",
        user_id=session_key,
        content=prompt,
    ))

    event.wait(timeout=30)
    task_id = result_box[0]
    return task_id if task_id else None


def _poll_until_done(task_id: str):
    """Block until task is in a terminal state; return the Task object."""
    from birdclaw.memory.tasks import task_registry, _TERMINAL

    deadline = time.time() + _TIMEOUT
    while time.time() < deadline:
        task = task_registry.get(task_id)
        if task and task.status in _TERMINAL:
            return task
        time.sleep(_POLL_INTERVAL)

    # Timeout — return whatever we have
    return task_registry.get(task_id)


def _count_steps(task) -> int:
    if task is None:
        return 0
    # Approximate: count tool-call messages in the task's message list
    return sum(1 for m in task.messages if m.role == "tool")


def start(daemon: bool = True) -> threading.Thread:
    """Start the Unix socket server in a background thread.

    Removes any stale socket file from a previous crash before binding.
    Safe to call multiple times — subsequent calls are no-ops if already running.
    """
    _SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale socket
    try:
        _SOCK_PATH.unlink()
    except FileNotFoundError:
        pass

    def _serve() -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(_SOCK_PATH))
            srv.listen(8)
            logger.info("test_socket: listening on %s", _SOCK_PATH)
            while True:
                try:
                    conn, _ = srv.accept()
                    t = threading.Thread(
                        target=_handle, args=(conn,), daemon=True, name="test-sock-handler"
                    )
                    t.start()
                except Exception as e:
                    logger.warning("test_socket: accept error: %s", e)
        finally:
            srv.close()
            try:
                _SOCK_PATH.unlink()
            except FileNotFoundError:
                pass

    t = threading.Thread(target=_serve, daemon=daemon, name="test-socket-server")
    t.start()
    return t
