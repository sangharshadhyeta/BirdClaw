"""Session manager — maps (channel_id, user_id) to persistent conversation state.

Each session owns:
    history     — conversation turns (History object, persisted to JSONL)
    task_ids    — tasks spawned in this session (for soul context scoping)

Session IDs are deterministic: f"{channel_id}:{user_id}"
e.g. "tui:local", "http:user_abc123"

Thread safety: all mutations go through a single lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id:  str
    channel_id:  str
    user_id:     str
    history:     "History"
    task_ids:    list[str] = field(default_factory=list)
    created_at:  float     = field(default_factory=time.time)
    last_active: float     = field(default_factory=time.time)
    launch_cwd:  str       = ""   # cwd of the client that opened this session

    def touch(self) -> None:
        self.last_active = time.time()

    def add_task(self, task_id: str) -> None:
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Thread-safe registry of active sessions."""

    def __init__(self) -> None:
        self._lock:     threading.Lock         = threading.Lock()
        self._sessions: dict[str, Session]     = {}

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get_or_create(self, channel_id: str, user_id: str) -> Session:
        """Return the existing session or create a new one."""
        session_id = _make_session_id(channel_id, user_id)
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

        # Create outside lock (History.load is IO)
        from birdclaw.memory.history import History
        history = History.load(session_id) or History.new(session_id=session_id)
        session = Session(
            session_id=session_id,
            channel_id=channel_id,
            user_id=user_id,
            history=history,
        )
        with self._lock:
            # Double-check in case another thread raced us
            if session_id not in self._sessions:
                self._sessions[session_id] = session
                logger.info("session: created %s", session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def touch(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.touch()

    def add_task(self, session_id: str, task_id: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.add_task(task_id)

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a conversation turn to this session's history."""
        with self._lock:
            s = self._sessions.get(session_id)
        if s:
            try:
                s.history.add_turn(role, content)
            except Exception as e:
                logger.warning("session: failed to save turn for %s: %s", session_id, e)

    # ── Queries ───────────────────────────────────────────────────────────────

    def all_sessions(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_id(channel_id: str, user_id: str) -> str:
    return f"{channel_id}:{user_id}"
