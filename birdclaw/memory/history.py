"""Conversation history — long-term user ↔ BirdClaw turn log.

This is the soul agent's memory of what was said. Separate from session_log.py
(which records tool calls and stage summaries inside a task run).

What goes here:
  - user turns: what the user asked BirdClaw
  - assistant turns: what BirdClaw answered the user
  - compaction events: when old turns were summarised to save space

What does NOT go here:
  - tool calls, bash output, stage transitions → session_log.py
  - knowledge entities, research facts → graph.py
  - task status, agent lifecycle → tasks.py

File format (JSONL, one record per line):
    {"type": "session_meta", "session_id": "...", "version": 1,
     "created_at": 1234.5, "updated_at": 1234.5, "fork": {...} | null}
    {"type": "turn", "role": "user", "content": "...", "ts": 1234.5}
    {"type": "turn", "role": "assistant", "content": "...", "ts": 1234.5}
    {"type": "compaction", "summary": "...", "removed": 12, "ts": 1234.5}

Storage:
    ~/.birdclaw/history/<session_id>.jsonl        active file
    ~/.birdclaw/history/<session_id>.1.jsonl      first rotated copy
    ~/.birdclaw/history/<session_id>.2.jsonl      ...
    ~/.birdclaw/history/<session_id>.3.jsonl      oldest kept copy (max 3)

Rotation: triggered when the active file exceeds 256 KB. The active file
becomes .1.jsonl, old rotated files are shifted (.1→.2, .2→.3), and the
oldest (.4) is deleted. A fresh file is started with a new session_meta header.

Reference: claw-code-parity/rust/crates/runtime/src/session.rs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from birdclaw.config import settings

logger = logging.getLogger(__name__)

VERSION = 1
ROTATE_AFTER_BYTES = 256 * 1024   # 256 KB — matches Rust ROTATE_AFTER_BYTES
MAX_ROTATED_FILES = 3              # keep .1 .2 .3 — matches Rust MAX_ROTATED_FILES

TurnRole = Literal["user", "assistant", "system"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: TurnRole
    content: str
    ts: float = field(default_factory=time.time)

    def to_record(self) -> dict:
        return {"type": "turn", "role": self.role, "content": self.content, "ts": self.ts}

    @classmethod
    def from_record(cls, d: dict) -> "Turn":
        return cls(role=d["role"], content=d["content"], ts=d.get("ts", 0.0))


@dataclass
class Compaction:
    summary: str
    removed: int
    ts: float = field(default_factory=time.time)

    def to_record(self) -> dict:
        return {"type": "compaction", "summary": self.summary, "removed": self.removed, "ts": self.ts}

    @classmethod
    def from_record(cls, d: dict) -> "Compaction":
        return cls(summary=d.get("summary", ""), removed=d.get("removed", 0), ts=d.get("ts", 0.0))


@dataclass
class Fork:
    parent_session_id: str
    branch_name: str = ""

    def to_dict(self) -> dict:
        return {"parent_session_id": self.parent_session_id, "branch_name": self.branch_name}

    @classmethod
    def from_dict(cls, d: dict) -> "Fork":
        return cls(
            parent_session_id=d.get("parent_session_id", ""),
            branch_name=d.get("branch_name", ""),
        )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class History:
    """Append-only JSONL conversation history for one session.

    Usage:
        h = History.new()               # start a new session
        h = History.load(session_id)    # resume an existing session

        h.add_turn("user", "what time is it?")
        h.add_turn("assistant", "It's 3pm.")

        turns = h.recent(10)            # last 10 turns for soul agent context
        child = h.fork("experiment")    # branch from current point

        h.record_compaction("discussed X and Y", removed=8)
    """

    def __init__(
        self,
        session_id: str,
        path: Path,
        *,
        created_at: float | None = None,
        parent_fork: Fork | None = None,
    ) -> None:
        self.session_id = session_id
        self._path = path
        self.created_at: float = created_at or time.time()
        self.updated_at: float = self.created_at
        self.parent_fork = parent_fork
        self._turns: list[Turn] = []
        self._compactions: list[Compaction] = []

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def new(cls, session_id: str | None = None) -> "History":
        """Start a fresh conversation history.

        Args:
            session_id: Use a specific ID (e.g. gateway session IDs like "tui:local").
                        If omitted, a new UUID-based ID is generated.
        """
        _ensure_history_dir()
        sid = session_id or _new_session_id()
        path = _history_dir() / f"{sid}.jsonl"
        h = cls(session_id=sid, path=path)
        h._write_meta()
        logger.debug("new history session %s", sid)
        return h

    @classmethod
    def load(cls, session_id: str) -> "History | None":
        """Load an existing history session from disk. Returns None if not found."""
        path = _history_dir() / f"{session_id}.jsonl"
        if not path.exists():
            return None
        return cls._parse(path)

    @classmethod
    def load_latest(cls) -> "History | None":
        """Load the most recently modified history session, or None if none exist."""
        paths = sorted(_history_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Skip rotated files (.1.jsonl, .2.jsonl, .3.jsonl)
        active = [p for p in paths if p.stem.count(".") == 0]
        if not active:
            return None
        return cls._parse(active[0])

    # ── Core write operations ─────────────────────────────────────────────────

    def add_turn(self, role: TurnRole, content: str) -> Turn:
        """Append a user or assistant turn. Returns the recorded Turn."""
        turn = Turn(role=role, content=content)
        self._turns.append(turn)
        self.updated_at = turn.ts
        self._append_record(turn.to_record())
        return turn

    def record_compaction(self, summary: str, removed: int) -> None:
        """Record that old turns were summarised and removed from active context."""
        c = Compaction(summary=summary, removed=removed)
        self._compactions.append(c)
        self.updated_at = c.ts
        self._append_record(c.to_record())

    # ── Fork ──────────────────────────────────────────────────────────────────

    def fork(self, branch_name: str = "") -> "History":
        """Create a child session that starts with all current turns.

        The child records its parent so we can trace lineage. The parent
        session is unchanged — both are independent from this point on.
        """
        _ensure_history_dir()
        child_id = _new_session_id()
        child_path = _history_dir() / f"{child_id}.jsonl"
        child = History(
            session_id=child_id,
            path=child_path,
            parent_fork=Fork(parent_session_id=self.session_id, branch_name=branch_name),
        )
        # Write header with fork reference
        child._write_meta()
        # Copy all current turns into the child
        for turn in self._turns:
            child._turns.append(turn)
            child._append_record(turn.to_record())
        logger.debug(
            "forked %s → %s (branch=%r, %d turns copied)",
            self.session_id, child_id, branch_name, len(self._turns),
        )
        return child

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[Turn]:
        """Return the last n turns (most recent last)."""
        return self._turns[-n:]

    def recent_text(self, n: int = 20) -> str:
        """Compact text representation of the last n turns — for soul agent context."""
        turns = self.recent(n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            prefix = "User" if t.role == "user" else "BirdClaw"
            lines.append(f"{prefix}: {t.content[:300]}")
        return "\n".join(lines)

    def summary_text(self, n: int = 3) -> str:
        """Short summarised form of the last n turns for soul prompt injection.

        More compact than recent_text() — 150 chars per turn, no padding.
        """
        turns = self.recent(n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            prefix = "User" if t.role == "user" else "BirdClaw"
            body = t.content[:150].replace("\n", " ").strip()
            lines.append(f"{prefix}: {body}")
        return "\n".join(lines)

    def search(self, query: str, n: int = 5) -> list[Turn]:
        """Simple substring search through all turns. Returns up to n matches."""
        q = query.lower()
        matches = [t for t in self._turns if q in t.content.lower()]
        return matches[-n:]

    def all_turns(self) -> list[Turn]:
        return list(self._turns)

    def turn_count(self) -> int:
        return len(self._turns)

    def last_user_turn(self) -> Turn | None:
        for t in reversed(self._turns):
            if t.role == "user":
                return t
        return None

    def last_assistant_turn(self) -> Turn | None:
        for t in reversed(self._turns):
            if t.role == "assistant":
                return t
        return None

    def clear(self) -> None:
        """Erase all turns from this session (keeps the file, resets to meta-only)."""
        self._turns.clear()
        self._compactions.clear()
        self.updated_at = time.time()
        self._write_meta()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _write_meta(self) -> None:
        """Write (or overwrite) the session_meta header line."""
        record = {
            "type": "session_meta",
            "version": VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.parent_fork:
            record["fork"] = self.parent_fork.to_dict()
        try:
            with self._path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("could not write history meta %s: %s", self._path, e)

    def _append_record(self, record: dict) -> None:
        """Append one JSONL record, rotating the file if it exceeds ROTATE_AFTER_BYTES."""
        try:
            # Rotate before appending if file is too large
            if self._path.exists() and self._path.stat().st_size >= ROTATE_AFTER_BYTES:
                _rotate(self._path, MAX_ROTATED_FILES)
                # After rotation the active file is gone — write a fresh header
                self._write_meta()

            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("could not append history record: %s", e)

    @classmethod
    def _parse(cls, path: Path) -> "History":
        """Parse a JSONL history file into a History object."""
        turns: list[Turn] = []
        compactions: list[Compaction] = []
        session_id = path.stem.split(".")[0]
        created_at: float = 0.0
        updated_at: float = 0.0
        fork: Fork | None = None

        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed line %d in %s", i, path)
                continue

            rtype = record.get("type", "")
            if rtype == "session_meta":
                session_id = record.get("session_id", session_id)
                created_at = record.get("created_at", 0.0)
                updated_at = record.get("updated_at", created_at)
                if fd := record.get("fork"):
                    fork = Fork.from_dict(fd)
            elif rtype == "turn":
                turns.append(Turn.from_record(record))
            elif rtype == "compaction":
                compactions.append(Compaction.from_record(record))
            else:
                logger.debug("unknown history record type %r at line %d — skipping", rtype, i)

        h = cls(session_id=session_id, path=path, created_at=created_at, parent_fork=fork)
        h.updated_at = updated_at or (turns[-1].ts if turns else created_at)
        h._turns = turns
        h._compactions = compactions
        return h


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotate(path: Path, max_rotated: int) -> None:
    """Rotate active file: active→.1, .1→.2, ..., oldest deleted.

    Example for max_rotated=3:
        session.3.jsonl  → deleted
        session.2.jsonl  → session.3.jsonl
        session.1.jsonl  → session.2.jsonl
        session.jsonl    → session.1.jsonl  (active file)
    """
    stem = path.stem          # e.g. "abc123"
    suffix = path.suffix      # ".jsonl"
    parent = path.parent

    # Delete the oldest rotated file if it exists
    oldest = parent / f"{stem}.{max_rotated}{suffix}"
    if oldest.exists():
        oldest.unlink()

    # Shift existing rotated files up
    for n in range(max_rotated - 1, 0, -1):
        src = parent / f"{stem}.{n}{suffix}"
        dst = parent / f"{stem}.{n + 1}{suffix}"
        if src.exists():
            src.rename(dst)

    # Move active file to .1
    if path.exists():
        path.rename(parent / f"{stem}.1{suffix}")

    logger.debug("rotated %s (max_rotated=%d)", path.name, max_rotated)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _history_dir() -> Path:
    d = settings.data_dir / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_history_dir() -> None:
    _history_dir()


def _new_session_id() -> str:
    return str(uuid.uuid4())[:16]
