"""Session log — append-only JSONL event stream for a single session.

Two layers of memory:
  task_list.py  — step decomposition and status tracking (already built)
  session_log   — full event history for context injection and later dreaming

Event types:
    user_message      — raw message from the user
    assistant_message — final answer delivered to the user
    tool_call         — tool invoked by the agent (name + args)
    tool_result       — result returned by a tool
    task_created      — TaskList decomposed from a request
    step_done         — a TaskStep completed (with result)
    memory_hit        — graph nodes retrieved for a query
    session_summary   — generated 1-2 sentence summary of the full session so far

Storage:
    ~/.birdclaw/sessions/<session_id>.jsonl   one event per line

Injection (planning stage):
    - session_summary (latest) — full session condensed to 1-2 sentences
    - last 2 raw user_message events — recency and specificity
    - pending task steps (from task_list)
    - top-3 graph nodes (from retrieval)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from birdclaw.config import settings

logger = logging.getLogger(__name__)

EventType = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "task_created",
    "step_done",
    "memory_hit",
    "session_summary",
    "usage",
    "compaction",
    "plan",
    "stage_start",
    "stage_done",
    "subtask_manifest",
    "task_spawned",
]


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"type": self.type, "ts": self.ts, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(type=d["type"], data=d.get("data", {}), ts=d.get("ts", ""))


# ---------------------------------------------------------------------------
# SessionLog
# ---------------------------------------------------------------------------

class SessionLog:
    """Append-only JSONL log for one session.

    Usage:
        log = SessionLog.new()          # start fresh
        log = SessionLog.load(sid)      # resume existing

        log.user_message("do the thing")
        log.tool_call("bash", {"command": "ls"})
        log.tool_result("bash", '{"stdout": "..."}')
        log.step_done("step-id", "step result")

        ctx = log.planning_context()    # inject into agent system prompt
    """

    def __init__(self, session_id: str, path: Path) -> None:
        self.session_id = session_id
        self._path = path
        self._events: list[Event] = []

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def new(cls, session_id: str | None = None) -> "SessionLog":
        """Create a new session log.

        Args:
            session_id: Use this ID (e.g. the task_id) so the file can be
                        found by the TUI.  Defaults to a random UUID.
        """
        settings.ensure_dirs()
        sid = session_id if session_id else str(uuid.uuid4())[:12]
        path = settings.sessions_dir / f"{sid}.jsonl"
        log = cls(session_id=sid, path=path)
        logger.debug("new session %s", sid)
        return log

    @classmethod
    def load(cls, session_id: str) -> "SessionLog":
        """Load an existing session log from disk."""
        path = settings.sessions_dir / f"{session_id}.jsonl"
        log = cls(session_id=session_id, path=path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        log._events.append(Event.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("skipping malformed event line in %s", path)
        return log

    # ── Rotation constants ────────────────────────────────────────────────────

    _ROTATE_BYTES: int = 256 * 1024   # rotate at 256 KB
    _KEEP_ROTATED: int = 3            # keep .1 .2 .3 archives

    # ── Append helpers ────────────────────────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        """Rotate the log file when it exceeds _ROTATE_BYTES.

        Shifts existing archives: .2 → .3, .1 → .2, current → .1
        Archives beyond _KEEP_ROTATED are deleted.
        """
        try:
            if not self._path.exists() or self._path.stat().st_size < self._ROTATE_BYTES:
                return
        except OSError:
            return

        # Shift archives down: .N → .(N+1), delete if > _KEEP_ROTATED
        for i in range(self._KEEP_ROTATED, 0, -1):
            src = self._path.with_suffix(f".{i}.jsonl") if i > 1 else self._path.parent / f"{self._path.stem}.1.jsonl"
            dst = self._path.parent / f"{self._path.stem}.{i + 1}.jsonl"
            # Use explicit naming to avoid confusion
            src = self._path.parent / f"{self._path.stem}.{i}.jsonl"
            dst = self._path.parent / f"{self._path.stem}.{i + 1}.jsonl"
            if src.exists():
                if i >= self._KEEP_ROTATED:
                    src.unlink(missing_ok=True)
                else:
                    src.rename(dst)

        # Rotate current file to .1
        archive = self._path.parent / f"{self._path.stem}.1.jsonl"
        try:
            self._path.rename(archive)
            logger.info("session log rotated: %s → %s", self._path.name, archive.name)
        except OSError as e:
            logger.warning("session log rotation failed: %s", e)

    def _append(self, event: Event) -> None:
        self._events.append(event)
        try:
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("could not write session log: %s", e)

    def user_message(self, content: str) -> None:
        self._append(Event(type="user_message", data={"content": content}))

    def assistant_message(self, content: str) -> None:
        self._append(Event(type="assistant_message", data={"content": content}))

    def tool_call(self, name: str, arguments: dict) -> None:
        self._append(Event(type="tool_call", data={
            "name": name,
            "arguments": arguments,
            "called_at": time.time(),
        }))

    def tool_result(self, name: str, result: str, duration_ms: int = 0) -> None:
        # Truncate long results — we only need the gist for history
        truncated = result[:500] + ("…" if len(result) > 500 else "")
        self._append(Event(type="tool_result", data={
            "name": name,
            "result": truncated,
            "duration_ms": duration_ms,
        }))

    def plan(self, outcome: str, steps: list[str]) -> None:
        self._append(Event(type="plan", data={
            "outcome": outcome,
            "steps": steps,
            "planned_at": time.time(),
        }))
        # Forward phase list to task registry so TUI can show the plan tree.
        # session_id == task_id when launched via orchestrator.
        try:
            from birdclaw.memory.tasks import task_registry
            task_registry.set_phases(self.session_id, steps)
        except Exception:
            pass

    def stage_start(self, stage_type: str, goal: str) -> None:
        self._append(Event(type="stage_start", data={
            "stage_type": stage_type,
            "goal": goal,
            "started_at": time.time(),
        }))
        # No extra task_registry call needed — phase index is advanced on stage_done.

    def stage_done(self, stage_type: str, goal: str, summary: str, duration_ms: int = 0, ok: bool = True) -> None:
        self._append(Event(type="stage_done", data={
            "stage_type": stage_type,
            "goal": goal,
            "summary": summary,
            "duration_ms": duration_ms,
            "ok": ok,
        }))
        # Advance phase pointer so TUI shows the next phase as current.
        try:
            from birdclaw.memory.tasks import task_registry
            task_registry.advance_phase(self.session_id)
        except Exception:
            pass

    def task_created(self, request_id: str, original_request: str, step_count: int) -> None:
        self._append(Event(
            type="task_created",
            data={"request_id": request_id, "request": original_request, "steps": step_count},
        ))

    def step_done(self, step_id: str, description: str, result: str) -> None:
        truncated = result[:300] + ("…" if len(result) > 300 else "")
        self._append(Event(
            type="step_done",
            data={"step_id": step_id, "description": description, "result": truncated},
        ))

    def memory_hit(self, query: str, node_names: list[str]) -> None:
        self._append(Event(type="memory_hit", data={"query": query, "nodes": node_names}))

    def session_summary(self, summary: str) -> None:
        """Store a generated summary of the full session so far."""
        self._append(Event(type="session_summary", data={"summary": summary}))

    def usage(self, requests: int, responses: int, total_tokens: int, model: str = "") -> None:
        """Record cumulative usage stats at the end of a loop turn."""
        self._append(Event(type="usage", data={
            "requests": requests,
            "responses": responses,
            "total_tokens": total_tokens,
            "model": model,
        }))

    def compaction(self, removed: int, summary_preview: str) -> None:
        """Record that a compaction occurred."""
        self._append(Event(type="compaction", data={
            "removed_messages": removed,
            "summary_preview": summary_preview[:200],
        }))

    def subtask_manifest(self, stage_index: int, manifest_dict: dict) -> None:
        """Record a subtask manifest snapshot (durable; survives restart)."""
        self._append(Event(type="subtask_manifest", data={
            "stage_index": stage_index,
            "manifest": manifest_dict,
        }))

    def task_spawned(self, task_id: str, task_slug: str, trigger: str, parent_session: str = "") -> None:
        """Record that a background task was spawned from this session."""
        logger.info("[session] task_spawned  task_id=%s  slug=%s  trigger=%s", task_id, task_slug, trigger)
        self._append(Event(type="task_spawned", data={
            "task_id": task_id,
            "task_slug": task_slug,
            "trigger": trigger,
            "parent_session": parent_session,
        }))

    # ── Query helpers ─────────────────────────────────────────────────────────

    def all_events(self) -> list[Event]:
        """Return a snapshot of all events (safe for iteration by external callers)."""
        return list(self._events)

    def events_of_type(self, *types: EventType) -> list[Event]:
        return [e for e in self._events if e.type in types]

    def last_user_messages(self, n: int = 2) -> list[str]:
        """Return the last N raw user message contents (most recent last)."""
        msgs = [e.data["content"] for e in self._events if e.type == "user_message"]
        return msgs[-n:]

    def latest_summary(self) -> str | None:
        """Return the most recently stored session summary, if any."""
        summaries = [e for e in self._events if e.type == "session_summary"]
        return summaries[-1].data["summary"] if summaries else None

    def completed_steps(self) -> list[dict]:
        return [e.data for e in self._events if e.type == "step_done"]

    # ── Context injection ─────────────────────────────────────────────────────

    def planning_context(
        self,
        pending_steps: list[str] | None = None,
        graph_nodes: list[str] | None = None,
    ) -> str:
        """Build the planning-stage context block injected into the system prompt.

        Contains:
          - Full session summary (if generated)
          - Last 2 raw user messages
          - Pending task steps
          - Top-3 graph node snippets
        """
        parts: list[str] = []

        summary = self.latest_summary()
        if summary:
            parts.append(f"Session objective: {summary}")

        recent = self.last_user_messages(2)
        if recent:
            parts.append("Recent requests:\n" + "\n".join(f"  - {m}" for m in recent))

        if pending_steps:
            parts.append("Pending steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(pending_steps)))

        if graph_nodes:
            parts.append("Relevant memory:\n" + "\n".join(f"  - {n}" for n in graph_nodes[:3]))

        return "\n\n".join(parts)

    def executing_context(self, task_context_so_far: str, graph_snippet: str | None = None) -> str:
        """Build the executing-stage context block."""
        parts: list[str] = []
        if task_context_so_far:
            parts.append(f"Progress so far:\n{task_context_so_far}")
        if graph_snippet:
            parts.append(f"Relevant memory:\n{graph_snippet}")
        return "\n\n".join(parts)

    def answering_context(self) -> str:
        """Build the answering-stage context block — completed steps only."""
        done = self.completed_steps()
        if not done:
            return ""
        lines = [f"  - {s['description']}: {s['result']}" for s in done]
        return "Completed steps:\n" + "\n".join(lines)

    # ── Summary generation ────────────────────────────────────────────────────

    def generate_summary(self) -> str:
        """Call the LLM to produce a 1-2 sentence summary of the session so far.

        Stores the result as a session_summary event and returns it.
        Falls back to the latest user message if the LLM call fails.
        """
        from birdclaw.agent.prompts import ANSWER_SCHEMA
        from birdclaw.llm.client import llm_client
        from birdclaw.llm.types import Message

        user_msgs = self.last_user_messages(3)
        if not user_msgs:
            return ""

        prompt = (
            "Summarise what the user is trying to accomplish in 1-2 sentences. "
            "Be concrete. No filler phrases.\n\n"
            "Recent requests:\n" + "\n".join(f"- {m}" for m in user_msgs)
        )

        try:
            result = llm_client.generate(
                [Message(role="user", content=prompt)],
                tools=[ANSWER_SCHEMA],
                tool_choice="auto",
            )
            if result.tool_calls and result.tool_calls[0].name == "answer":
                summary = result.tool_calls[0].arguments.get("content", "").strip()
            else:
                summary = result.content.strip()

            if not summary:
                raise ValueError("empty summary")
        except Exception as e:
            logger.warning("summary generation failed (%s) — using last user message", e)
            summary = user_msgs[-1]

        self.session_summary(summary)
        return summary
