"""Task registry — lifecycle management for agent tasks.

Python port of claw-code-parity/rust/crates/runtime/src/task_registry.rs,
extended with context, expected_outcome, agent_id, and started_at/ended_at.

Every task tracks:
  - what it is (prompt, description)
  - what it needs to know (context)
  - how we know it worked (expected_outcome)
  - who is running it (agent_id)
  - when it ran (started_at, ended_at, duration_ms)
  - what happened (messages[], output)

Persistence:
    ~/.birdclaw/tasks/<task_id>.json   one file per task, written on every mutation

Thread safety:
    All mutations go through a single threading.Lock() — mirrors Arc<Mutex<>> in Rust.

Module-level singleton:
    from birdclaw.memory.tasks import task_registry
    task = task_registry.create("do the thing", context="use requests lib")
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from birdclaw.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Title generation — 2-5 meaningful words from a prompt (no LLM, no deps)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "is", "are", "was", "be", "do",
    "can", "could", "would", "should", "will", "me", "my", "your", "its",
    "this", "that", "it", "i", "you", "we", "they", "all", "any", "some",
    "how", "what", "which", "where", "when", "please", "just", "also",
    "into", "about", "as", "get", "make", "let", "so", "then", "than",
    "using", "use", "used", "run", "tell", "show", "give", "find", "now",
})


def _make_title(prompt: str, max_words: int = 4) -> str:
    """Extract 2-4 content words from a prompt to form a short display title.

    Strips stop words, punctuation, and numbers. Falls back to the first
    max_words words verbatim if nothing meaningful remains.
    """
    import re
    # Drop everything after a newline (task prompts are often first-line only)
    first_line = prompt.split("\n")[0].strip()
    # Tokenise — keep only alphabetic tokens
    tokens = re.findall(r"[a-zA-Z]+", first_line)
    # Filter stop words; keep capitalisation of first token for display
    content = [t for t in tokens if t.lower() not in _STOP_WORDS]
    chosen = content[:max_words] if content else tokens[:max_words]
    if not chosen:
        chosen = tokens[:max_words] if tokens else [prompt[:20]]
    # Title-case the result (e.g. "Financial SAP Audit")
    return " ".join(w.capitalize() for w in chosen)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TaskStatus = Literal["created", "waiting", "running", "completed", "failed", "stopped"]

_TERMINAL: frozenset[str] = frozenset({"completed", "failed", "stopped"})


@dataclass
class TaskMessage:
    role: str        # "user" | "assistant" | "system"
    content: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskMessage":
        return cls(role=d["role"], content=d["content"], ts=d.get("ts", 0.0))


@dataclass
class Task:
    task_id: str
    prompt: str
    status: TaskStatus = "created"
    description: str = ""
    title: str = ""              # 2-5 word display name generated from prompt
    # Our additions — not in claw-code-parity
    context: str = ""           # what the task needs to know to succeed
    expected_outcome: str = ""  # how we verify success
    agent_id: str = ""          # which orchestration agent is running this
    after_task_id: str = ""     # if set, stay "waiting" until this task completes
    session_id: str = ""        # TUI/CLI session that spawned this task
    # Timestamps (float Unix epoch — matches Port 1)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: int = 0
    # Content
    messages: list[TaskMessage] = field(default_factory=list)
    output: str = ""
    team_id: str = ""
    # Phase tracking — updated by agent loop, used by TUI task pane tree
    phases: list[str] = field(default_factory=list)  # plan steps (plain English)
    current_phase_index: int = -1      # index into phases list; -1 = not started
    completed_phase_count: int = 0     # how many phases have finished
    # Subtask manifest — live section/function progress within a format stage
    subtask_manifest: dict | None = None  # serialised SubtaskManifest (avoid circular import)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status,
            "description": self.description,
            "title": self.title,
            "context": self.context,
            "expected_outcome": self.expected_outcome,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "messages": [m.to_dict() for m in self.messages],
            "output": self.output,
            "team_id": self.team_id,
            "after_task_id": self.after_task_id,
            "session_id": self.session_id,
            "phases": self.phases,
            "current_phase_index": self.current_phase_index,
            "completed_phase_count": self.completed_phase_count,
            "subtask_manifest": self.subtask_manifest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        t = cls(
            task_id=d["task_id"],
            prompt=d["prompt"],
            status=d.get("status", "created"),
            description=d.get("description", ""),
            title=d.get("title", ""),
            context=d.get("context", ""),
            expected_outcome=d.get("expected_outcome", ""),
            agent_id=d.get("agent_id", ""),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            started_at=d.get("started_at", 0.0),
            ended_at=d.get("ended_at", 0.0),
            duration_ms=d.get("duration_ms", 0),
            output=d.get("output", ""),
            team_id=d.get("team_id", ""),
            after_task_id=d.get("after_task_id", ""),
            session_id=d.get("session_id", ""),
            phases=d.get("phases", []),
            current_phase_index=d.get("current_phase_index", -1),
            completed_phase_count=d.get("completed_phase_count", 0),
            subtask_manifest=d.get("subtask_manifest"),
        )
        t.messages = [TaskMessage.from_dict(m) for m in d.get("messages", [])]
        return t


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TaskRegistry:
    """Thread-safe in-memory task registry with disk persistence.

    Mirrors task_registry.rs: all mutations are serialised through a lock.
    Each task is saved as an individual JSON file so the registry survives
    process restarts (load_all() on init).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._counter: int = 0
        self._load_all()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _tasks_dir(self) -> Path:
        d = settings.data_dir / "tasks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, task_id: str) -> Path:
        return self._tasks_dir() / f"{task_id}.json"

    def _save(self, task: Task) -> None:
        """Write a single task to disk atomically (called inside the lock)."""
        try:
            dest = self._path(task.task_id)
            tmp  = dest.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(task.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(dest)
        except OSError as e:
            logger.warning("could not save task %s: %s", task.task_id, e)

    _STALE_ACTIVE_HOURS = 2  # tasks "running/created/waiting" older than this are zombie

    def _load_all(self, live_session_ids: "set[str] | None" = None) -> None:
        """Load all task JSON files on startup.

        Any task in an active state (running/created/waiting) whose session_id
        is NOT in live_session_ids is from a dead session — stop it immediately.
        This is the primary orphan detection mechanism: session identity beats age.
        As a fallback, tasks with no session_id that are older than
        _STALE_ACTIVE_HOURS are also stopped.
        """
        import time as _time
        stale_cutoff = _time.time() - self._STALE_ACTIVE_HOURS * 3600
        live = live_session_ids or set()
        try:
            d = self._tasks_dir()
        except Exception:
            return
        for p in d.glob("task_*.json"):
            try:
                task = Task.from_dict(json.loads(p.read_text(encoding="utf-8")))
                if task.status in ("running", "created", "waiting"):
                    sid = task.session_id or ""
                    # Primary: session no longer alive → orphan
                    orphan_by_session = bool(sid) and sid not in live
                    # No session + never started → no agent will ever pick this up
                    orphan_no_session = not sid and task.started_at == 0.0
                    # Fallback: has session but old
                    orphan_by_age = bool(sid) and task.created_at < stale_cutoff
                    if orphan_by_session or orphan_no_session or orphan_by_age:
                        reason = (
                            f"session {sid} ended" if orphan_by_session
                            else "no session, never started" if orphan_no_session
                            else "stale"
                        )
                        task.status = "stopped"
                        task.output = (task.output or "") + f"\n[stopped: {reason}]"
                        logger.info(
                            "[task] load  task_id=%s  status=%s→stopped  prompt=%r  reason=%s",
                            task.task_id, task.status, task.prompt[:50], reason,
                        )
                        self._tasks[task.task_id] = task
                        self._save(task)
                        continue
                    # Genuinely active — log it clearly
                    logger.info(
                        "[task] load  task_id=%s  status=%s  session=%s  prompt=%r",
                        task.task_id, task.status, sid or "none", task.prompt[:50],
                    )
                self._tasks[task.task_id] = task
            except Exception as e:
                logger.warning("could not load task %s: %s", p.name, e)
        logger.info("[task] load complete  total=%d", len(self._tasks))

    def sync_from_disk(self) -> int:
        """Reload tasks written by another process (daemon). Returns count of new/updated tasks.

        Called periodically by the TUI's poll loop so tasks created by the daemon
        appear in the task pane without a full restart.
        """
        try:
            d = self._tasks_dir()
        except Exception:
            return 0
        changed = 0
        for p in d.glob("task_*.json"):
            try:
                mtime = p.stat().st_mtime
                raw = p.read_text(encoding="utf-8")
                task = Task.from_dict(json.loads(raw))
                with self._lock:
                    existing = self._tasks.get(task.task_id)
                    if existing is None or existing.updated_at < task.updated_at:
                        self._tasks[task.task_id] = task
                        changed += 1
            except Exception:
                continue
        return changed

    # ── Factory ───────────────────────────────────────────────────────────────

    def create(
        self,
        prompt: str,
        *,
        description: str = "",
        context: str = "",
        expected_outcome: str = "",
        after_task_id: str = "",
        session_id: str = "",
    ) -> Task:
        """Create a new task in Created status. Returns the Task."""
        with self._lock:
            self._counter += 1
            ts = time.time()
            # task_<hex_secs>_<counter> — same format as Rust
            task_id = f"task_{int(ts):08x}_{self._counter}"
            task = Task(
                task_id=task_id,
                prompt=prompt,
                title=_make_title(prompt),
                description=description,
                context=context,
                expected_outcome=expected_outcome,
                after_task_id=after_task_id,
                session_id=session_id,
                created_at=ts,
                updated_at=ts,
            )
            self._tasks[task_id] = task
            self._save(task)
            logger.info("[task] created  task_id=%s  prompt=%r  session=%s", task_id, prompt[:60], session_id or "none")
            return task

    # ── Reads (no lock needed for immutable snapshot) ─────────────────────────

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return Task.from_dict(task.to_dict()) if task else None

    def list(self, status: TaskStatus | None = None) -> list[Task]:
        """List all tasks, optionally filtered by status. Sorted by created_at desc."""
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def output(self, task_id: str) -> str:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            return task.output

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)

    def is_empty(self) -> bool:
        return len(self) == 0

    # ── Mutations ─────────────────────────────────────────────────────────────

    def set_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            task.status = status
            task.updated_at = time.time()
            self._save(task)

    def start(self, task_id: str, agent_id: str = "") -> Task:
        """Mark task as running, record started_at and agent_id."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            now = time.time()
            task.status = "running"
            task.started_at = now
            task.updated_at = now
            if agent_id:
                task.agent_id = agent_id
            self._save(task)
            logger.info("[task] start  task_id=%s  agent_id=%s", task_id, agent_id or "none")
            return Task.from_dict(task.to_dict())

    def complete(self, task_id: str, output: str = "") -> Task:
        """Mark task completed with final output."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            now = time.time()
            task.status = "completed"
            task.ended_at = now
            task.updated_at = now
            if output:
                task.output = output
            if task.started_at:
                task.duration_ms = int((now - task.started_at) * 1000)
            self._save(task)
            logger.info("[task] complete  task_id=%s  dur=%.1fs  output_chars=%d",
                        task_id, task.duration_ms / 1000, len(task.output or ""))
            return Task.from_dict(task.to_dict())

    def fail(self, task_id: str, reason: str = "") -> Task:
        """Mark task failed with a reason appended to output."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            now = time.time()
            task.status = "failed"
            task.ended_at = now
            task.updated_at = now
            if reason:
                task.output = (task.output + "\n" + reason).strip()
            if task.started_at:
                task.duration_ms = int((now - task.started_at) * 1000)
            self._save(task)
            logger.warning("[task] fail  task_id=%s  dur=%.1fs  reason=%r",
                           task_id, task.duration_ms / 1000, reason[:60])
            return Task.from_dict(task.to_dict())

    def stop(self, task_id: str) -> Task:
        """Stop a non-terminal task. Raises ValueError if already terminal."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            if task.status in _TERMINAL:
                raise ValueError(
                    f"task {task_id} is already in terminal state: {task.status}"
                )
            now = time.time()
            task.status = "stopped"
            task.ended_at = now
            task.updated_at = now
            if task.started_at:
                task.duration_ms = int((now - task.started_at) * 1000)
            self._save(task)
            logger.debug("task stopped: %s", task_id)
            return Task.from_dict(task.to_dict())

    def append_message(self, task_id: str, content: str, role: str = "user") -> Task:
        """Append a message (stage summary, event, note) to the task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            task.messages.append(TaskMessage(role=role, content=content))
            task.updated_at = time.time()
            self._save(task)
            return Task.from_dict(task.to_dict())

    def append_output(self, task_id: str, text: str) -> None:
        """Append streaming output text to a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            task.output += text
            task.updated_at = time.time()
            self._save(task)

    def set_agent(self, task_id: str, agent_id: str) -> None:
        """Record which agent is running this task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            task.agent_id = agent_id
            task.updated_at = time.time()
            self._save(task)

    def assign_team(self, task_id: str, team_id: str) -> None:
        """Group this task under a team for multi-agent coordination."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            task.team_id = team_id
            task.updated_at = time.time()
            self._save(task)

    def set_phases(self, task_id: str, phases: list[str]) -> None:
        """Store the plan's phase list when the plan is generated."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.phases = [p[:80] for p in phases]
            task.current_phase_index = 0 if phases else -1
            task.updated_at = time.time()
            self._save(task)

    def add_phase_after_current(self, task_id: str, goal: str) -> None:
        """Insert a dynamically-added stage right after the current phase index."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            idx = task.current_phase_index
            insert_at = (idx + 1) if idx >= 0 else len(task.phases)
            task.phases.insert(insert_at, goal[:80])
            task.current_phase_index = insert_at  # point to newly inserted stage
            task.updated_at = time.time()
            self._save(task)

    def advance_phase(self, task_id: str) -> None:
        """Mark current phase complete and advance to the next one."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if task.current_phase_index >= 0:
                task.completed_phase_count += 1
                task.current_phase_index += 1
                if task.current_phase_index >= len(task.phases):
                    task.current_phase_index = -1  # all done
            task.updated_at = time.time()
            self._save(task)

    def set_manifest(self, task_id: str, manifest_dict: dict) -> None:
        """Store a serialised SubtaskManifest in the task (TUI reads this)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.subtask_manifest = manifest_dict
            task.updated_at = time.time()
            self._save(task)

    def get_manifest(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.subtask_manifest if task else None

    def remove(self, task_id: str) -> Task | None:
        """Remove a task from the registry and delete its file."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                try:
                    self._path(task_id).unlink(missing_ok=True)
                except OSError:
                    pass
            return task


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

task_registry = TaskRegistry()
