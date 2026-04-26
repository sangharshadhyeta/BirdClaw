"""Cron service — scheduled execution of skills (standing goals).

Skills with a `schedule` frontmatter field become standing goals that fire
automatically. The cron service boots as a daemon thread, checks due entries
every 30s, and submits tasks to the gateway as if the user had asked for them.

Schedule formats:
    "0 9 * * *"     — standard 5-field cron expression (UTC)
    "every:3600"    — interval in seconds (fires every N seconds from first run)
    "every:1h"      — human interval: Ns, Nm, Nh, Nd (seconds/minutes/hours/days)

Persistence:
    ~/.birdclaw/cron/jobs.json   — one entry per scheduled skill

Usage:
    from birdclaw.skills.cron import cron_service
    cron_service.start()          # idempotent — safe to call multiple times
    cron_service.list()           # → list[CronEntry]
    cron_service.trigger("name")  # run a skill immediately
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECK_INTERVAL = 30.0    # seconds between due-job sweeps
_SKILL_USER_ID  = "cron"  # gateway user_id for cron-triggered tasks


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _parse_every(spec: str) -> float | None:
    """Parse 'every:N' or 'every:Xh' → interval seconds. Returns None on error."""
    raw = spec[len("every:"):].strip()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if raw and raw[-1] in multipliers:
        try:
            return float(raw[:-1]) * multipliers[raw[-1]]
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def next_run_after(schedule: str, after: float | None = None) -> float | None:
    """Compute the next run time (Unix timestamp) for a schedule expression.

    Returns None if the schedule is invalid.
    """
    now = after or time.time()

    if schedule.startswith("every:"):
        interval = _parse_every(schedule)
        if interval is None:
            return None
        return now + interval

    # Standard cron expression
    try:
        from croniter import croniter
        return croniter(schedule, now).get_next(float)
    except Exception as e:
        logger.warning("cron: invalid schedule %r: %s", schedule, e)
        return None


# ---------------------------------------------------------------------------
# CronEntry
# ---------------------------------------------------------------------------

@dataclass
class CronEntry:
    cron_id:      str
    skill_name:   str
    schedule:     str
    description:  str   = ""
    enabled:      bool  = True
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)
    last_run_at:  float = 0.0
    next_run_at:  float = 0.0
    run_count:    int   = 0

    def is_due(self) -> bool:
        return self.enabled and self.next_run_at > 0 and time.time() >= self.next_run_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CronEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# CronRegistry — file-backed JSON store
# ---------------------------------------------------------------------------

class CronRegistry:
    """Persistent store of cron entries at ~/.birdclaw/cron/jobs.json."""

    def __init__(self) -> None:
        self._lock:    threading.Lock          = threading.Lock()
        self._entries: dict[str, CronEntry]    = {}
        self._loaded:  bool                    = False

    def _path(self) -> Path:
        from birdclaw.config import settings
        return settings.data_dir / "cron" / "jobs.json"

    def _load(self) -> None:
        if self._loaded:
            return
        p = self._path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for d in data:
                    e = CronEntry.from_dict(d)
                    self._entries[e.cron_id] = e
                logger.debug("cron: loaded %d entries", len(self._entries))
            except Exception as ex:
                logger.warning("cron: failed to load jobs.json: %s", ex)
        self._loaded = True

    def _save(self) -> None:
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries.values()]
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def upsert(self, skill_name: str, schedule: str, description: str = "") -> CronEntry:
        """Create or update the cron entry for a skill."""
        with self._lock:
            self._load()
            # Find existing by skill_name
            existing = next(
                (e for e in self._entries.values() if e.skill_name == skill_name),
                None,
            )
            if existing:
                existing.schedule    = schedule
                existing.description = description
                existing.updated_at  = time.time()
                existing.next_run_at = next_run_after(schedule) or 0.0
                self._save()
                return existing

            cron_id = f"cron_{uuid.uuid4().hex[:10]}"
            entry = CronEntry(
                cron_id=cron_id,
                skill_name=skill_name,
                schedule=schedule,
                description=description,
                next_run_at=next_run_after(schedule) or 0.0,
            )
            self._entries[cron_id] = entry
            self._save()
            logger.info("cron: registered %s → %s", skill_name, schedule)
            return entry

    def get(self, cron_id: str) -> CronEntry | None:
        with self._lock:
            self._load()
            return self._entries.get(cron_id)

    def get_by_skill(self, skill_name: str) -> CronEntry | None:
        with self._lock:
            self._load()
            return next(
                (e for e in self._entries.values() if e.skill_name == skill_name),
                None,
            )

    def list(self, enabled_only: bool = False) -> list[CronEntry]:
        with self._lock:
            self._load()
            entries = list(self._entries.values())
        if enabled_only:
            entries = [e for e in entries if e.enabled]
        return sorted(entries, key=lambda e: e.skill_name)

    def enable(self, cron_id: str) -> bool:
        with self._lock:
            self._load()
            e = self._entries.get(cron_id)
            if e is None:
                return False
            e.enabled    = True
            e.updated_at = time.time()
            e.next_run_at = next_run_after(e.schedule) or 0.0
            self._save()
        return True

    def disable(self, cron_id: str) -> bool:
        with self._lock:
            self._load()
            e = self._entries.get(cron_id)
            if e is None:
                return False
            e.enabled    = False
            e.updated_at = time.time()
            self._save()
        return True

    def delete(self, cron_id: str) -> bool:
        with self._lock:
            self._load()
            if cron_id not in self._entries:
                return False
            del self._entries[cron_id]
            self._save()
        return True

    def record_run(self, cron_id: str) -> None:
        with self._lock:
            self._load()
            e = self._entries.get(cron_id)
            if e is None:
                return
            e.last_run_at = time.time()
            e.run_count  += 1
            e.next_run_at = next_run_after(e.schedule, after=e.last_run_at) or 0.0
            e.updated_at  = time.time()
            self._save()

    def due_entries(self) -> list[CronEntry]:
        with self._lock:
            self._load()
            return [e for e in self._entries.values() if e.is_due()]


# ---------------------------------------------------------------------------
# CronService — background thread + skill sync
# ---------------------------------------------------------------------------

class CronService:
    """Background daemon: sync skill schedules → registry, fire due entries."""

    def __init__(self) -> None:
        self._registry        = CronRegistry()
        self._lock            = threading.Lock()
        self._worker_started  = False
        # Built-in system jobs: name → (callable, schedule, description, last_run_at)
        self._system_jobs: dict[str, tuple] = {}
        self._system_last_run: dict[str, float] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the cron worker thread (idempotent)."""
        with self._lock:
            if self._worker_started:
                return
            self._worker_started = True

        self._sync_skills()   # register scheduled skills before first tick

        t = threading.Thread(target=self._worker, daemon=True, name="cron-worker")
        t.start()
        logger.info("cron: service started")

    # ── Public API ────────────────────────────────────────────────────────────

    def list(self) -> list[CronEntry]:
        return self._registry.list()

    def register_system_job(
        self,
        name: str,
        fn,
        schedule: str,
        description: str = "",
    ) -> None:
        """Register a built-in system job that calls fn() directly (no agent loop).

        System jobs are in-memory only — re-registered at each daemon startup.
        They fire at the given cron schedule alongside skill-based jobs.
        """
        self._system_jobs[name] = (fn, schedule, description)
        logger.info("cron: registered system job %r → %s", name, schedule)

    def trigger(self, skill_name: str) -> bool:
        """Fire a skill immediately, regardless of its schedule. Returns False if not found."""
        from birdclaw.skills.loader import load_skills
        skills = {s.name: s for s in load_skills()}
        skill = skills.get(skill_name)
        if skill is None:
            logger.warning("cron: trigger — skill %r not found", skill_name)
            return False
        self._fire(skill_name, skill.description, skill.body)
        return True

    def enable(self, cron_id: str) -> bool:
        return self._registry.enable(cron_id)

    def disable(self, cron_id: str) -> bool:
        return self._registry.disable(cron_id)

    def delete(self, cron_id: str) -> bool:
        return self._registry.delete(cron_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            try:
                self._sync_skills()
                self._fire_due()
                self._fire_system_due()
            except Exception as e:
                logger.error("cron worker error: %s", e)
            time.sleep(_CHECK_INTERVAL)

    def _fire_system_due(self) -> None:
        """Check and fire any system jobs whose next scheduled time has passed."""
        now = time.time()
        for name, (fn, schedule, description) in list(self._system_jobs.items()):
            last = self._system_last_run.get(name, 0.0)
            next_run = next_run_after(schedule, after=last or None) or 0.0
            if now < next_run:
                continue
            self._system_last_run[name] = now
            logger.info("cron: firing system job %r", name)
            threading.Thread(
                target=self._call_system_job,
                args=(name, fn),
                daemon=True,
                name=f"system-job-{name}",
            ).start()

    def _call_system_job(self, name: str, fn) -> None:
        try:
            fn()
        except Exception as e:
            logger.error("cron: system job %r failed: %s", name, e)

    def _sync_skills(self) -> None:
        """Register any skills with a `schedule` field that aren't in the registry yet."""
        try:
            from birdclaw.skills.loader import load_skills
            for skill in load_skills():
                if not skill.schedule:
                    continue
                existing = self._registry.get_by_skill(skill.name)
                if existing is None:
                    self._registry.upsert(
                        skill_name=skill.name,
                        schedule=skill.schedule,
                        description=skill.description,
                    )
                elif existing.schedule != skill.schedule:
                    # Schedule changed in the skill file — update
                    self._registry.upsert(
                        skill_name=skill.name,
                        schedule=skill.schedule,
                        description=skill.description,
                    )
        except Exception as e:
            logger.error("cron: skill sync failed: %s", e)

    def _fire_due(self) -> None:
        for entry in self._registry.due_entries():
            logger.info("cron: firing %s (run #%d)", entry.skill_name, entry.run_count + 1)
            try:
                from birdclaw.skills.loader import load_skills
                skills = {s.name: s for s in load_skills()}
                skill = skills.get(entry.skill_name)
                prompt = skill.body if skill else entry.description or entry.skill_name
                self._fire(entry.skill_name, entry.description, prompt)
                self._registry.record_run(entry.cron_id)
            except Exception as e:
                logger.error("cron: failed to fire %s: %s", entry.skill_name, e)

    def _fire(self, skill_name: str, description: str, prompt: str) -> None:
        """Spawn a task directly via the orchestrator (bypasses soul routing)."""
        from birdclaw.agent.orchestrator import orchestrator
        from birdclaw.llm.scheduler import LLMPriority
        from birdclaw.memory.tasks import task_registry
        from birdclaw.tools.context_vars import set_llm_priority
        set_llm_priority(LLMPriority.CRON)

        task = task_registry.create(
            prompt=prompt,
            context=f"This is a scheduled skill run: {skill_name}",
            expected_outcome=description or f"Complete the {skill_name} skill runbook",
        )
        # Tag the task so the TUI task list can show it as a standing task
        task_registry.assign_team(task.task_id, f"skill:{skill_name}")

        agent_id = orchestrator.spawn(
            task.task_id, prompt,
            context=f"skill:{skill_name}",
            expected_outcome=description,
        )
        task_registry.set_agent(task.task_id, agent_id)
        logger.info("cron: spawned task %s for skill %r", task.task_id[:8], skill_name)

        # Post-completion: ingest session log findings into knowledge graph
        threading.Thread(
            target=self._ingest_after_complete,
            args=(task.task_id, skill_name),
            daemon=True,
            name=f"skill-ingest:{skill_name}",
        ).start()

    def _ingest_after_complete(self, task_id: str, skill_name: str) -> None:
        """Wait for the skill task to finish, then ingest its findings into GraphRAG."""
        from birdclaw.memory.tasks import task_registry
        from birdclaw.config import settings

        # Poll until terminal state (max 2 hours)
        deadline = time.time() + 7200
        while time.time() < deadline:
            task = task_registry.get(task_id)
            if task is None:
                return
            if task.status in ("completed", "failed", "stopped"):
                break
            time.sleep(30)
        else:
            return  # timed out

        # Mine the session log for research findings
        session_log_path = settings.sessions_dir / f"{task_id}.jsonl"
        if not session_log_path.exists():
            return
        try:
            import json as _json
            from birdclaw.memory.retrieval import extract_and_index
            count = 0
            with session_log_path.open(encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        record = _json.loads(raw)
                    except Exception:
                        continue
                    if record.get("type") not in ("tool_result", "assistant_message"):
                        continue
                    data = record.get("data", {})
                    text = data.get("result") or data.get("content") or ""
                    if text and len(text) > 30:
                        count += extract_and_index(
                            text,
                            context=f"skill:{skill_name}",
                        )
            logger.info("cron: ingested %d entities from skill %r (task %s)",
                        count, skill_name, task_id[:8])
        except Exception as e:
            logger.warning("cron: ingest failed for skill %r: %s", skill_name, e)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

cron_service = CronService()
