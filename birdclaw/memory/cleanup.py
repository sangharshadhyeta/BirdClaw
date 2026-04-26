"""File cleanup — prune stale data that is no longer useful for learning or dreaming.

Retention policy:
  sessions/     Keep if not yet memorised OR younger than MAX_SESSION_AGE_DAYS.
                Memorised sessions are safe to delete — their knowledge is in the graph.
  tasks/        Keep running/failed tasks indefinitely. Delete completed tasks
                older than MAX_TASK_AGE_DAYS (their outcomes are in the graph).
  pages/        Delete files older than MAX_PAGE_AGE_HOURS — web content is transient.
  self_update/  Keep only the last MAX_SELF_UPDATE_BACKUPS backup snapshots.
  stage_history Trim to last MAX_STAGE_HISTORY_ENTRIES lines — older entries no
                longer improve budget P75 estimates significantly.

Call run_cleanup() at the end of each dream cycle, or via:
    python main.py cleanup
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from birdclaw.config import settings

logger = logging.getLogger(__name__)

MAX_SESSION_AGE_DAYS      = 7
MAX_TASK_AGE_DAYS         = 3
MAX_PAGE_AGE_HOURS        = 24
MAX_SELF_UPDATE_BACKUPS   = 5
MAX_STAGE_HISTORY_ENTRIES = 1000


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _memorised_ids() -> set[str]:
    """Return session IDs that have already been processed by memorise."""
    path = settings.data_dir / "memory" / "memorised.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("sessions", {}).keys())
    except Exception:
        return set()


def cleanup_sessions(max_age_days: int = MAX_SESSION_AGE_DAYS) -> int:
    """Delete session log files that are both memorised AND older than max_age_days.

    Keeps un-memorised sessions regardless of age — they still contain
    knowledge the dreaming cycle hasn't extracted yet.
    """
    sessions_dir = settings.sessions_dir
    if not sessions_dir.exists():
        return 0

    memorised = _memorised_ids()
    cutoff = time.time() - max_age_days * 86400
    deleted = 0

    for f in sessions_dir.glob("*.jsonl"):
        session_id = f.stem
        if session_id not in memorised:
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
                logger.debug("cleanup: deleted session %s", session_id)
        except OSError as e:
            logger.warning("cleanup: could not delete session %s: %s", session_id, e)

    return deleted


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def cleanup_tasks(max_age_days: int = MAX_TASK_AGE_DAYS) -> int:
    """Delete completed task files older than max_age_days.

    Running and failed tasks are kept — they may still be referenced or retried.
    """
    tasks_dir = settings.data_dir / "tasks"
    if not tasks_dir.exists():
        return 0

    cutoff = time.time() - max_age_days * 86400
    deleted = 0

    for f in tasks_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") not in ("completed", "done"):
                continue
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
                logger.debug("cleanup: deleted task %s", f.stem)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("cleanup: could not process task %s: %s", f.name, e)

    return deleted


# ---------------------------------------------------------------------------
# Page store
# ---------------------------------------------------------------------------

def cleanup_pages(max_age_hours: int = MAX_PAGE_AGE_HOURS) -> int:
    """Delete page store files past their age limit.

    Web content is transient — stale pages are not useful for model learning.
    """
    pages_dir = settings.data_dir / "pages"
    if not pages_dir.exists():
        return 0

    cutoff = time.time() - max_age_hours * 3600
    deleted = 0

    for f in pages_dir.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError as e:
            logger.warning("cleanup: could not delete page %s: %s", f.name, e)

    return deleted


# ---------------------------------------------------------------------------
# Self-update backups
# ---------------------------------------------------------------------------

def cleanup_self_update_backups(keep: int = MAX_SELF_UPDATE_BACKUPS) -> int:
    """Keep only the N most recent self-update backup snapshots."""
    backup_dir = settings.data_dir / "self_update"
    if not backup_dir.exists():
        return 0

    backups = sorted(
        [d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    deleted = 0
    for old in backups[keep:]:
        try:
            import shutil
            shutil.rmtree(old)
            deleted += 1
            logger.debug("cleanup: deleted self_update backup %s", old.name)
        except OSError as e:
            logger.warning("cleanup: could not delete backup %s: %s", old.name, e)

    return deleted


# ---------------------------------------------------------------------------
# Stage history
# ---------------------------------------------------------------------------

def cleanup_stage_history(max_entries: int = MAX_STAGE_HISTORY_ENTRIES) -> int:
    """Trim stage_history.jsonl to the most recent max_entries lines.

    Older entries no longer improve P75 estimates — recent history is more
    representative of current model behaviour after any fine-tuning or updates.
    """
    from birdclaw.agent.budget import history_path
    path = history_path()
    if not path.exists():
        return 0

    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return 0

    if len(lines) <= max_entries:
        return 0

    trimmed = len(lines) - max_entries
    try:
        path.write_text("\n".join(lines[-max_entries:]) + "\n", encoding="utf-8")
        logger.debug("cleanup: trimmed %d old stage_history entries", trimmed)
    except OSError as e:
        logger.warning("cleanup: could not trim stage_history: %s", e)
        return 0

    return trimmed


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_cleanup() -> dict[str, int]:
    """Run all cleanup routines. Returns summary of what was deleted/trimmed."""
    results = {
        "sessions_deleted":       cleanup_sessions(),
        "tasks_deleted":          cleanup_tasks(),
        "pages_deleted":          cleanup_pages(),
        "backups_deleted":        cleanup_self_update_backups(),
        "stage_history_trimmed":  cleanup_stage_history(),
    }
    total = sum(results.values())
    if total:
        logger.info("cleanup complete: %s", results)
    else:
        logger.debug("cleanup: nothing to remove")
    return results
