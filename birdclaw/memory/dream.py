"""Standalone dream cycle — runs all memory consolidation phases in order.

Phases:
  1. memorise  — ingest pages + session logs into GraphRAG (one unit at a time,
                 stops early if a task starts)
  2. reflect   — merge session_graph → knowledge_graph, surface patterns
  3. inner life — synthesise reflections into agent's inner life doc
  4. user knowledge — extract user facts from completed tasks
  5. self-concept — update identity reasoning from session logs
  6. cleanup   — prune stale sessions, tasks, pages, backups

Called from:
  - python main.py dream  (CLI)
  - cron system (nightly at 03:00)
"""

from __future__ import annotations

import logging
import time
import threading

logger = logging.getLogger(__name__)

# Maximum time (seconds) to wait for running tasks to finish before starting.
_WAIT_TIMEOUT = 600   # 10 minutes
_WAIT_POLL    = 15    # check every 15 s

# Singleton lock — prevents two dream cycles running concurrently.
_dream_lock = threading.Lock()


def _tasks_running() -> bool:
    try:
        from birdclaw.memory.tasks import task_registry
        return bool(task_registry.list(status="running"))
    except Exception:
        return False


def run_dream_cycle(*, quiet: bool = False) -> dict[str, str]:
    """Run all dream phases. Returns {phase: status} for each phase.

    quiet=True suppresses console output (used when called from cron).
    """
    if not _dream_lock.acquire(blocking=False):
        logger.info("dream: already running — skipping")
        return {"dream": "skipped (already running)"}

    results: dict[str, str] = {}
    try:
        _run(results, quiet=quiet)
    finally:
        _dream_lock.release()
    return results


def _run(results: dict[str, str], quiet: bool) -> None:
    from birdclaw.gateway.notify import push_notification

    def log(msg: str) -> None:
        logger.info("dream: %s", msg)
        if not quiet:
            from rich.console import Console
            Console().print(f"[cyan]{msg}[/]")

    def ok(phase: str, detail: str = "") -> None:
        results[phase] = f"ok{': ' + detail if detail else ''}"
        log(f"✓ {phase}" + (f" — {detail}" if detail else ""))

    def err(phase: str, e: Exception) -> None:
        results[phase] = f"error: {e}"
        logger.error("dream: %s failed: %s", phase, e)
        if not quiet:
            from rich.console import Console
            Console().print(f"[red]{phase} failed:[/] {e}")

    push_notification("Dreaming started…", title="Dream", severity="information")

    # ── Wait for idle ─────────────────────────────────────────────────────────
    deadline = time.time() + _WAIT_TIMEOUT
    while _tasks_running() and time.time() < deadline:
        logger.info("dream: tasks running — waiting %ds before starting", _WAIT_POLL)
        time.sleep(_WAIT_POLL)

    if _tasks_running():
        logger.info("dream: tasks still running after %ds — proceeding at MEMORY priority", _WAIT_TIMEOUT)

    # stop_fn passed to memorise: abort after each unit if a task starts.
    stop_fn = _tasks_running

    # ── Phase 1: memorise ─────────────────────────────────────────────────────
    try:
        from birdclaw.memory.memorise import run_memorise
        from birdclaw.tools.context_vars import set_llm_priority
        from birdclaw.llm.scheduler import LLMPriority
        set_llm_priority(LLMPriority.MEMORY)
        n = run_memorise(stop_fn=stop_fn)
        ok("memorise", f"{n} unit(s)")
    except Exception as e:
        err("memorise", e)

    # ── Phase 2: graph reflection ─────────────────────────────────────────────
    try:
        from birdclaw.memory.memorise import _dream_reflect
        seen: dict[str, bool] = {}
        _dream_reflect(seen)
        ok("graph")
    except Exception as e:
        err("graph", e)

    # ── Phase 3: inner life ───────────────────────────────────────────────────
    try:
        from birdclaw.memory.inner_life import update_from_reflections
        n = update_from_reflections()
        ok("inner life", f"{n} reflection(s)")
    except Exception as e:
        err("inner life", e)

    # ── Phase 4: user knowledge ───────────────────────────────────────────────
    try:
        from birdclaw.memory.memorise import _extract_user_knowledge_from_tasks
        _extract_user_knowledge_from_tasks()
        ok("user knowledge")
    except Exception as e:
        err("user knowledge", e)

    # ── Phase 5: self-concept ─────────────────────────────────────────────────
    try:
        from birdclaw.memory.self_concept import update_self_concept
        updated = update_self_concept()
        ok("self-concept", "updated" if updated else "no new reasoning")
    except Exception as e:
        err("self-concept", e)

    # ── Phase 6: cleanup ──────────────────────────────────────────────────────
    try:
        from birdclaw.memory.cleanup import run_cleanup
        removed = run_cleanup()
        total = sum(removed.values())
        ok("cleanup", f"{total} removed" if total else "nothing to remove")
    except Exception as e:
        err("cleanup", e)

    push_notification(
        f"Dream complete — {len([v for v in results.values() if v.startswith('ok')])} phases ok",
        title="Dream",
        severity="information",
    )
    log("Dreaming complete.")
