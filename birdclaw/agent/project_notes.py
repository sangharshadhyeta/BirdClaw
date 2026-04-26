"""Project notes — keeps BIRDCLAW.md in the working directory up to date.

`loop.py` already calls `update_project_notes()` at the end of every task
(see the try/except block in `run_agent_loop`).  This module implements that
function and routes it through `workspace_log.py`.

It also exposes `inject_workspace_context()` for use at the top of
`run_agent_loop` so the agent reads BIRDCLAW.md before planning.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def update_project_notes(
    cwd: Path,
    question: str,
    completed_stages: list[dict],
) -> None:
    """Append a task summary entry to BIRDCLAW.md in the project directory.

    Called by run_agent_loop() after every completed task. Never raises —
    failures are logged and swallowed so the agent loop is never blocked.

    ``completed_stages`` is the list of {"type", "goal", "summary"} dicts
    accumulated by _run_loop.  We build a human-readable result from their
    summaries and write one entry to BIRDCLAW.md.
    """
    try:
        from birdclaw.memory.workspace_log import append_task_entry
        from birdclaw.tools.context_vars import get_task_id
        from birdclaw.memory.tasks import task_registry

        task_id = get_task_id() or ""
        title = ""
        if task_id:
            task = task_registry.get(task_id)
            title = getattr(task, "title", "") if task else ""

        # Build result text from stage summaries — full, not truncated
        if completed_stages:
            result_lines = []
            for s in completed_stages:
                stage_type = s.get("type", "?")
                summary    = s.get("summary", s.get("goal", ""))
                result_lines.append(f"**{stage_type}:** {summary}")
            result = "\n".join(result_lines)
        else:
            result = "(no stages recorded)"

        # Collect files changed from write_code / write_doc / edit_file stages
        files_changed: list[str] = []
        for s in completed_stages:
            stype = s.get("type", "")
            if stype in ("write_code", "write_doc", "edit_file"):
                summary = s.get("summary", "")
                # Summary often starts with "Wrote N chars to /path/file"
                import re as _re
                m = _re.search(r"(?:to|edited)\s+(/[\w./\-]+\.\w+)", summary)
                if m:
                    files_changed.append(m.group(1))

        append_task_entry(
            task_id=task_id,
            title=title,
            goal=question,
            result=result,
            files_changed=files_changed,
            cwd=cwd,
        )
    except Exception as exc:
        logger.warning("project_notes: update failed: %s", exc)


def workspace_context_for_task(question: str) -> str:
    """Return relevant context from prior task logs for injection at task start.

    Searches all {task_slug}/BIRDCLAW.md files in cwd for lines relevant to
    the current question, so the agent knows what has been done before.
    Falls back to cwd/BIRDCLAW.md (legacy location) if no task logs found.

    Returns empty string on a fresh workspace or read error.
    """
    import os
    try:
        from birdclaw.tools.line_search import search_relevant
        cwd = Path(os.getcwd()).resolve()

        # Primary: per-task BIRDCLAW.md files in subdirs
        task_logs = sorted(cwd.glob("*/BIRDCLAW.md"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if task_logs:
            ctx = search_relevant(question, task_logs[:10], context_lines=2, max_results=6)
            if ctx:
                return f"=== Prior task history (relevant to your goal) ===\n{ctx}"

        # Fallback: legacy cwd-root BIRDCLAW.md
        from birdclaw.memory.workspace_log import read_for_context
        return read_for_context(cwd=cwd, max_chars=3000)
    except Exception as exc:
        logger.debug("project_notes: workspace context read failed: %s", exc)
        return ""