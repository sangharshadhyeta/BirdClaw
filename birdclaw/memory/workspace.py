"""Workspace state snapshot — injected into every agent turn.

Captures a lightweight, always-current picture of what the agent is working
in. Injected as part of the system prompt so the model never needs to ask
"where am I?" or "what files changed?".

Snapshot fields:
    cwd             current working directory
    recent_files    last N files modified within workspace roots (by mtime)
    active_tasks    pending TaskSteps from the most recent TaskList, if any

All fields are optional — missing data is omitted rather than filled with
placeholder text. The rendered output must stay under CHAR_CAP characters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from birdclaw.config import settings

logger = logging.getLogger(__name__)

RECENT_FILE_COUNT = 8    # max files to list
MAX_DEPTH = 4            # how many directory levels to scan
CHAR_CAP = 600           # hard cap on rendered snapshot text


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceSnapshot:
    cwd: str
    recent_files: list[str] = field(default_factory=list)
    active_tasks: list[str] = field(default_factory=list)  # step descriptions

    def render(self) -> str:
        """Compact text block for system prompt injection."""
        parts: list[str] = [f"Working directory: {self.cwd}"]

        if self.recent_files:
            files_block = "Recent files:\n" + "\n".join(f"  {f}" for f in self.recent_files)
            parts.append(files_block)

        if self.active_tasks:
            tasks_block = "Pending steps:\n" + "\n".join(
                f"  {i+1}. {t}" for i, t in enumerate(self.active_tasks)
            )
            parts.append(tasks_block)

        rendered = "\n\n".join(parts)
        if len(rendered) > CHAR_CAP:
            rendered = rendered[:CHAR_CAP] + "\n[workspace snapshot truncated]"
        return rendered


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------

def _recent_modified_files(
    roots: list[Path],
    max_files: int = RECENT_FILE_COUNT,
    max_depth: int = MAX_DEPTH,
) -> list[str]:
    """Walk workspace roots and return the most recently modified file paths."""
    entries: list[tuple[float, str]] = []

    for root in roots:
        if not root.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune depth
                depth = len(Path(dirpath).relative_to(root).parts)
                if depth >= max_depth:
                    dirnames.clear()
                    continue

                # Skip hidden dirs and common noise
                dirnames[:] = [
                    d for d in dirnames
                    if not d.startswith(".")
                    and d not in ("__pycache__", "node_modules", ".git", "venv", ".venv", "dist", "build")
                ]

                for fname in filenames:
                    if fname.startswith("."):
                        continue
                    fpath = Path(dirpath) / fname
                    try:
                        mtime = fpath.stat().st_mtime
                        entries.append((mtime, str(fpath)))
                    except OSError:
                        continue
        except PermissionError:
            continue

    entries.sort(reverse=True)
    return [path for _, path in entries[:max_files]]


# ---------------------------------------------------------------------------
# Active task loader
# ---------------------------------------------------------------------------

def _active_task_steps() -> list[str]:
    """Load pending step descriptions from the most recently created TaskList.

    Returns an empty list if no task files exist or all tasks are complete.
    """
    tasks_dir = settings.data_dir / "tasks"
    if not tasks_dir.exists():
        return []

    task_files = sorted(tasks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not task_files:
        return []

    try:
        import json
        data = json.loads(task_files[0].read_text(encoding="utf-8"))
        steps = data.get("steps", [])
        return [
            s["description"]
            for s in steps
            if s.get("status") == "pending"
        ]
    except Exception as e:
        logger.debug("could not load active task: %s", e)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapshot() -> WorkspaceSnapshot:
    """Capture the current workspace state."""
    cwd = os.getcwd()
    recent = _recent_modified_files(settings.workspace_roots)
    tasks = _active_task_steps()

    return WorkspaceSnapshot(
        cwd=cwd,
        recent_files=recent,
        active_tasks=tasks,
    )


def render() -> str:
    """Convenience: snapshot + render in one call."""
    return snapshot().render()
