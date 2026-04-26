"""Workspace log — reads and writes BIRDCLAW.md in the project working directory.

BIRDCLAW.md accumulates task history, findings, and file inventory for the
current project. The agent reads it at the start of each task (for context)
and appends to it after completion (for persistence).

Format:
    # BIRDCLAW Project Log
    ## [task_id] Task Title
    **Goal:** ...
    **Result:** ... (full output, not truncated)
    **Files changed:** ...
    ---
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_FILENAME   = "BIRDCLAW.md"
_MAX_ENTRY_CHARS = 8000   # per-task result limit (not truncated mid-sentence)


# ---------------------------------------------------------------------------
# Path resolution — always cwd, never the install dir
# ---------------------------------------------------------------------------

def _log_path(cwd: Path | None = None) -> Path:
    """Return the BIRDCLAW.md path in the working (launch) directory."""
    import os as _os
    base = cwd or Path(_os.getcwd()).resolve()
    return base / _LOG_FILENAME


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_workspace_log(cwd: Path | None = None) -> str:
    """Read BIRDCLAW.md from the project working directory."""
    try:
        return _log_path(cwd).read_text(encoding="utf-8")
    except OSError:
        return ""


def read_file_inventory(cwd: Path | None = None) -> str:
    """Scan the working directory and return a human-readable file inventory.

    For text files: shows the first meaningful line as a description.
    For binary/large files: shows size.
    Skips hidden dirs, __pycache__, node_modules, .git, etc.
    """
    import os as _os

    base = cwd or Path(_os.getcwd()).resolve()

    _SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "env", "venv",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
    }
    _TEXT_EXTS = {
        ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json",
        ".js", ".ts", ".rs", ".go", ".sh", ".html", ".css", ".env",
        ".cfg", ".ini", ".rst", ".ipynb",
    }

    lines = [f"Files in {base}:"]
    try:
        for dirpath, dirnames, filenames in _os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            )
            rel_dir = Path(dirpath).relative_to(base)
            if len(rel_dir.parts) > 3:
                dirnames.clear()
                continue
            for fname in sorted(filenames):
                fpath  = Path(dirpath) / fname
                rel    = fpath.relative_to(base)
                ext    = fpath.suffix.lower()
                try:
                    size = fpath.stat().st_size
                except OSError:
                    lines.append(f"  {rel}  [unreadable]")
                    continue

                if size > 10 * 1024 * 1024:
                    lines.append(f"  {rel}  [{size // 1024}KB — large, skipped]")
                    continue

                if ext in _TEXT_EXTS:
                    first_line = ""
                    try:
                        with fpath.open(encoding="utf-8", errors="replace") as f:
                            for ln in f:
                                ln = ln.strip()
                                if ln and not ln.startswith("#!"):
                                    first_line = ln[:120]
                                    break
                    except OSError:
                        pass
                    lines.append(
                        f"  {rel}  — {first_line}" if first_line else f"  {rel}"
                    )
                else:
                    lines.append(f"  {rel}  [{size} bytes]")
    except Exception as exc:
        logger.warning("workspace_log: inventory scan failed: %s", exc)
        lines.append(f"  (scan error: {exc})")

    return "\n".join(lines)


def read_for_context(cwd: Path | None = None, max_chars: int = 2000) -> str:
    """Read BIRDCLAW.md for injection into agent context.

    Only returns content when BIRDCLAW.md exists in the working directory —
    i.e., the agent has been run here before. Returns empty string otherwise
    so the context window isn't wasted on a new/unrelated directory.

    File inventory is intentionally excluded: it can be arbitrarily large
    and blows the context window when run from large directories.
    """
    import os as _os
    base = cwd or Path(_os.getcwd()).resolve()
    log_path = base / _LOG_FILENAME

    if not log_path.exists():
        return ""

    log = read_workspace_log(cwd)
    if not log.strip():
        return ""

    # Keep the most recent portion — recent history matters most
    if len(log) > max_chars:
        log = "…(earlier entries omitted — see BIRDCLAW.md)\n" + log[-max_chars:]

    return f"=== BIRDCLAW.md (project history) ===\n{log.strip()}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def append_task_entry(
    task_id:       str,
    title:         str,
    goal:          str,
    result:        str,
    files_changed: list[str] | None = None,
    cwd:           Path | None      = None,
) -> None:
    """Append a completed task entry to BIRDCLAW.md.

    ``result`` should be the FULL task output — not truncated.
    Entries exceeding _MAX_ENTRY_CHARS are soft-truncated with a note
    (the full output is always in the session log).
    """
    path     = _log_path(cwd)
    ts       = time.strftime("%Y-%m-%d %H:%M")
    short_id = task_id
    heading  = title or goal[:60]

    if len(result) > _MAX_ENTRY_CHARS:
        result_text = (
            result[:_MAX_ENTRY_CHARS]
            + "\n\n*(output truncated — full result in session log)*"
        )
    else:
        result_text = result

    entry_lines = [
        f"\n## [{short_id}] {heading}",
        f"*{ts}*",
        "",
        f"**Goal:** {goal}",
        "",
        "**Result:**",
        result_text,
    ]

    if files_changed:
        entry_lines += ["", "**Files changed:**"]
        for f in (files_changed or [])[:20]:
            entry_lines.append(f"- `{f}`")

    entry_lines.append("\n---")
    entry = "\n".join(entry_lines)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            header = (
                "# BIRDCLAW Project Log\n\n"
                "Generated and maintained by the BirdClaw autonomous agent.\n"
                "Each entry records a completed task: goal, full result, and changed files.\n"
            )
            path.write_text(header + entry, encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        logger.info("workspace_log: wrote entry for task %s → %s", short_id, path)
    except OSError as exc:
        logger.warning("workspace_log: write failed: %s", exc)