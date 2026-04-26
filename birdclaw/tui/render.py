"""Output pane renderers — pretty and raw views for session_log events.

Two modes toggled by the user (Ctrl+O):
  pretty  — human-readable, colour-coded per event type
  raw     — the raw JSONL line, monospace, useful for debugging

Spinner frames taken directly from claw-code-parity/rust/crates/rusty-claude-cli/src/render.rs.
"""

from __future__ import annotations

import json

from rich.text import Text


# ---------------------------------------------------------------------------
# Spinner (ported from render.rs Spinner::FRAMES)
# ---------------------------------------------------------------------------

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Colours matching render.rs ColorTheme defaults
SPINNER_ACTIVE  = "blue"
SPINNER_DONE    = "green"
SPINNER_FAILED  = "red"


def spinner_frame(tick: int) -> str:
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def spinner_text(label: str, tick: int) -> Text:
    t = Text()
    t.append(spinner_frame(tick), style=f"bold {SPINNER_ACTIVE}")
    t.append(f" {label}", style="dim")
    return t


def spinner_done(label: str) -> Text:
    t = Text()
    t.append("✔ ", style=f"bold {SPINNER_DONE}")
    t.append(label, style=SPINNER_DONE)
    return t


def spinner_failed(label: str) -> Text:
    t = Text()
    t.append("✘ ", style=f"bold {SPINNER_FAILED}")
    t.append(label, style=SPINNER_FAILED)
    return t


# ---------------------------------------------------------------------------
# Task status badge
# ---------------------------------------------------------------------------

_STATUS: dict[str, tuple[str, str]] = {
    "running":   ("", SPINNER_ACTIVE),   # icon replaced by live spinner
    "created":   ("○", "dim"),
    "completed": ("✔", SPINNER_DONE),
    "failed":    ("✘", SPINNER_FAILED),
    "stopped":   ("⏹", "dim"),
}


def status_badge(status: str, tick: int = 0) -> Text:
    if status == "running":
        icon = spinner_frame(tick)
        colour = SPINNER_ACTIVE
    else:
        icon, colour = _STATUS.get(status, ("?", "dim"))
    t = Text()
    t.append(icon, style=f"bold {colour}")
    return t


# ---------------------------------------------------------------------------
# Pretty renderer — one Text line per session_log record
# ---------------------------------------------------------------------------

def _tool_colour(name: str) -> str:
    """Return a Rich colour for a tool name based on its category."""
    n = name.lower()
    if n in ("bash", "run_bash", "bash_poll"):
        return "cyan"
    if n in ("read_file", "read", "code_search", "find_symbol"):
        return "green"
    if n in ("web_search", "web_fetch", "search"):
        return "yellow"
    if n in ("write_file", "edit_file", "write", "edit"):
        return "red"
    if n in ("think",):
        return "magenta"
    return "blue"


_STAGE_COLOUR = {
    "research":   "cyan",
    "write_code": "green",
    "write_doc":  "blue",
    "verify":     "yellow",
    "reflect":    "magenta",
}


def render_pretty(record: dict) -> Text | None:
    rtype = record.get("type", "")
    t = Text()

    if rtype == "plan":
        d = record.get("data", {})
        outcome = d.get("outcome", "")
        steps   = d.get("steps", [])
        t.append("Plan  ", style="bold white")
        t.append(outcome[:70], style="white")
        if steps:
            return t  # steps shown individually via stage_start/stage_done

    elif rtype == "stage_start":
        d    = record.get("data", {})
        goal = d.get("goal", "")
        t.append("  ○ ", style="dim")
        t.append(goal[:80], style="dim white")

    elif rtype == "stage_done":
        d       = record.get("data", {})
        goal    = d.get("goal", "")
        dur     = d.get("duration_ms", 0)
        summary = d.get("summary", "")
        t.append("  ✔ ", style="bold green")
        t.append(goal[:60], style="green")
        t.append(f"  ({dur / 1000:.1f}s)", style="dim")
        if summary and summary != goal:
            t.append(f"  {summary[:60]}", style="dim white")

    elif rtype == "tool_call":
        d    = record.get("data", {})
        name = d.get("name", "")
        args = d.get("arguments", {})
        preview = str(next(iter(args.values()), ""))[:60] if args else ""
        colour = _tool_colour(name)
        t.append("  › ", style=f"bold {colour}")
        t.append(name, style=f"bold {colour}")
        t.append(f"({preview})", style="dim")

    elif rtype == "tool_result":
        d      = record.get("data", {})
        name   = d.get("name", "")
        result = d.get("result", "")
        dur    = d.get("duration_ms", 0)
        colour = _tool_colour(name)
        lines  = result.splitlines()
        preview = lines[0][:80] if lines else ""
        hidden  = len(lines) - 1
        t.append("    ← ", style=f"dim {colour}")
        t.append(name, style=colour)
        t.append(f"  {preview}", style="dim")
        if hidden > 0:
            t.append(f"  [+{hidden} lines]", style="dim italic")
        if dur:
            t.append(f"  ({dur}ms)", style="dim")

    elif rtype == "user_message":
        content = record.get("data", {}).get("content", "")
        t.append("You:      ", style="bold white")
        t.append(content[:120], style="white")

    elif rtype == "assistant_message":
        content = record.get("data", {}).get("content", "")
        t.append("BirdClaw: ", style="bold green")
        t.append(content[:120], style="green")

    else:
        return None

    return t


# ---------------------------------------------------------------------------
# Raw renderer
# ---------------------------------------------------------------------------

def render_raw(record: dict) -> Text:
    return Text(json.dumps(record, ensure_ascii=False), style="dim white", no_wrap=True)


# ---------------------------------------------------------------------------
# Conversation turn renderer
# ---------------------------------------------------------------------------

def render_turn(role: str, content: str) -> Text:
    t = Text()
    if role == "user":
        t.append("You      ", style="bold white")
        t.append(content, style="white")
    elif role == "assistant":
        t.append("BirdClaw ", style="bold green")
        t.append(content, style="green")
    else:
        t.append(f"{role[:8]:<8} ", style="bold dim")
        t.append(content, style="dim")
    return t
