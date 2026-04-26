"""Output pane card widgets — structured widget tree replacing the flat RichLog.

Widget hierarchy inside OutputPane's VerticalScroll:

    PlanBanner    — task outcome + stage pipeline (once per task)
    StageHeader   — stage boundary + live status (one per stage)
    ToolCard      — one per tool call; header always visible, body click-to-expand
    Static        — user_message / assistant_message lines
    ApprovalCard  — inline permission gate with [Y][A][N] buttons + countdown

All widgets update in-place via method calls from OutputPane — no widget
is ever removed and re-added; state changes only update child Statics/RichLogs.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static
from textual.widget import Widget

from birdclaw.tui.render import _tool_colour, spinner_frame


# ---------------------------------------------------------------------------
# PlanBanner
# ---------------------------------------------------------------------------

class PlanBanner(Widget):
    """Task outcome + numbered stage pipeline — shown at the top of each task."""

    DEFAULT_CSS = """
    PlanBanner {
        width: 100%;
        height: auto;
        padding: 0 1 1 1;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-3;
    }
    """

    def __init__(self, outcome: str, steps: list[str], **kw) -> None:
        super().__init__(**kw)
        self._outcome = outcome
        self._steps   = steps

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("Outcome  ", style="bold white")
        t.append(self._outcome, style="white")
        yield Static(t, id="plan-outcome")
        if self._steps:
            for i, step in enumerate(self._steps):
                row = Text("  ")
                row.append(f"{i+1}. ", style="dim")
                row.append(step, style="dim cyan")
                yield Static(row, classes="plan-step")


# ---------------------------------------------------------------------------
# StageHeader
# ---------------------------------------------------------------------------

_STAGE_COLOUR = {
    "research":   "cyan",
    "write_code": "green",
    "write_doc":  "blue",
    "verify":     "yellow",
    "reflect":    "magenta",
}


class StageHeader(Widget):
    """Stage boundary — animated spinner while running, ✓ + duration when done."""

    DEFAULT_CSS = """
    StageHeader {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-top: 1;
        background: $surface-darken-2;
    }
    """

    def __init__(self, stage_type: str, goal: str, index: int, **kw) -> None:
        super().__init__(**kw)
        self._stage_type = stage_type
        self._goal       = goal
        self._index      = index
        self._done       = False
        self._duration   = 0
        self._ok         = True

    def compose(self) -> ComposeResult:
        yield Static("", id="sh-text")

    def on_mount(self) -> None:
        self._draw(tick=0)

    def tick(self, tick: int) -> None:
        if not self._done:
            self._draw(tick)

    def mark_done(self, duration_ms: int, ok: bool = True) -> None:
        self._done     = True
        self._duration = duration_ms
        self._ok       = ok
        self._draw(tick=0)

    def _draw(self, tick: int) -> None:
        colour = _STAGE_COLOUR.get(self._stage_type, "white")
        t = Text()
        t.append("── ", style="dim")
        t.append(self._stage_type, style=f"bold {colour}")
        t.append(f"  {self._goal}", style=colour)
        if self._done:
            dur_s = f"  {self._duration / 1000:.1f}s" if self._duration else ""
            if self._ok:
                t.append(f"  ✓{dur_s}", style="bold green")
            else:
                t.append(f"  ~{dur_s}", style="bold yellow")
        else:
            t.append(f"  {spinner_frame(tick)}", style="bold blue")
        t.append(" ──", style="dim")
        try:
            self.query_one("#sh-text", Static).update(t)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ToolCard
# ---------------------------------------------------------------------------

class ToolCard(Widget):
    """One tool call: compact header always visible; click to expand/collapse body."""

    collapsed: reactive[bool] = reactive(True)

    DEFAULT_CSS = """
    ToolCard {
        width: 100%;
        height: auto;
        padding: 0 0 0 3;
    }
    ToolCard:hover {
        background: $surface-darken-1;
    }
    ToolCard #tc-header {
        width: 100%;
        height: 1;
    }
    ToolCard #tc-body {
        display: none;
        width: 100%;
        height: auto;
        max-height: 14;
        padding: 0 3;
        background: $surface-darken-1;
    }
    ToolCard.-expanded #tc-body {
        display: block;
    }
    """

    def __init__(self, name: str, args: dict, body_visible: bool = False, **kw) -> None:
        super().__init__(**kw)
        self._name         = name
        self._args         = args
        self._status       = "running"   # running | done | failed
        self._duration     = 0
        self._result       = ""
        self._tick         = 0
        self._body_written = False
        # verbose=full → start expanded; verbose=on → collapsed header only
        self._start_expanded = body_visible

    def compose(self) -> ComposeResult:
        yield Static("", id="tc-header")
        yield RichLog(id="tc-body", max_lines=20, wrap=True, markup=False,
                      highlight=False)

    def on_mount(self) -> None:
        self._render_header()
        if self._start_expanded:
            self.collapsed = False

    # ── Interaction ──────────────────────────────────────────────────────────

    def on_click(self) -> None:
        self.collapsed = not self.collapsed

    def watch_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self.remove_class("-expanded")
        else:
            self.add_class("-expanded")
            if not self._body_written:
                self._write_body()

    # ── Live updates ─────────────────────────────────────────────────────────

    def tick(self, tick: int) -> None:
        self._tick = tick
        if self._status == "running":
            self._render_header()

    def complete(self, result: str, duration_ms: int) -> None:
        self._status   = "done"
        self._result   = result
        self._duration = duration_ms
        self._render_header()
        if not self.collapsed and not self._body_written:
            self._write_body()
        elif not self.collapsed:
            # Refresh body with result
            try:
                log = self.query_one("#tc-body", RichLog)
                log.clear()
                self._body_written = False
                self._write_body()
            except Exception:
                pass

    def fail(self, error: str = "") -> None:
        self._status = "failed"
        self._result = error
        self._render_header()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_header(self) -> None:
        colour  = _tool_colour(self._name)
        preview = self._args_preview()
        t = Text()
        if self._status == "running":
            t.append(f"{spinner_frame(self._tick)} ", style="bold blue")
        elif self._status == "done":
            t.append("✓ ", style="bold green")
        else:
            t.append("✘ ", style="bold red")
        t.append(self._name, style=f"bold {colour}")
        if preview:
            t.append(f"  {preview}", style="dim")
        if self._status == "done" and self._duration:
            t.append(f"  {self._duration}ms", style="dim green")
        elif self._status == "running":
            t.append("  …", style="dim blue")
        if self._status == "done" and self.collapsed:
            t.append("  ▶", style="dim")   # expand hint
        try:
            self.query_one("#tc-header", Static).update(t)
        except Exception:
            pass

    def _args_preview(self) -> str:
        if not self._args:
            return ""
        first_val = str(next(iter(self._args.values()), ""))
        return first_val[:55] + ("…" if len(first_val) > 55 else "")

    def _write_body(self) -> None:
        self._body_written = True
        try:
            log = self.query_one("#tc-body", RichLog)
        except Exception:
            return
        # Args section — skip large content fields (write_file body, etc.)
        _SKIP_LARGE = {"content", "text", "body", "data"}
        if self._args:
            log.write(Text("Args", style="dim bold white"))
            for k, v in self._args.items():
                val = str(v)
                if k in _SKIP_LARGE and len(val) > 120:
                    t = Text()
                    t.append(f"  {k}: ", style="dim")
                    t.append(f"[{len(val)} chars]", style="dim italic")
                    log.write(t)
                else:
                    t = Text()
                    t.append(f"  {k}: ", style="dim")
                    t.append(val[:140] + ("…" if len(val) > 140 else ""), style="dim white")
                    log.write(t)
        # Result — diff-style for write/edit, bash-pretty for bash, plain for rest
        if self._result:
            if self._name in ("write_file", "edit_file", "write", "edit"):
                self._write_diff(log)
            elif self._name in ("bash", "run_command"):
                self._write_bash_result(log)
            elif self._name == "web_fetch":
                self._write_web_fetch_result(log)
            elif self._name == "web_search":
                self._write_web_search_result(log)
            else:
                log.write(Text("Result", style="dim bold white"))
                lines = self._result.splitlines()
                for line in lines[:15]:
                    log.write(Text(line, style="dim white"))
                if len(lines) > 15:
                    log.write(Text(f"  [+{len(lines) - 15} more lines]",
                                   style="dim italic"))

    def _write_bash_result(self, log: RichLog) -> None:
        """Render bash tool result in a human-readable format."""
        import json as _json
        try:
            data = _json.loads(self._result)
        except Exception:
            log.write(Text(self._result[:300], style="dim white"))
            return

        # Handle truncated wrapper
        if data.get("truncated"):
            preview_raw = data.get("preview", "")
            try:
                inner = _json.loads(preview_raw)
                stdout = inner.get("stdout", "")
                rc = inner.get("return_code_interpretation", "")
                dur = inner.get("duration_ms", 0)
            except Exception:
                stdout = preview_raw
                rc = ""
                dur = 0
        else:
            stdout = data.get("stdout", "")
            rc = data.get("return_code_interpretation", "")
            dur = data.get("duration_ms", 0)

        exit_ok = "exit_code:0" in rc or rc == "0"
        rc_style = "bold green" if exit_ok else "bold red"
        header = Text()
        header.append("exit ", style="dim")
        header.append(rc.replace("exit_code:", ""), style=rc_style)
        if dur:
            header.append(f"  {dur}ms", style="dim")
        if data.get("truncated"):
            header.append("  [truncated]", style="dim yellow")
        log.write(header)

        if stdout:
            lines = stdout.splitlines()
            for line in lines[:20]:
                log.write(Text(line, style="dim white"))
            if len(lines) > 20:
                log.write(Text(f"  [+{len(lines) - 20} more lines]", style="dim italic"))

    def _write_diff(self, log: RichLog) -> None:
        """Render result as coloured unified diff when tool is write/edit."""
        lines = self._result.splitlines()
        log.write(Text("Output", style="dim bold white"))
        for line in lines[:20]:
            if line.startswith("+") and not line.startswith("+++"):
                log.write(Text(line, style="green"))
            elif line.startswith("-") and not line.startswith("---"):
                log.write(Text(line, style="red"))
            elif line.startswith("@@"):
                log.write(Text(line, style="cyan"))
            elif line.startswith(("---", "+++")):
                log.write(Text(line, style="bold dim"))
            else:
                log.write(Text(line, style="dim white"))
        if len(lines) > 20:
            log.write(Text(f"  [+{len(lines) - 20} more lines]", style="dim italic"))

    def _write_web_fetch_result(self, log: RichLog) -> None:
        import json as _json
        try:
            data = _json.loads(self._result)
        except Exception:
            log.write(Text(self._result[:300], style="dim white"))
            return
        # Handle truncation wrapper
        if data.get("truncated"):
            try:
                inner = _json.loads(data.get("preview", "{}"))
            except Exception:
                inner = {}
            content = inner.get("content", data.get("preview", ""))
            url     = inner.get("url", data.get("url", ""))
            truncated = True
        else:
            content   = data.get("content", "")
            url       = data.get("url", "")
            truncated = False
        header = Text()
        header.append("url  ", style="dim")
        header.append(url[:80], style="cyan")
        if truncated:
            header.append("  [truncated]", style="dim yellow")
        log.write(header)
        if content:
            lines = content.splitlines()
            for line in lines[:18]:
                log.write(Text(line[:120], style="dim white"))
            if len(lines) > 18:
                log.write(Text(f"  [+{len(lines) - 18} more lines]", style="dim italic"))

    def _write_web_search_result(self, log: RichLog) -> None:
        import json as _json
        try:
            results = _json.loads(self._result)
            if not isinstance(results, list):
                raise ValueError
        except Exception:
            log.write(Text(self._result[:300], style="dim white"))
            return
        for item in results[:6]:
            t = Text()
            t.append("● ", style="bold cyan")
            t.append((item.get("title") or "")[:70], style="white")
            log.write(t)
            url_line = Text()
            url_line.append("  ", style="")
            url_line.append((item.get("url") or "")[:80], style="dim cyan")
            log.write(url_line)
            snippet = (item.get("content") or item.get("snippet") or "").strip()[:100]
            if snippet:
                log.write(Text(f"  {snippet}", style="dim white"))
        if len(results) > 6:
            log.write(Text(f"  [+{len(results) - 6} more results]", style="dim italic"))


# ---------------------------------------------------------------------------
# ApprovalCard
# ---------------------------------------------------------------------------

class ApprovalCard(Widget):
    """Inline permission gate — appears in the output pane for the focused task.

    Resolves the approval via approval_queue.resolve() when a button is clicked
    or the countdown expires. Removes itself from the DOM on resolution.
    """

    DEFAULT_CSS = """
    ApprovalCard {
        width: 100%;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        margin: 1 0;
        background: $surface-darken-1;
    }
    ApprovalCard #ac-info {
        height: auto;
        margin-bottom: 1;
    }
    ApprovalCard #ac-buttons {
        height: 3;
        layout: horizontal;
    }
    ApprovalCard #ac-btn-allow  { width: 16; margin-right: 1; }
    ApprovalCard #ac-btn-always { width: 14; margin-right: 1; }
    ApprovalCard #ac-btn-deny   { width: 10; }
    ApprovalCard #ac-timer {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, approval_id: str, tool_name: str, description: str,
                 expires_at: float, **kw) -> None:
        super().__init__(**kw)
        self._approval_id = approval_id
        self._tool_name   = tool_name
        self._description = description
        self._expires_at  = expires_at

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("⚠  Permission Required\n", style="bold yellow")
        t.append(f"   Tool:  ", style="dim")
        t.append(f"{self._tool_name}\n", style="white")
        t.append(f"   {self._description[:120]}", style="dim white")
        yield Static(t, id="ac-info")
        with Horizontal(id="ac-buttons"):
            yield Button("[Y] Allow once", id="ac-btn-allow",  variant="success")
            yield Button("[A] Always",     id="ac-btn-always", variant="warning")
            yield Button("[N] Deny",       id="ac-btn-deny",   variant="error")
        yield Static("", id="ac-timer")

    def on_mount(self) -> None:
        self.border_title = "APPROVAL REQUIRED"
        self._render_timer()

    def tick(self, tick: int) -> None:    # noqa: ARG002
        self._render_timer()
        if time.time() >= self._expires_at:
            self._resolve("deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "ac-btn-allow":  "allow_once",
            "ac-btn-always": "allow_always",
            "ac-btn-deny":   "deny",
        }
        decision = mapping.get(event.button.id or "")
        if decision:
            self._resolve(decision)
        event.stop()

    def _resolve(self, decision: str) -> None:
        from birdclaw.agent.approvals import approval_queue
        approval_queue.resolve(self._approval_id, decision)   # type: ignore[arg-type]
        self.remove()

    def _render_timer(self) -> None:
        secs = max(0, int(self._expires_at - time.time()))
        style = "bold yellow" if secs < 15 else "dim"
        try:
            self.query_one("#ac-timer", Static).update(
                Text(f"   ⏱ {secs}s remaining — /approve {self._approval_id[:6]}", style=style)
            )
        except Exception:
            pass

