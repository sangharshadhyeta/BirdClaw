"""TUI overlays — modal screens that float above the three-pane layout.

Each overlay uses Textual's ModalScreen so the background remains visible
through the margins (background: $background 60% makes it semi-transparent).

Overlays:
  ShellOverlay   — interactive shell, Ctrl+S or /shell
  HelpOverlay    — full command reference, ? or /help
"""

from __future__ import annotations

import threading
import time

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, ListItem, Label, RichLog, Static


# ---------------------------------------------------------------------------
# Shell overlay
# ---------------------------------------------------------------------------

class ShellOverlay(ModalScreen):
    """Interactive shell floating above the main layout.

    - 6% margin on all sides — background bleeds through
    - Scrollable output (RichLog)
    - Command history: ↑/↓ to navigate
    - x or Escape to close
    - Commands run via async bash registry, output streamed back
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    DEFAULT_CSS = """
    ShellOverlay {
        align: center middle;
        background: $background 55%;
    }
    #shell-container {
        width: 88%;
        height: 88%;
        border: double $primary;
        background: $surface;
        layout: vertical;
    }
    #shell-output {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
    }
    #shell-divider {
        height: 1;
        background: $primary-darken-3;
        color: $text-muted;
        padding: 0 1;
    }
    #shell-input-row {
        height: 3;
        layout: horizontal;
    }
    #shell-prompt-label {
        width: 4;
        height: 3;
        content-align: center middle;
        color: $success;
        padding: 0 1;
    }
    #shell-input {
        width: 1fr;
        height: 3;
        border: none;
        background: $surface-darken-1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._cmd_history: list[str] = []
        self._history_idx: int = -1

    def compose(self) -> ComposeResult:
        with Vertical(id="shell-container"):
            yield RichLog(id="shell-output", highlight=True, markup=False,
                          max_lines=500, wrap=True)
            yield Static("─" * 80, id="shell-divider")
            with Horizontal(id="shell-input-row"):
                yield Static("$", id="shell-prompt-label")
                yield Input(placeholder="command… (↑↓=history, Escape=close)", id="shell-input")

    def on_mount(self) -> None:
        c = self.query_one("#shell-container")
        c.border_title = "SHELL"
        c.border_subtitle = "x=close  ↑↓=history  Escape=close"
        log = self.query_one("#shell-output", RichLog)
        log.write(Text("Shell ready. Type a command and press Enter.", style="dim"))
        log.write(Text("Type 'x' or press Escape to close.", style="dim"))
        self.query_one("#shell-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "shell-input":
            return
        cmd = event.value.strip()
        event.input.value = ""
        self._history_idx = -1
        if not cmd:
            return
        if cmd in ("x", "exit", "close"):
            self.dismiss()
            return
        self._cmd_history.insert(0, cmd)
        self._exec(cmd)

    def on_key(self, event) -> None:
        inp = self.query_one("#shell-input", Input)
        if event.key == "x" and not inp.value:
            self.dismiss()
            event.stop()
            return
        if event.key == "up":
            if self._cmd_history:
                self._history_idx = min(self._history_idx + 1,
                                        len(self._cmd_history) - 1)
                inp.value = self._cmd_history[self._history_idx]
                inp.cursor_position = len(inp.value)
            event.stop()
        elif event.key == "down":
            if self._history_idx > 0:
                self._history_idx -= 1
                inp.value = self._cmd_history[self._history_idx]
                inp.cursor_position = len(inp.value)
            elif self._history_idx == 0:
                self._history_idx = -1
                inp.value = ""
            event.stop()

    def _exec(self, cmd: str) -> None:
        log = self.query_one("#shell-output", RichLog)
        prompt = Text()
        prompt.append("$ ", style="bold green")
        prompt.append(cmd, style="white")
        log.write(prompt)

        def _run() -> None:
            import json
            from birdclaw.tools.bash import bash_poll, run_bash
            result = json.loads(run_bash(cmd, background=True))
            if "error" in result:
                self.app.call_from_thread(
                    log.write, Text(result["error"], style="red"))
                return
            sid = result["session_id"]
            while True:
                poll = json.loads(bash_poll(sid))
                if poll["status"] != "running":
                    break
                time.sleep(0.1)
            stdout = (poll.get("stdout_tail") or "").rstrip()
            stderr = (poll.get("stderr_tail") or "").rstrip()
            exit_code = poll.get("exit_code", 0)
            if stdout:
                for line in stdout.splitlines():
                    self.app.call_from_thread(
                        log.write, Text(line, style="dim white"))
            if stderr:
                for line in stderr.splitlines():
                    self.app.call_from_thread(
                        log.write, Text(line, style="dim red"))
            style = "dim green" if exit_code == 0 else "dim red"
            self.app.call_from_thread(
                log.write, Text(f"[exit {exit_code}]", style=style))

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Help overlay
# ---------------------------------------------------------------------------

_HELP_MD = """\
# BirdClaw TUI — Command Reference

## Global keys
| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+C` | Clear input → warn → stop running task |
| `Ctrl+D` | Exit (alternate) |
| `Ctrl+S` | Open shell overlay |
| `Ctrl+L` | Model selector |
| `Ctrl+G` | Agent selector |
| `Ctrl+P` | Command palette (fuzzy search) |
| `Tab` / `Shift+Tab` | Cycle pane focus |
| `[` / `]` | Shrink / grow conversation pane |
| `?` | This help screen |

## Task list pane
| Key | Action |
|-----|--------|
| `Enter` | Focus output on selected task |
| `F1` | Show standing (scheduled) tasks |
| `F2` | Show active tasks |

## Output pane
| Key | Action |
|-----|--------|
| `Ctrl+O` | Toggle card view / raw log |
| `Ctrl+T` | Toggle thinking cards |
| `Click`  | Expand / collapse a tool card |

## Conversation pane
| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+E` | Toggle multiline edit mode |
| `↑` / `↓` | Browse message history (when input empty) |

## Shell overlay
| Key | Action |
|-----|--------|
| `Enter` | Run command |
| `↑` / `↓` | Navigate command history |
| `x` or `Escape` | Close shell |

## Slash commands
| Command | Action |
|---------|--------|
| `/help` | This help screen |
| `/shell` | Open shell overlay |
| `/clear` | Clear conversation pane |
| `/model` | Open model selector |
| `/agents` | Open agent selector |
| `/abort` | Stop focused running task |
| `/status` | Show gateway status |
| `/tasks` | List all tasks |
| `/standing` | List standing (scheduled) tasks |
| `/buddy` | Toggle companion panel (show/hide) |

## Spinner states
| Icon | Meaning |
|------|---------|
| ⠋⠙⠹… | Running / thinking |
| ✔ | Completed |
| ✘ | Failed |
| ⏹ | Stopped |
| ○ | Queued |
"""


class HelpOverlay(ModalScreen):
    """Full command reference, scrollable."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
        background: $background 55%;
    }
    #help-container {
        width: 72%;
        height: 84%;
        border: solid $primary;
        background: $surface;
        layout: vertical;
    }
    #help-log {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield RichLog(id="help-log", markup=False, max_lines=300)

    def on_mount(self) -> None:
        c = self.query_one("#help-container")
        c.border_title = "HELP"
        c.border_subtitle = "Escape or ? to close"
        log = self.query_one("#help-log", RichLog)
        log.write(Markdown(_HELP_MD))


# ---------------------------------------------------------------------------
# Fuzzy picker overlay (model selector, session picker)
# ---------------------------------------------------------------------------

class FuzzyPickerOverlay(ModalScreen):
    """Generic searchable list picker.

    Caller provides a list of (display_label, value) pairs.
    Dismissed with the selected value, or None on cancel.

    Usage:
        def callback(value):
            if value: do_something(value)
        self.app.push_screen(FuzzyPickerOverlay("Model", items), callback)
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    FuzzyPickerOverlay {
        align: center middle;
        background: $background 55%;
    }
    #picker-container {
        width: 60%;
        height: 70%;
        max-width: 80;
        border: double $primary;
        background: $surface;
        layout: vertical;
    }
    #picker-search {
        height: 3;
        border: none;
        border-bottom: solid $primary-darken-3;
        background: $surface-darken-1;
        padding: 0 1;
    }
    #picker-list {
        height: 1fr;
        background: transparent;
    }
    #picker-list ListItem { padding: 0 1; }
    #picker-list ListItem:hover { background: $primary-darken-3; }
    #picker-list ListItem.-highlighted { background: $primary-darken-2; }
    """

    def __init__(self, title: str,
                 items: list[tuple[str, str]], **kw) -> None:
        super().__init__(**kw)
        self._title   = title
        self._all     = items          # [(label, value), …]
        self._visible = list(items)

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-container"):
            yield Input(placeholder=f"Filter {self._title}…", id="picker-search")
            yield ListView(id="picker-list")

    def on_mount(self) -> None:
        c = self.query_one("#picker-container")
        c.border_title = self._title
        c.border_subtitle = "Enter=select  Escape=cancel"
        self._rebuild_list()
        self.query_one("#picker-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.lower()
        self._visible = [(l, v) for l, v in self._all
                         if q in l.lower() or q in v.lower()]
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#picker-list", ListView)
        lv.clear()
        for label, _ in self._visible:
            lv.append(ListItem(Label(label)))
        if self._visible:
            lv.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = self.query_one("#picker-list", ListView).index
        if idx is not None and 0 <= idx < len(self._visible):
            self.dismiss(self._visible[idx][1])
        else:
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "enter":
            lv = self.query_one("#picker-list", ListView)
            idx = lv.index
            if idx is not None and 0 <= idx < len(self._visible):
                self.dismiss(self._visible[idx][1])
                event.stop()


# ---------------------------------------------------------------------------
# Output search overlay  (Ctrl+F)
# ---------------------------------------------------------------------------

class SearchOverlay(ModalScreen):
    """Grep through the current task's session log. Highlights matching lines."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    DEFAULT_CSS = """
    SearchOverlay {
        align: center middle;
        background: $background 55%;
    }
    #search-container {
        width: 80%;
        height: 80%;
        border: double $primary;
        background: $surface;
        layout: vertical;
    }
    #search-input {
        height: 3;
        border: none;
        border-bottom: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    #search-results {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
    }
    #search-status {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, session_log_path: str, **kw) -> None:
        super().__init__(**kw)
        self._log_path = session_log_path

    def compose(self) -> ComposeResult:
        with Vertical(id="search-container"):
            yield Input(placeholder="Search session log… (regex OK)", id="search-input")
            yield RichLog(id="search-results", max_lines=500, wrap=True, markup=False)
            yield Static("", id="search-status")

    def on_mount(self) -> None:
        c = self.query_one("#search-container")
        c.border_title = "SEARCH"
        c.border_subtitle = "Escape to close"
        self.query_one("#search-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._run_search(event.value.strip())

    def _run_search(self, query: str) -> None:
        import json, re
        log  = self.query_one("#search-results", RichLog)
        stat = self.query_one("#search-status", Static)
        log.clear()
        if not query:
            stat.update("")
            return
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            stat.update(Text("Invalid regex", style="red"))
            return
        try:
            lines = open(self._log_path, encoding="utf-8").readlines()
        except OSError:
            stat.update(Text("Log not found", style="red"))
            return
        hits = 0
        for raw in lines:
            try:
                record = json.loads(raw)
            except Exception:
                continue
            flat = json.dumps(record, ensure_ascii=False)
            if pattern.search(flat):
                hits += 1
                rtype = record.get("type", "?")
                data  = record.get("data", {})
                # Pull out the most useful text field
                text = (data.get("content") or data.get("result") or
                        data.get("goal") or data.get("outcome") or
                        str(data))[:200]
                t = Text()
                t.append(f"[{rtype}] ", style="dim cyan")
                # Highlight the matching part
                try:
                    m = pattern.search(text)
                    if m:
                        t.append(text[:m.start()], style="dim white")
                        t.append(text[m.start():m.end()], style="bold yellow")
                        t.append(text[m.end():], style="dim white")
                    else:
                        t.append(text, style="dim white")
                except Exception:
                    t.append(text, style="dim white")
                log.write(t)
        stat.update(Text(f"  {hits} match{'es' if hits != 1 else ''} in {len(lines)} events",
                         style="dim"))
