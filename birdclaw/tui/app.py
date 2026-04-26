"""BirdClaw TUI — three-pane multi-agent interface.

Layout (default 70/30 top/bottom split):

    ┌─ Header: BirdClaw · session · model ───────────────────┐
    │ TASKS [Active|Standing] │ OUTPUT [task] (raw/pretty)   │
    │  ⠼ research task  (anim)│  ▶ [research] find docs      │
    │  ✔ fix bug               │  › web_search(…)            │
    │  ○ write doc             │  ← web_search (210ms)       │
    │                          │  ⠋ thinking…                 │
    ├──────────────────────────┴──────────────────────────────┤
    │ CONVERSATION [chat|edit|plan]                           │
    │  You:      research async python                        │
    │  BirdClaw: Starting now…                                │
    │  ⠋ BirdClaw is thinking…                                │
    │ > _                                                     │
    └─ Footer: contextual keybindings ───────────────────────┘

Keybindings (safe — verified against Textual defaults and openclaw):
    Ctrl+Q          quit
    Ctrl+C          3-state: clear input → warn → stop task  (openclaw pattern)
    Ctrl+D          exit (alternate)
    Ctrl+S          shell overlay
    Ctrl+O          toggle output raw/pretty  (output pane focused)
    Ctrl+T          toggle thinking display   (output pane focused)
    Ctrl+L          model selector
    Ctrl+G          agent selector
    Ctrl+P          command palette (Textual built-in)
    Ctrl+E          toggle edit mode          (conversation focused)
    F1 / F2         standing / active task tab
    [ / ]           resize conversation pane (quick height step)
    \\              layout picker — chat height / task width / arrangement
    /layout         same as \\ but via slash command
    Tab/Shift+Tab   Textual focus cycle (kept as-is)
    Escape          close overlay / exit edit mode
    ?               help overlay (when input empty)
    /command        slash commands
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.message import Message
from textual.widgets import (
    Footer, Header, Input, Label, ListItem, ListView, RichLog, Static, TextArea,
)
from textual.widget import Widget

from birdclaw.memory.tasks import task_registry
from birdclaw.tui.prefs import TuiPrefs
from birdclaw.tui.render import (
    SPINNER_FRAMES, render_pretty, render_raw, render_turn,
    spinner_done, spinner_failed, spinner_text, status_badge,
)
from birdclaw.tui.cards import ApprovalCard, PlanBanner, StageHeader, ToolCard
from birdclaw.tui.overlays import FuzzyPickerOverlay, HelpOverlay, SearchOverlay, ShellOverlay

_CONV_HEIGHT_PRESETS  = [20, 30, 40, 50, 60, 70]
_TASK_WIDTH_PRESETS   = [25, 30, 35, 40, 50]

_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"

# Line index where the BIRDCLAW text block starts (bird above, text below)
_SPLASH_SPLIT = 40


def _load_ascii_art() -> list[str]:
    import textwrap
    try:
        raw = (_ASSETS_DIR / "ascii_art").read_text(encoding="utf-8")
        return textwrap.dedent(raw).splitlines()
    except OSError:
        return []


_ASCII_ART_LINES: list[str] = _load_ascii_art()


# ---------------------------------------------------------------------------
# FlatBtn — single-row clickable label (replaces Button in toolbars)
# ---------------------------------------------------------------------------

class FlatBtn(Static):
    """Height-1 clickable button for toolbars and tab bars.

    Textual's Button widget requires ≥3 rows to render its label at
    height: 1 the label is clipped or invisible.  FlatBtn is a Static
    that posts FlatBtn.Pressed on click, so existing on_button_pressed
    handlers need only change to on_flat_btn_pressed.  CSS is applied
    without Textual's Button component-class overrides, so $primary-*
    and $text-* variables work as expected across all themes.
    """

    class Pressed(Message):
        def __init__(self, button: "FlatBtn") -> None:
            self.button = button
            super().__init__()

        @property
        def id(self) -> str | None:  # compat with Button.Pressed.button.id callers
            return self.button.id

    can_focus = False

    def on_click(self) -> None:
        self.post_message(FlatBtn.Pressed(self))
_POLL_INTERVAL        = 0.1   # fast poll → streaming feel
_SPINNER_INTERVAL     = 0.1
_OUTPUT_MAX_LINES     = 500
_MAX_EVENTS_PER_TICK  = 20    # process at most N new events per poll cycle
_BURST_THRESHOLD      = 30    # if backlog > this, burst-process up to 50 events


# ---------------------------------------------------------------------------
# Task list pane
# ---------------------------------------------------------------------------

class TaskListPane(Widget):
    """Left top pane — task list with live spinners, two tabs."""

    tab: reactive[str] = reactive("active")   # "active" | "standing"

    BINDINGS: ClassVar = [
        Binding("enter", "focus_output", "View output", show=True),
        Binding("f1", "switch_standing", "Standing tasks", show=True),
        Binding("f2", "switch_active",   "Active tasks",  show=True),
    ]

    DEFAULT_CSS = """
    TaskListPane {
        width: 40%;
        height: 100%;
        border: solid $primary-darken-2;
    }
    TaskListPane ListView {
        height: 1fr;
        background: transparent;
    }
    TaskListPane ListItem { padding: 0 1; }
    TaskListPane ListItem:hover { background: $primary-darken-3; }
    TaskListPane ListItem.-highlighted { background: $primary-darken-2; }
    TaskListPane #tab-bar {
        height: 1;
        layout: horizontal;
        background: $surface-darken-1;
    }
    TaskListPane .tab-btn {
        width: 1fr;
        height: 1;
        border: none;
        min-width: 0;
        padding: 0 1;
        background: transparent;
        color: $text-muted;
    }
    TaskListPane .tab-btn:hover {
        background: $primary-darken-3;
        color: $text;
    }
    TaskListPane .tab-btn.-active-tab {
        background: $primary-darken-2;
        color: $text;
    }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._tick:     int        = 0
        self._task_ids: list[str]  = []   # task_id by list position (no widget IDs needed)
        self._last_task_snapshot: list[tuple] = []  # for change detection
        self._approval_task_ids: set[str] = set()   # tasks with pending approval → flash ⚠
        self._label_cache: dict[int, str] = {}      # row_index → last rendered plain text
        # row_index → (indent_str, goal_text) for the currently-executing phase row
        self._active_phase_rows: dict[int, tuple[str, str]] = {}

    def mark_approval_pending(self, task_id: str) -> None:
        self._approval_task_ids.add(task_id)

    def clear_approval_pending(self, task_id: str) -> None:
        self._approval_task_ids.discard(task_id)

    def compose(self) -> ComposeResult:
        with Horizontal(id="tab-bar"):
            yield FlatBtn("Active ▶", id="tab-active-label",   classes="tab-btn -active-tab")
            yield FlatBtn("Standing", id="tab-standing-label",  classes="tab-btn")
        yield ListView(id="task-list")

    def on_flat_btn_pressed(self, event: FlatBtn.Pressed) -> None:
        if event.button.id == "tab-active-label":
            self.tab = "active"
        elif event.button.id == "tab-standing-label":
            self.tab = "standing"
        event.stop()

    def on_mount(self) -> None:
        self.border_title = "TASKS"
        self.refresh_tasks()
    
    def tick_spinner(self, tick: int) -> None:
        self._tick = tick
        lv: ListView = self.query_one("#task-list", ListView)
        flash_on = (tick // 4) % 2 == 0   # ⚠ flashes at ~2 Hz

        # Animate the currently-executing phase row (braille spinner)
        from birdclaw.tui.render import spinner_frame
        for row_idx, (indent, goal) in self._active_phase_rows.items():
            try:
                item = lv.children[row_idx]
                label = item.query_one(Label)
                pt = Text()
                pt.append(indent, style="dim")
                pt.append(spinner_frame(tick) + " ", style="yellow")
                pt.append(goal, style="yellow")
                key = str(pt)
                if self._label_cache.get(row_idx) != key:
                    self._label_cache[row_idx] = key
                    label.update(pt)
            except Exception:
                pass

        # Only update a label when its text actually changes — prevents flicker
        for i, task_id in enumerate(self._task_ids):
            if not task_id:
                continue   # phase row sentinel — no live update needed
            task = task_registry.get(task_id)
            if task is None:
                continue
            needs_approval = task_id in self._approval_task_ids
            # Spinner chars cycle; only running tasks + approval tasks need per-tick updates
            if task.status != "running" and not needs_approval:
                continue
            try:
                item = lv.children[i]
                label = item.query_one(Label)
                if needs_approval and flash_on:
                    badge = Text("⚠ ", style="bold yellow")
                else:
                    badge = status_badge(task.status, tick)
                # Use generated title (2-5 words) or fall back to truncated prompt
                display = (
                    task.title if getattr(task, "title", "")
                    else task.prompt[:34] + ("…" if len(task.prompt) > 34 else "")
                )
                # Phase counter and elapsed time for running tasks
                phase_str = ""
                elapsed_str = ""
                if task.status == "running":
                    import time as _ti
                    phases = getattr(task, "phases", [])
                    cur_idx = getattr(task, "current_phase_index", -1)
                    if phases and cur_idx >= 0:
                        phase_str = f" {cur_idx + 1}/{len(phases)}"
                    # Use started_at so the clock never resets mid-run
                    ref = task.started_at or task.updated_at
                    secs = int(_ti.time() - ref)
                    if secs >= 60:
                        elapsed_str = f" {secs // 60}m{secs % 60:02d}s"
                    elif secs >= 3:
                        elapsed_str = f" {secs}s"
                t = Text()
                t.append_text(badge)
                text_style = "bold yellow" if (needs_approval and flash_on) else "white"
                t.append(f" {display}", style=text_style)
                if phase_str:
                    t.append(phase_str, style="dim yellow")
                if elapsed_str:
                    t.append(elapsed_str, style="dim cyan")
                key = str(t)
                if self._label_cache.get(i) != key:
                    self._label_cache[i] = key
                    label.update(t)
            except Exception:
                pass

        # Full rebuild when task composition OR phase progress changes
        snapshot = [
            (t.task_id, t.status,
             getattr(t, "current_phase_index", -1),
             len(getattr(t, "phases", [])))
            for t in self._current_tasks()
        ]
        if snapshot != self._last_task_snapshot:
            self._last_task_snapshot = snapshot
            self._label_cache.clear()
            self.refresh_tasks()

    def _current_tasks(self):
        """Tasks to show in the current tab.

        Active tab: running/created tasks + recently finished (last 10 min) so
        the user can inspect output after completion.
        """
        import time as _t
        all_tasks = task_registry.list()
        now = _t.time()
        if self.tab == "standing":
            # Show skill tasks: running/created always; for completed/failed keep only the
            # most recent instance per skill (team_id) so old runs don't pile up.
            skill_tasks = [t for t in all_tasks if t.team_id and t.team_id.startswith("skill:")]
            active = [t for t in skill_tasks if t.status in ("running", "created", "waiting")]
            active_team_ids = {t.team_id for t in active}
            # For each skill not currently active, pick only the single most recent terminal task
            finished: dict[str, object] = {}
            for t in skill_tasks:
                if t.status not in ("running", "created", "waiting") and t.team_id not in active_team_ids:
                    prev = finished.get(t.team_id)
                    if prev is None or t.updated_at > prev.updated_at:  # type: ignore[union-attr]
                        finished[t.team_id] = t
            tasks = active + list(finished.values())
        else:
            tui_session_id = getattr(self.app, "tui_session_id", "")

            tasks = [
                t for t in all_tasks
                if not (t.team_id and t.team_id.startswith("skill:"))
                and (
                    t.status in ("running", "created", "waiting")
                    or (
                        t.status in ("completed", "failed", "stopped")
                        and tui_session_id
                        and t.session_id == tui_session_id
                    )
                )
            ]
        return tasks

    def refresh_tasks(self) -> None:
        lv: ListView = self.query_one("#task-list", ListView)
        prev_idx = lv.index or 0
        lv.clear()
        self._task_ids = []
        self._active_phase_rows = {}

        tasks = self._current_tasks()

        if self.tab == "standing":
            # Standing tab: skill tasks + cron schedule entries
            self._render_standing(lv, tasks)
        else:
            # Active tab: root tasks first, then subtasks indented underneath
            self._render_active_tree(lv, tasks)

        if self._task_ids and prev_idx < len(self._task_ids):
            lv.index = prev_idx

        self._update_tab_counts()

    def _render_active_tree(self, lv: ListView, tasks: list) -> None:
        """Render active tasks as a tree.

        Layout per root task:
          [badge] Task Title          ← 2-5 word title
            └ ✓ Phase one goal        ← completed phases
            └ ⠼ Phase two goal        ← current phase (running)
            └ ○ Phase three goal      ← pending phases
          [badge] Follow-up Title     ← subtask (follow-up message)
            └ ...
        """
        task_id_set = {t.task_id for t in tasks}
        children_by_parent: dict[str, list] = {}
        roots = []
        for t in tasks:
            parent = t.team_id if t.team_id and t.team_id in task_id_set else None
            if parent:
                children_by_parent.setdefault(parent, []).append(t)
            else:
                roots.append(t)

        for task in roots:
            self._task_ids.append(task.task_id)
            badge = status_badge(task.status, self._tick)
            # Use 2-5 word title if available, fall back to truncated prompt
            display_name = task.title if getattr(task, "title", "") else task.prompt[:32]
            t = Text()
            t.append_text(badge)
            t.append(f" {display_name}", style="white" if task.status == "running" else "dim white")
            # Phase counter (e.g. "2/4") for running tasks
            phases = getattr(task, "phases", [])
            cur_idx = getattr(task, "current_phase_index", -1)
            if task.status == "running" and phases and cur_idx >= 0:
                t.append(f" {cur_idx + 1}/{len(phases)}", style="dim yellow")
            lv.append(ListItem(Label(t)))

            # Phase tree — show plan phases as children (only when phases are known)
            phases = getattr(task, "phases", [])
            current_idx = getattr(task, "current_phase_index", -1)
            if phases:
                for i, phase_goal in enumerate(phases):
                    row_idx = len(self._task_ids)
                    pt = Text()
                    pt.append("    └ ", style="dim")
                    if i < current_idx or (i == current_idx and task.status in ("completed", "failed", "stopped")):
                        if task.status == "failed" and i == current_idx:
                            pt.append("✘ ", style="bold red")
                            pt.append(phase_goal[:40], style="dim red")
                        elif task.status == "stopped" and i == current_idx:
                            pt.append("⏹ ", style="bold yellow")
                            pt.append(phase_goal[:40], style="dim yellow")
                        else:
                            pt.append("✓ ", style="green")
                            pt.append(phase_goal[:40], style="dim green")
                    elif i == current_idx:
                        pt.append("⠼ ", style="yellow")
                        pt.append(phase_goal[:40], style="yellow")
                        self._active_phase_rows[row_idx] = ("    └ ", phase_goal[:40])
                    else:
                        pt.append("○ ", style="dim")
                        pt.append(phase_goal[:40], style="dim")
                    lv.append(ListItem(Label(pt)))
                    # Phase rows are visual-only — sentinel keeps _task_ids in sync with lv rows
                    self._task_ids.append("")

            # Subtasks (follow-up messages) indented after phases
            for sub in children_by_parent.get(task.task_id, []):
                self._task_ids.append(sub.task_id)
                sbadge = status_badge(sub.status, self._tick)
                sub_name = sub.title if getattr(sub, "title", "") else sub.prompt[:28]
                st = Text()
                st.append("  └ ", style="dim")
                st.append_text(sbadge)
                st.append(f" {sub_name}", style="white" if sub.status == "running" else "dim white")
                sub_phases_preview = getattr(sub, "phases", [])
                sub_cur = getattr(sub, "current_phase_index", -1)
                if sub.status == "running" and sub_phases_preview and sub_cur >= 0:
                    st.append(f" {sub_cur + 1}/{len(sub_phases_preview)}", style="dim yellow")
                lv.append(ListItem(Label(st)))
                # Show phases for subtasks too
                sub_phases = getattr(sub, "phases", [])
                sub_idx = getattr(sub, "current_phase_index", -1)
                for j, sp_goal in enumerate(sub_phases):
                    sub_row_idx = len(self._task_ids)
                    spt = Text()
                    spt.append("      └ ", style="dim")
                    if j < sub_idx or (j == sub_idx and sub.status in ("completed", "failed", "stopped")):
                        if sub.status == "failed" and j == sub_idx:
                            spt.append("✘ ", style="bold red")
                            spt.append(sp_goal[:36], style="dim red")
                        elif sub.status == "stopped" and j == sub_idx:
                            spt.append("⏹ ", style="bold yellow")
                            spt.append(sp_goal[:36], style="dim yellow")
                        else:
                            spt.append("✓ ", style="green")
                            spt.append(sp_goal[:36], style="dim green")
                    elif j == sub_idx:
                        spt.append("⠼ ", style="yellow")
                        spt.append(sp_goal[:36], style="yellow")
                        self._active_phase_rows[sub_row_idx] = ("      └ ", sp_goal[:36])
                    else:
                        spt.append("○ ", style="dim")
                        spt.append(sp_goal[:36], style="dim")
                    lv.append(ListItem(Label(spt)))
                    self._task_ids.append("")  # sentinel for subtask phase rows

                # Grandchild tasks — same indent as children, ↳ marker to distinguish
                for gsub in children_by_parent.get(sub.task_id, []):
                    self._task_ids.append(gsub.task_id)
                    gbadge = status_badge(gsub.status, self._tick)
                    gsub_name = gsub.title if getattr(gsub, "title", "") else gsub.prompt[:28]
                    gt = Text()
                    gt.append("  ↳ ", style="dim")
                    gt.append_text(gbadge)
                    gt.append(f" {gsub_name}", style="white" if gsub.status == "running" else "dim white")
                    gsub_cur = getattr(gsub, "current_phase_index", -1)
                    gsub_phases = getattr(gsub, "phases", [])
                    if gsub.status == "running" and gsub_phases and gsub_cur >= 0:
                        gt.append(f" {gsub_cur + 1}/{len(gsub_phases)}", style="dim yellow")
                    lv.append(ListItem(Label(gt)))

    def _render_standing(self, lv: ListView, tasks: list) -> None:
        """Render standing tab: one row per skill combining schedule + last run status.

        Cron-scheduled skills always appear (with next-run time).  Ad-hoc skill
        tasks that have no cron entry are shown below.  Old failed runs auto-clear
        because _current_tasks() already keeps only the most recent per skill.
        """
        import time as _t
        now = _t.time()

        # Build lookup: skill_name → most-recent task
        task_by_skill: dict[str, object] = {}
        for task in tasks:
            skill = (task.team_id or "").replace("skill:", "")
            prev = task_by_skill.get(skill)
            if prev is None or task.updated_at > prev.updated_at:  # type: ignore[union-attr]
                task_by_skill[skill] = task

        seen_skills: set[str] = set()

        # ── Scheduled skills (cron entries) ──────────────────────────────────
        try:
            from birdclaw.skills.cron import cron_service
            for entry in cron_service.list():
                skill = entry.skill_name
                seen_skills.add(skill)
                task = task_by_skill.get(skill)

                row = Text()
                row.append("⏰ ", style="bold cyan")
                row.append(f"{skill[:18]:<18}", style="cyan bold")
                row.append(f"  {entry.schedule}", style="dim")
                if entry.next_run_at:
                    mins = int((entry.next_run_at - now) / 60)
                    row.append(f"  in {mins}m" if mins >= 0 else "  due", style="dim green")

                # Last-run status appended inline
                if task is not None:
                    status = task.status  # type: ignore[union-attr]
                    if status == "running":
                        row.append("  ⠼", style="yellow")
                    elif status in ("created", "waiting"):
                        row.append("  queued", style="dim yellow")
                    elif status == "completed":
                        ago = max(0, int((now - task.updated_at) / 60))  # type: ignore[union-attr]
                        row.append(f"  ✓ {ago}m ago", style="green")
                    else:  # failed / stopped
                        ago = max(0, int((now - task.updated_at) / 60))  # type: ignore[union-attr]
                        row.append(f"  ✘ {ago}m ago", style="bold red")
                    self._task_ids.append(task.task_id)  # type: ignore[union-attr]
                else:
                    row.append("  never run", style="dim")
                    self._task_ids.append("")

                lv.append(ListItem(Label(row)))
        except Exception:
            pass

        # ── Ad-hoc skill tasks not in any cron schedule ───────────────────────
        for task in tasks:
            skill = (task.team_id or "").replace("skill:", "")
            if skill in seen_skills:
                continue
            badge = status_badge(task.status, self._tick)
            prompt = task.prompt[:28] + ("…" if len(task.prompt) > 28 else "")
            row = Text()
            row.append_text(badge)
            row.append(f" [{skill}]", style="dim cyan")
            row.append(f" {prompt}", style="white" if task.status == "running" else "dim white")
            lv.append(ListItem(Label(row)))
            self._task_ids.append(task.task_id)

    def _selected_task_id(self) -> str:
        lv: ListView = self.query_one("#task-list", ListView)
        idx = lv.index
        if idx is not None and 0 <= idx < len(self._task_ids):
            return self._task_ids[idx]
        return ""

    def _update_tab_counts(self) -> None:
        """Refresh tab labels with live task counts."""
        try:
            all_tasks = task_registry.list()
            n_active = sum(
                1 for t in all_tasks
                if not (t.team_id and t.team_id.startswith("skill:"))
                and t.status in ("running", "created")
            )
            # Count unique skills (not raw task instances)
            _skill_ids = {t.team_id for t in all_tasks if t.team_id and t.team_id.startswith("skill:")}
            try:
                from birdclaw.skills.cron import cron_service as _cs
                _skill_ids |= {f"skill:{e.skill_name}" for e in _cs.list()}
            except Exception:
                pass
            n_standing = len(_skill_ids)
            a: FlatBtn = self.query_one("#tab-active-label",   FlatBtn)
            s: FlatBtn = self.query_one("#tab-standing-label", FlatBtn)
            a_text = a.renderable if hasattr(a, "renderable") else ""
            s_text = s.renderable if hasattr(s, "renderable") else ""
            a.update(f"Active({n_active}) ▶" if self.tab == "active" else f"Active({n_active})")
            s.update(f"Standing({n_standing}) ▶" if self.tab == "standing" else f"Standing({n_standing})")
        except NoMatches:
            pass

    def watch_tab(self, value: str) -> None:
        try:
            a: FlatBtn = self.query_one("#tab-active-label",   FlatBtn)
            s: FlatBtn = self.query_one("#tab-standing-label", FlatBtn)
            if value == "active":
                a.add_class("-active-tab");    s.remove_class("-active-tab")
                self.border_title = "TASKS — Active"
            else:
                s.add_class("-active-tab");    a.remove_class("-active-tab")
                self.border_title = "TASKS — Standing"
        except NoMatches:
            pass
        self.refresh_tasks()

    def action_switch_active(self)   -> None: self.tab = "active"
    def action_switch_standing(self) -> None: self.tab = "standing"

    def action_focus_output(self) -> None:
        task_id = self._selected_task_id()
        if task_id:   # "" = cron entry row, skip
            try:
                self.app.query_one("#output-pane", OutputPane).focused_task_id = task_id
                self.app.query_one("#output-pane").focus()
            except NoMatches:
                pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.action_focus_output()


# ---------------------------------------------------------------------------
# Assistant message widget — Markdown-rendered reply block
# ---------------------------------------------------------------------------

class _AssistantMessage(Widget):
    """Renders a BirdClaw assistant reply with full Rich Markdown support."""

    DEFAULT_CSS = """
    _AssistantMessage {
        width: 100%;
        height: auto;
        padding: 0 1 1 1;
        border-left: solid $success-darken-2;
        margin-top: 1;
    }
    _AssistantMessage #am-label {
        height: 1;
        color: $success;
    }
    _AssistantMessage #am-body {
        height: auto;
        background: transparent;
    }
    """

    def __init__(self, content: str, **kw) -> None:
        super().__init__(**kw)
        self._content = content

    def compose(self) -> ComposeResult:
        yield Static(Text("BirdClaw", style="bold green"), id="am-label")
        yield RichLog(id="am-body", markup=False, highlight=False,
                      max_lines=300, wrap=True)

    def on_mount(self) -> None:
        # Defer write until after the first refresh — compose() children may not
        # be in the DOM yet when on_mount fires for dynamically-mounted widgets.
        self.call_after_refresh(self._write_content)

    def _write_content(self) -> None:
        from rich.markdown import Markdown
        try:
            log = self.query_one("#am-body", RichLog)
        except Exception:
            return
        if self._content.strip():
            log.write(Markdown(self._content))
        else:
            log.write(Text("(no content)", style="dim"))


# ---------------------------------------------------------------------------
# ASCII splash — animated bird shown in empty output pane
# ---------------------------------------------------------------------------

class AsciiSplashWidget(Static):
    """Animated ASCII art shown when no task is selected.

    Bird section (lines 0-39) scrolls slowly upward; BIRDCLAW text (40+)
    stays fixed at the bottom.  Rich Text objects are used so the art
    characters are never misinterpreted as markup.
    """

    DEFAULT_CSS = """
    AsciiSplashWidget {
        height: 1fr;
        overflow: hidden;
        color: $text-muted;
    }
    """

    # Colour codes for the three bird gradient zones and the text block
    _C = [
        "dim cyan",       # upper bird
        "cyan",           # mid bird
        "bold cyan",      # lower bird
        "bold bright_cyan",  # BIRDCLAW text
    ]

    def on_mount(self) -> None:
        self._tick = 0
        self._bird_offset = 0
        self._update_content()
        self.set_interval(0.15, self._on_tick)

    def _on_tick(self) -> None:
        self._tick += 1
        if self._tick % 12 == 0:
            bird_lines = _ASCII_ART_LINES[:_SPLASH_SPLIT]
            if bird_lines:
                self._bird_offset = (self._bird_offset + 1) % len(bird_lines)
                self._update_content()

    def _update_content(self) -> None:
        from rich.text import Text as _Text
        if not _ASCII_ART_LINES:
            self.update(_Text("🐦 BirdClaw"))
            return

        bird  = _ASCII_ART_LINES[:_SPLASH_SPLIT]
        text  = _ASCII_ART_LINES[_SPLASH_SPLIT:]
        n     = len(bird)

        # Rotate bird lines for scroll effect
        rotated = bird[self._bird_offset:] + bird[:self._bird_offset]

        out = _Text(no_wrap=False)
        for i, ln in enumerate(rotated):
            zone = 0 if i < n // 3 else (1 if i < 2 * n // 3 else 2)
            out.append(ln + "\n", style=self._C[zone])
        for ln in text:
            out.append(ln + "\n", style=self._C[3])

        self.update(out)


# ---------------------------------------------------------------------------
# Output pane — card-based widget tree (replaces flat RichLog)
# ---------------------------------------------------------------------------

class OutputPane(Widget):
    """Right top pane — structured card tree for the focused task.

    Each session_log event is mapped to a live widget:
      plan          → PlanBanner
      stage_start   → StageHeader
      stage_done    → StageHeader.mark_done()
      tool_call     → ToolCard (running state)
      tool_result   → ToolCard.complete()
      assistant_msg → Static text line
      user_msg      → Static text line
      approval      → ApprovalCard (from gateway, not session log)

    Ctrl+O toggles a raw-mode fallback (flat RichLog view — existing behaviour).
    Ctrl+T shows/hides think tool cards.
    """

    raw_mode:        reactive[bool] = reactive(False)
    show_thinking:   reactive[bool] = reactive(True)
    focused_task_id: reactive[str]  = reactive("")
    verbose_mode:    reactive[str]  = reactive("full")   # "off" | "on" | "full"

    BINDINGS: ClassVar = [
        Binding("ctrl+o", "toggle_raw",      "Raw/pretty", show=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking",   show=True),
    ]

    DEFAULT_CSS = """
    OutputPane {
        width: 1fr;
        height: 100%;
        border: solid $primary-darken-2;
        layout: vertical;
    }
    OutputPane #output-toolbar {
        height: 1;
        layout: horizontal;
        background: $surface-darken-1;
    }
    OutputPane .out-btn {
        height: 1;
        border: none;
        min-width: 0;
        padding: 0 1;
        background: transparent;
        color: $text-muted;
    }
    OutputPane .out-btn:hover { background: $primary-darken-3; color: $text; }
    OutputPane .out-btn.-on    { background: $primary-darken-2; color: $text; }
    OutputPane #output-scroll {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
    }
    OutputPane #output-log {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
        display: none;
    }
    OutputPane #output-empty {
        height: 1fr;
        overflow: hidden;
        color: $text-muted;
    }
    OutputPane .msg-user {
        width: 100%;
        height: auto;
        padding: 0 1 1 1;
        border-left: solid $primary-darken-1;
        margin-top: 1;
        color: cyan;
    }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        # Session log tracking
        self._event_cursor:    int  = 0
        self._last_task_id:    str  = ""
        # Card references for in-place update
        self._pending_card:    ToolCard | None       = None   # last tool_call, awaiting result
        self._last_stage:      StageHeader | None    = None   # for mark_done()
        self._approval_card:   ApprovalCard | None   = None
        self._stage_headers:   list[StageHeader]     = []
        self._all_tool_cards:  list[ToolCard]        = []
        # Status
        self._stage_tick:      int  = 0
        self._current_stage:   str  = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="output-toolbar"):
            yield FlatBtn("Cards",   id="out-btn-cards",    classes="out-btn -on")
            yield FlatBtn("Raw",     id="out-btn-raw",      classes="out-btn")
            yield FlatBtn("Think",   id="out-btn-think",    classes="out-btn -on")
            yield FlatBtn("Verbose", id="out-btn-verbose",  classes="out-btn -on")
            yield FlatBtn("Compact", id="out-btn-compact",  classes="out-btn")
        yield AsciiSplashWidget(id="output-empty")
        yield VerticalScroll(id="output-scroll")
        yield RichLog(id="output-log", highlight=False, markup=False,
                      max_lines=_OUTPUT_MAX_LINES, wrap=True)

    def on_flat_btn_pressed(self, event: FlatBtn.Pressed) -> None:
        bid = event.button.id
        if bid == "out-btn-cards":
            self.raw_mode = False
            self._sync_toolbar_buttons()
        elif bid == "out-btn-raw":
            self.raw_mode = True
            self._sync_toolbar_buttons()
        elif bid == "out-btn-think":
            self.show_thinking = not self.show_thinking
            self._rebuild()
            self._sync_toolbar_buttons()
        elif bid == "out-btn-verbose":
            self.verbose_mode = "full"
            self._rebuild()
            self._sync_toolbar_buttons()
        elif bid == "out-btn-compact":
            self.verbose_mode = "on"
            self._rebuild()
            self._sync_toolbar_buttons()
        else:
            return
        event.stop()

    def _sync_toolbar_buttons(self) -> None:
        """Keep toolbar button -on classes in sync with current state."""
        try:
            self.query_one("#out-btn-cards",   FlatBtn).set_class(not self.raw_mode,      "-on")
            self.query_one("#out-btn-raw",     FlatBtn).set_class(self.raw_mode,           "-on")
            self.query_one("#out-btn-think",   FlatBtn).set_class(self.show_thinking,      "-on")
            self.query_one("#out-btn-verbose", FlatBtn).set_class(self.verbose_mode == "full", "-on")
            self.query_one("#out-btn-compact", FlatBtn).set_class(self.verbose_mode == "on",   "-on")
        except NoMatches:
            pass

    def on_mount(self) -> None:
        self._update_title()

    # ── Title ─────────────────────────────────────────────────────────────────

    def _update_title(self) -> None:
        if self.focused_task_id:
            task = task_registry.get(self.focused_task_id)
            label = (getattr(task, "title", "") or self.focused_task_id[:12]) if task else self.focused_task_id[:12]
        else:
            label = "none"
        self.border_title = f"OUTPUT  {label}"

    # ── Reactive watchers ─────────────────────────────────────────────────────

    def watch_focused_task_id(self, _: str) -> None:
        self._update_title()
        self._rebuild()

    def watch_raw_mode(self, raw: bool) -> None:
        self._update_title()
        try:
            self.query_one("#output-scroll").display = not raw
            self.query_one("#output-log").display    = raw
        except NoMatches:
            pass
        if raw:
            self._reload_raw()
        else:
            self._rebuild()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_toggle_raw(self)      -> None: self.raw_mode      = not self.raw_mode
    def action_toggle_thinking(self) -> None: self.show_thinking = not self.show_thinking; self._rebuild()

    # ── Tick (called from BirdClawApp._tick_spinners) ─────────────────────────

    def tick_spinner(self, tick: int) -> None:
        self._stage_tick = tick
        for sh in self._stage_headers:
            sh.tick(tick)
        if self._pending_card:
            self._pending_card.tick(tick)
        if self._approval_card:
            try:
                self._approval_card.tick(tick)
            except Exception:
                self._approval_card = None

    # ── Rebuild / refresh ─────────────────────────────────────────────────────

    def refresh_output(self) -> None:
        """Called every 0.5 s by BirdClawApp._poll_tasks."""
        if not self.focused_task_id:
            return
        if self.focused_task_id != self._last_task_id:
            self._last_task_id = self.focused_task_id
            self._rebuild()
            return
        self._process_new_events()

    def _rebuild(self) -> None:
        """Clear the scroll container and replay all events from scratch."""
        self._event_cursor  = 0
        self._pending_card  = None
        self._last_stage    = None
        self._approval_card = None
        self._stage_headers = []
        self._all_tool_cards = []
        self._current_stage  = ""

        # Clear scroll container
        try:
            scroll = self.query_one("#output-scroll", VerticalScroll)
            for child in list(scroll.children):
                child.remove()
        except NoMatches:
            pass

        # Hide/show empty placeholder
        has_task = bool(self.focused_task_id)
        try:
            self.query_one("#output-empty").display = not has_task
            self.query_one("#output-scroll").display = has_task and not self.raw_mode
        except NoMatches:
            pass

        if has_task:
            self._process_new_events()
            self._check_approval()

    def _process_new_events(self) -> None:
        import json as _json
        path = self._session_log_path()
        if path is None:
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        new_lines = lines[self._event_cursor:]
        # Burst mode: if backlog is large, process more events per tick so the
        # output pane catches up quickly (e.g. after task completion flood).
        limit = 50 if len(new_lines) > _BURST_THRESHOLD else _MAX_EVENTS_PER_TICK
        batch = new_lines[:limit]
        self._event_cursor += len(batch)
        had_new = False
        for raw in batch:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            self._dispatch_event(record)
            had_new = True
        if had_new:
            try:
                scroll = self.query_one("#output-scroll", VerticalScroll)
                # Autoscroll unless user has scrolled up (more than 3 lines from bottom)
                if scroll.scroll_y >= scroll.max_scroll_y - 3:
                    self.call_after_refresh(lambda: scroll.scroll_end(animate=False))
            except NoMatches:
                pass

    def _dispatch_event(self, record: dict) -> None:
        rtype = record.get("type", "")
        data  = record.get("data", {})
        try:
            scroll = self.query_one("#output-scroll", VerticalScroll)
        except NoMatches:
            return

        if rtype == "user_message":
            content = data.get("content", "")
            t = Text()
            t.append("Task  ", style="bold cyan")
            t.append(content, style="cyan")
            scroll.mount(Static(t, classes="msg-user", markup=False))

        elif rtype == "plan":
            outcome = data.get("outcome", "")
            steps   = data.get("steps", [])
            logger.info("[tui] plan  outcome=%r  steps=%d", outcome[:60], len(steps))
            banner = PlanBanner(outcome, steps)
            scroll.mount(banner)

        elif rtype == "stage_start":
            stage_type = data.get("stage_type", "")
            goal       = data.get("goal", "")
            self._current_stage = stage_type
            logger.info("[tui] stage_start  type=%s  goal=%r", stage_type, goal[:60])
            sh = StageHeader(
                stage_type=stage_type,
                goal=goal,
                index=len(self._stage_headers),
            )
            self._stage_headers.append(sh)
            self._last_stage = sh
            scroll.mount(sh)
            self.app.update_status(session=self.focused_task_id, stage=stage_type)

        elif rtype == "stage_done":
            self._current_stage = ""
            dur = data.get("duration_ms", 0)
            ok  = data.get("ok", True)
            logger.info("[tui] stage_done  dur_ms=%d  ok=%s", dur, ok)
            if self._last_stage:
                self._last_stage.mark_done(dur, ok=ok)
            self.app.update_status(stage="")

        elif rtype == "tool_call":
            name = data.get("name", "")
            args = data.get("arguments", {})
            logger.debug("[tui] tool_call  name=%s  args=%s", name, str(args)[:80])
            if not self.show_thinking and name == "think":
                return
            if self.verbose_mode == "off":
                return   # suppress all tool cards
            card = ToolCard(name=name, args=args,
                            body_visible=self.verbose_mode == "full")
            self._pending_card = card
            self._all_tool_cards.append(card)
            scroll.mount(card)
            self.call_after_refresh(lambda: scroll.scroll_end(animate=False))

        elif rtype == "tool_result":
            result = data.get("result", "")
            dur    = data.get("duration_ms", 0)
            logger.debug("[tui] tool_result  dur_ms=%d  len=%d", dur, len(result))
            if self._pending_card:
                self._pending_card.complete(result, dur)
                self._pending_card = None

        elif rtype == "assistant_message":
            content = data.get("content", "")
            logger.info("[tui] assistant_message  chars=%d  preview=%r", len(content), content[:80])
            scroll.mount(_AssistantMessage(content))
            self._current_stage = ""

        elif rtype == "usage":
            tokens = data.get("total_tokens", 0)
            model  = data.get("model", "")
            logger.debug("[tui] usage  tokens=%d  model=%s", tokens, model)
            self.app.update_status(tokens=tokens, model=model or None)

    # ── Approval card ─────────────────────────────────────────────────────────

    def _check_approval(self) -> None:
        """Mount ApprovalCard if focused task has a pending approval request."""
        if not self.focused_task_id:
            return
        from birdclaw.agent.approvals import approval_queue
        for req in approval_queue.list_pending():
            if req.task_id == self.focused_task_id:
                self._show_approval(req)
                return

    def finalize_pending(self, task_id: str = "") -> None:
        """Force-complete any still-spinning cards/stages (called when task ends).

        Only acts when the output pane is showing the task that completed.
        """
        if task_id and self.focused_task_id != task_id:
            return
        if self._pending_card:
            self._pending_card.complete("", 0)
            self._pending_card = None
        if self._last_stage and not self._last_stage._done:
            self._last_stage.mark_done(0)

    def show_approval(self, req: object) -> None:
        """Called from BirdClawApp when gateway pushes an approval_request."""
        if self.focused_task_id == getattr(req, "task_id", ""):
            self._show_approval(req)

    def _show_approval(self, req: object) -> None:
        if self._approval_card:
            return   # already showing one
        card = ApprovalCard(
            approval_id=getattr(req, "approval_id", ""),
            tool_name=getattr(req, "tool_name", ""),
            description=getattr(req, "description", ""),
            expires_at=getattr(req, "expires_at", 0.0),
        )
        self._approval_card = card
        try:
            scroll = self.query_one("#output-scroll", VerticalScroll)
            scroll.mount(card)
            scroll.scroll_end(animate=False)
        except NoMatches:
            pass

    # ── Raw mode fallback ─────────────────────────────────────────────────────

    def _reload_raw(self) -> None:
        import json as _json
        try:
            log = self.query_one("#output-log", RichLog)
            log.clear()
        except NoMatches:
            return
        path = self._session_log_path()
        if not path:
            return
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                rtype = record.get("type", "")
                if not self.show_thinking and rtype == "tool_call":
                    if record.get("data", {}).get("name") == "think":
                        continue
                rendered = render_pretty(record)
                if rendered:
                    log.write(rendered)
        except OSError:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _session_log_path(self) -> Path | None:
        if not self.focused_task_id:
            return None
        from birdclaw.config import settings
        sessions_dir = settings.data_dir / "sessions"
        # Exact match first (non-decomposed task)
        exact = sessions_dir / f"{self.focused_task_id}.jsonl"
        if exact.exists():
            return exact
        # Decomposed tasks write {task_id}_{step_id}.jsonl — pick most recently modified
        candidates = list(sessions_dir.glob(f"{self.focused_task_id}_*.jsonl"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)



# ---------------------------------------------------------------------------
# Buddy panel — GIF/image companion (right side of conversation pane)
# ---------------------------------------------------------------------------


# Fallback ASCII bird — used only when no GIF is present.
_BUDDY_ASCII: list[str] = [
    "  __  \n >('.')\n  /|  \n  \" \"  ",
    "  __  \n >(^.^)\n  /|  \n  \" \"  ",
    "  __  \n >(-.-)>\n   |  \n  \" \"  ",
    "  __  \n >('.') \n   |\\  \n  \" \"  ",
]

# Render width in terminal columns (panel is 22 wide; 2 border + 2 pad = 18 usable)
_GIF_COLS = 18
# Render height in terminal rows (each row = 2 image pixels via half-block ▄)
_GIF_ROWS = 7   # = 14 image pixels tall


def _q(v: int) -> int:
    """Quantize a colour channel to 32 levels (step=8).

    Reduces the number of unique ANSI truecolor escape sequences from
    16 million combinations down to ~32 000.  This prevents long runs of
    unique sequences from corrupting terminal multiplexer state (tmux pane
    bleed) while remaining visually indistinguishable at small render sizes.
    """
    return (v >> 3) << 3   # round down to nearest multiple of 8


def _render_pil_frame(
    frame, cols: int = _GIF_COLS, rows: int = _GIF_ROWS, contain: bool = False
) -> Text:
    """Convert a PIL Image frame to a Rich Text using half-block Unicode art.

    Each terminal row encodes two pixel rows:
      foreground = lower pixel colour,  background = upper pixel colour.

    contain=False (default): cover — scale to fill height, centre-crop width.
    contain=True:            contain — scale to fit within (cols × rows*2) preserving
                             aspect ratio; no cropping. Actual render size may be
                             smaller than requested if the aspect ratio demands it.
    """
    from PIL import Image as _PILImage
    from rich.style import Style

    img = frame.convert("RGB")
    src_w, src_h = img.size

    if contain:
        # Scale to fit within the available area without cropping.
        pixel_h = rows * 2
        scale = min(cols / src_w, pixel_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(2, int(src_h * scale))
        if new_h % 2:
            new_h -= 1   # must be even for half-block pairing
        img = img.resize((new_w, new_h), _PILImage.LANCZOS)
        render_cols = new_w
        render_rows = new_h // 2
    else:
        # Cover: scale so height fills rows*2, centre-crop to cols width.
        target_h = rows * 2
        scale_w  = int(src_w * target_h / src_h)
        img = img.resize((scale_w, target_h), _PILImage.LANCZOS)
        if scale_w > cols:
            x0  = (scale_w - cols) // 2
            img = img.crop((x0, 0, x0 + cols, target_h))
        elif scale_w < cols:
            img = img.resize((cols, target_h), _PILImage.LANCZOS)
        render_cols = cols
        render_rows = rows

    pixels = img.tobytes()
    stride = render_cols * 3
    _style_cache: dict[tuple, Style] = {}

    # In contain mode, horizontally centre the image if it's narrower than the panel.
    pad_left = ((cols - render_cols) // 2) if contain and render_cols < cols else 0

    result = Text(no_wrap=True)
    for y in range(0, render_rows * 2, 2):
        row_upper = y * stride
        row_lower = (y + 1) * stride
        if pad_left:
            result.append(" " * pad_left)
        for x in range(render_cols):
            ox = x * 3
            ur = _q(pixels[row_upper + ox]);     ug = _q(pixels[row_upper + ox + 1]); ub = _q(pixels[row_upper + ox + 2])
            lr = _q(pixels[row_lower + ox]);     lg = _q(pixels[row_lower + ox + 1]); lb = _q(pixels[row_lower + ox + 2])
            key = (ur, ug, ub, lr, lg, lb)
            style = _style_cache.get(key)
            if style is None:
                style = Style(color=f"rgb({lr},{lg},{lb})",
                              bgcolor=f"rgb({ur},{ug},{ub})")
                _style_cache[key] = style
            result.append("▄", style=style)
        result.append("\n")
    return result


class BuddyPanel(Widget):
    """GIF player companion panel on the right edge of the conversation pane.

    GIF playback
    ────────────
    Drop any *.gif file in ~/.birdclaw/buddy/ and it will be played back as
    half-block pixel art (2 image pixels per terminal row, full RGB color via
    truecolor escape codes).  Frame timing follows the GIF's own delay metadata.

    Multiple GIFs are played in alphabetical order, looping continuously.
    Falls back to built-in ASCII bird animation when no GIF is found.

    Dismiss / restore
    ─────────────────
    Click [-] to hide.  Click [+] to toggle compact/full display.
    Type /buddy to toggle visibility back on.
    """

    class ZoomToggle(Message):
        """Posted when the user clicks the [+] zoom button."""

    DEFAULT_CSS = """
    BuddyPanel {
        width: 44;
        height: 100%;
        background: $surface-darken-2;
        border: solid $primary-darken-2;
        padding: 0;
        margin: 0;
        layout: vertical;
    }
    BuddyPanel.buddy-full {
        width: 80;
    }
    BuddyPanel #buddy-art {
        width: 100%;
        height: 1fr;
        padding: 0;
        margin: 0;
        overflow-x: hidden;
    }
    BuddyPanel #buddy-controls {
        width: 100%;
        height: 1;
        background: $surface-darken-2;
        layout: horizontal;
    }
    BuddyPanel #buddy-ctrl-spacer {
        width: 1fr;
        height: 1;
        background: transparent;
    }
    BuddyPanel #buddy-zoom {
        width: 5;
        min-width: 5;
        height: 1;
        border: none;
        background: transparent;
        color: $primary;
        padding: 0;
        text-align: left;
    }
    BuddyPanel #buddy-dismiss {
        width: 3;
        min-width: 3;
        height: 1;
        border: none;
        background: transparent;
        color: $error;
        padding: 0;
        margin-right: 1;
        text-align: right;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="buddy-controls"):
            yield FlatBtn("[+]", id="buddy-zoom")
            yield Static("", id="buddy-ctrl-spacer")
            yield FlatBtn("✕", id="buddy-dismiss")
        yield Static("", id="buddy-art")

    def on_mount(self) -> None:
        self.border_title = "BUDDY"
        self._tick_count    = 0
        # GIF state
        self._gif_frames:    list      = []   # list of PIL Image objects
        self._gif_durations: list[int] = []   # ms per frame
        self._gif_frame_idx: int       = 0
        self._gif_elapsed:   int       = 0    # ms since last frame advance
        self._has_gif:       bool      = False
        # ASCII fallback state
        self._ascii_idx:     int       = 0
        # Render dimensions — updated on first on_resize
        self._render_cols:   int       = _GIF_COLS
        self._render_rows:   int       = _GIF_ROWS
        self._zoom_mode:     bool      = False   # True → contain scaling, False → cover
        # Try to load GIFs; fall back to ASCII
        self._try_load_gifs()
        self._render_current()

    def on_resize(self, event) -> None:   # noqa: ARG002
        """Schedule a re-render 120 ms after the resize event.

        A short timer (rather than call_after_refresh) lets Textual fully
        repaint the screen before we read art widget dimensions and render
        new pixel art — this eliminates the half-block artefacts that appear
        when the old frame overlaps a freshly-expanded area.
        """
        self.set_timer(0.12, self._rerender_after_settle)

    def _rerender_after_settle(self) -> None:
        """Clear art and re-render at the art widget's current size."""
        try:
            art = self.query_one("#buddy-art", Static)
        except NoMatches:
            return
        cols = max(4, art.content_size.width)
        rows = max(2, art.content_size.height)
        self._render_rows = rows
        self._render_cols = cols
        # Blank the art area first so old pixels don't bleed into the new frame.
        art.update("")
        self._render_current()

    def force_rerender(self) -> None:
        """Explicitly re-render after an external layout/theme change."""
        self.set_timer(0.12, self._rerender_after_settle)

    # ── GIF loading ───────────────────────────────────────────────────────────

    def _try_load_gifs(self) -> None:
        """Scan ~/.birdclaw/buddy/ for *.gif files and load all frames.

        Composites each GIF frame onto the accumulated canvas so delta-frames
        (partial updates) display correctly instead of showing blank regions.
        """
        try:
            from PIL import Image as _PILImage
            from birdclaw.config import settings
            buddy_dir = settings.data_dir / "buddy"
            if not buddy_dir.exists():
                return
            for gif_path in sorted(buddy_dir.glob("*.gif")):
                try:
                    gif = _PILImage.open(gif_path)
                    canvas = _PILImage.new("RGBA", gif.size, (18, 18, 18, 255))
                    n = getattr(gif, "n_frames", 1)
                    for i in range(n):
                        gif.seek(i)
                        frame_rgba = gif.convert("RGBA")
                        # Paste this frame onto the running canvas.
                        # For GIFs with a transparent colour the alpha channel
                        # acts as a mask so only opaque pixels overwrite.
                        canvas.paste(frame_rgba, (0, 0), mask=frame_rgba.split()[3])
                        self._gif_frames.append(canvas.copy())
                        self._gif_durations.append(
                            max(20, gif.info.get("duration", 100))
                        )
                except Exception:
                    continue
            self._has_gif = bool(self._gif_frames)
        except ImportError:
            pass   # Pillow not available — use ASCII fallback

    # ── Frame rendering ───────────────────────────────────────────────────────

    def _render_current(self) -> None:
        try:
            art = self.query_one("#buddy-art", Static)
        except NoMatches:
            return
        if self._render_cols < 4 or self._render_rows < 2:
            return   # not sized yet — skip to avoid zero-dimension renders
        if self._has_gif:
            frame = self._gif_frames[self._gif_frame_idx % len(self._gif_frames)]
            art.update(_render_pil_frame(
                frame, self._render_cols, self._render_rows, contain=self._zoom_mode
            ))
        else:
            art.update(_BUDDY_ASCII[self._ascii_idx % len(_BUDDY_ASCII)])

    # ── Tick (called from BirdClawApp._tick_spinners every ~100 ms) ───────────

    def tick(self, tick: int) -> None:   # noqa: ARG002
        self._tick_count += 1

        if self._has_gif:
            # Advance GIF frame when accumulated time exceeds current frame duration
            self._gif_elapsed += 100   # each tick ≈ 100 ms (_SPINNER_INTERVAL)
            cur_dur = self._gif_durations[
                self._gif_frame_idx % len(self._gif_durations)
            ]
            if self._gif_elapsed >= cur_dur:
                self._gif_elapsed = 0
                self._gif_frame_idx += 1
                self._render_current()
        else:
            # ASCII: advance every 8 ticks (~0.8 s)
            if self._tick_count % 8 == 0:
                self._ascii_idx += 1
                self._render_current()

    def update_zoom_btn(self, is_full: bool) -> None:
        self._zoom_mode = is_full
        try:
            self.query_one("#buddy-zoom", FlatBtn).update("[-]" if is_full else "[+]")
        except NoMatches:
            pass
        try:
            self.query_one("#buddy-art", Static).update("")  # clear stale frame
        except NoMatches:
            pass
        self.set_timer(0.12, self._rerender_after_settle)
        self.set_timer(0.35, self._rerender_after_settle)  # belt-and-suspenders

    def on_flat_btn_pressed(self, event: FlatBtn.Pressed) -> None:
        if event.button.id == "buddy-dismiss":
            self.display = False
            event.stop()
        elif event.button.id == "buddy-zoom":
            self.post_message(BuddyPanel.ZoomToggle())
            event.stop()


# ---------------------------------------------------------------------------
# Conversation pane
# ---------------------------------------------------------------------------

def _mode_bar_text(mode: str, hints: str) -> Text:
    """Build a Rich Text for the conversation mode bar.

    Uses explicit Text objects (not markup strings) so square-bracket
    tokens like [chat] are never mis-parsed by Rich's markup engine.
    """
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append(f" {mode} ", style="bold")   # mode label — bold, inherits CSS color
    t.append(f"  {hints}", style="dim")
    return t


class ConversationPane(Widget):
    """Bottom pane — soul conversation with chat / edit / plan modes."""

    input_mode: reactive[str] = reactive("chat")   # "chat" | "edit" | "plan"

    BINDINGS: ClassVar = [
        Binding("ctrl+e", "toggle_edit", "Edit mode", show=True),
    ]

    DEFAULT_CSS = """
    ConversationPane {
        height: 30%;
        min-height: 8;
    }
    ConversationPane #conv-row {
        height: 100%;
        layout: horizontal;
        padding: 0;
        margin: 0;
    }
    ConversationPane #conv-main {
        width: 1fr;
        height: 100%;
        layout: vertical;
        border: solid $primary-darken-2;
    }
    ConversationPane #conv-log {
        height: 1fr;
        background: transparent;
        scrollbar-gutter: stable;
    }
    ConversationPane #conv-input {
        height: 3;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
        display: block;
    }
    ConversationPane #conv-textarea {
        height: 6;
        border: solid $warning-darken-2;
        background: $surface-darken-1;
        display: none;
    }
    ConversationPane #conv-plan-box {
        height: 8;
        border: solid $accent-darken-2;
        background: $surface-darken-1;
        display: none;
        padding: 0 1;
    }
    ConversationPane #mode-bar {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    ConversationPane #conv-thinking {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="conv-row"):
            with Vertical(id="conv-main"):
                yield Static(_mode_bar_text("chat", "Ctrl+E=edit mode  Enter=send  ?=help"), id="mode-bar")
                yield RichLog(id="conv-log", highlight=False, markup=False,
                              max_lines=200, wrap=True)
                yield Static("", id="conv-thinking")
                yield Input(placeholder="Message BirdClaw… (?=help  /command)", id="conv-input")
                yield TextArea(id="conv-textarea")
                yield RichLog(id="conv-plan-box", max_lines=30)
            yield BuddyPanel(id="buddy-panel")

    def on_mount(self) -> None:
        try:
            conv_main = self.query_one("#conv-main")
            conv_main.border_title = "CHAT"
            conv_main.border_title_align = "left"
        except Exception:
            pass
        self._thinking_tick = 0

    def _load_history(self) -> None:
        from birdclaw.memory.history import History
        log: RichLog = self.query_one("#conv-log", RichLog)
        h = History.load_latest()
        if h:
            for turn in h.recent(20):
                log.write(render_turn(turn.role, turn.content))

    def append_turn(self, role: str, content: str) -> None:
        import time as _ti
        from rich.markdown import Markdown
        self.hide_thinking()
        log: RichLog = self.query_one("#conv-log", RichLog)
        ts = _ti.strftime("%H:%M")
        # Visual separator between turns
        log.write(Text("─" * 40, style="dim $primary-darken-3"))
        if role == "assistant" and content.strip():
            label = Text()
            label.append("BirdClaw  ", style="bold green")
            label.append(ts, style="dim")
            log.write(label)
            # code_theme="ansi_dark" prevents Monokai's hard dark background on code blocks
            log.write(Markdown(content, code_theme="ansi_dark"))
        else:
            label = Text()
            label.append("You  ", style="bold white")
            label.append(ts, style="dim")
            log.write(label)
            log.write(Text(content, style="white"))

    def show_thinking(self, tick: int) -> None:
        try:
            bar = self.query_one("#mode-bar", Static)
            t = spinner_text("BirdClaw is thinking…", tick)
            bar.update(t)
        except NoMatches:
            pass

    def hide_thinking(self) -> None:
        try:
            bar = self.query_one("#mode-bar", Static)
            mode = self.input_mode
            hints = {
                "chat": "Ctrl+E=edit mode  Enter=send  ?=help",
                "edit": "Ctrl+E=back to chat  Enter=newline  Ctrl+Enter=send",
                "plan": "Enter=approve  Escape=cancel",
            }.get(mode, "Enter=send")
            bar.update(_mode_bar_text(mode, hints))
        except NoMatches:
            pass

    def show_plan(self, outcome: str, stages: list[str]) -> None:
        """Switch to plan mode, showing the generated plan for approval."""
        self.input_mode = "plan"
        try:
            box: RichLog = self.query_one("#conv-plan-box", RichLog)
            box.clear()
            box.write(Text(f"Outcome: {outcome}", style="bold white"))
            for i, stage in enumerate(stages, 1):
                box.write(Text(f"  {i}. {stage}", style="cyan"))
            box.write(Text("", style=""))
            box.write(Text("Enter = approve and run  |  Escape = cancel", style="dim"))
        except NoMatches:
            pass

    def clear_input(self) -> None:
        try:
            self.query_one("#conv-input", Input).value = ""
        except NoMatches:
            pass

    def get_input(self) -> str:
        if self.input_mode == "edit":
            try:
                return self.query_one("#conv-textarea", TextArea).text
            except NoMatches:
                return ""
        try:
            return self.query_one("#conv-input", Input).value
        except NoMatches:
            return ""

    def focus_input(self) -> None:
        try:
            if self.input_mode == "edit":
                self.query_one("#conv-textarea", TextArea).focus()
            else:
                self.query_one("#conv-input", Input).focus()
        except NoMatches:
            pass

    def action_toggle_edit(self) -> None:
        if self.input_mode == "chat":
            self.input_mode = "edit"
        elif self.input_mode == "edit":
            self.input_mode = "chat"

    def watch_input_mode(self, mode: str) -> None:
        try:
            inp   = self.query_one("#conv-input",    Input)
            ta    = self.query_one("#conv-textarea",  TextArea)
            plan  = self.query_one("#conv-plan-box",  RichLog)
            bar   = self.query_one("#mode-bar",       Static)
            if mode == "chat":
                inp.display  = True
                ta.display   = False
                plan.display = False
                bar.update(_mode_bar_text("chat", "Ctrl+E=edit mode  Enter=send  ?=help"))
                self.border_subtitle = "chat"
                inp.focus()
            elif mode == "edit":
                inp.display  = False
                ta.display   = True
                plan.display = False
                bar.update(_mode_bar_text("edit", "Ctrl+E=back to chat  Enter=newline  Ctrl+Enter=send"))
                self.border_subtitle = "edit — Ctrl+E to exit"
                ta.focus()
            elif mode == "plan":
                inp.display  = False
                ta.display   = False
                plan.display = True
                bar.update(_mode_bar_text("plan", "Enter=approve  Escape=cancel"))
                self.border_subtitle = "plan approval"
                plan.focus()
        except NoMatches:
            pass


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

_THEMES = {
    "dark":       "textual-dark",
    "light":      "textual-light",
    "solarized":  "solarized-light",
    "catppuccin": "catppuccin-mocha",
}


class BirdClawApp(App):
    """BirdClaw three-pane TUI."""

    TITLE    = "🐦 BirdClaw"
    CSS_PATH = None   # all CSS is inline on widgets

    # Ctrl+P is claimed by Textual's command palette — we keep it.
    # Ctrl+C is handled manually (3-state) so we override the default.
    BINDINGS: ClassVar[list] = [
        # Quit
        Binding("ctrl+q",           "quit",             "Quit",         show=True,  priority=True),
        Binding("ctrl+d",           "quit",             "Exit",         show=False),
        # Ctrl+C — 3-state handler (not a direct quit)
        Binding("ctrl+c",           "handle_ctrl_c",    "",             show=False, priority=True),
        # Overlays
        Binding("ctrl+s",           "open_shell",       "Shell",        show=True),
        Binding("ctrl+l",           "open_model",       "Model",        show=True),
        Binding("ctrl+g",           "open_agents",      "Agents",       show=True),
        Binding("ctrl+f",           "open_search",      "Search",       show=True),
        Binding("question_mark",    "open_help",        "Help",         show=True),
        # Resize conversation
        Binding("[",                "shrink_conv",      "Shrink chat",  show=False),
        Binding("]",                "grow_conv",        "Grow chat",    show=False),
        Binding("backslash",        "cycle_layout",     "Layout \\",    show=True),
        # Task tabs (global so they work from any pane)
        Binding("f1",               "standing_tab",     "Standing",     show=False),
        Binding("f2",               "active_tab",       "Active",       show=False),
    ]

    _conv_height_idx: int   = 1      # index into _CONV_HEIGHT_PRESETS
    _task_width_idx:  int   = 1      # index into _TASK_WIDTH_PRESETS (default 30%)
    _layout_swapped:  bool  = False  # True → output left, tasks right
    _buddy_full:      bool  = False  # True → buddy panel 80 cols (full GIF view)
    _last_ctrl_c:     float = 0.0
    _spinner_tick:    int   = 0

    def __init__(self, **kw) -> None:
        import uuid as _uuid
        super().__init__(**kw)
        self._tui_uuid: str = _uuid.uuid4().hex[:12]
        # tui_session_id matches what gateway generates: _make_session_id("tui", _tui_uuid)
        self.tui_session_id: str = f"tui:{self._tui_uuid}"
        self._st_model:      str = ""
        self._st_session:    str = ""
        self._st_tokens:     int = 0
        self._st_ctx_limit:  int = 0
        self._st_stage:      str = ""
        self._st_sys_notice: str = ""   # transient — shows dream/cleanup events in subtitle

    # ── Compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield TaskListPane(id="tasks-pane")
            yield OutputPane(id="output-pane")
        yield ConversationPane(id="conv-pane")
        yield Footer()

    def on_mount(self) -> None:
        from birdclaw.config import settings
        try:
            from birdclaw.llm.model_profile import combined_display_name as _cdn
            _model_display = _cdn()
        except Exception:
            _model_display = settings.llm_model
        self._st_model     = _model_display
        self._st_ctx_limit = settings.n_ctx
        self.sub_title = f"{Path.cwd()}  ·  ● {_model_display}"
        # Apply saved theme after first render (call_after_refresh avoids Textual init race).
        # prefs.theme now stores the raw Textual name (e.g. "catppuccin-mocha").
        # _THEMES short-key lookup is kept for backwards compat with old prefs files.
        prefs = TuiPrefs.load()
        _raw = prefs.theme if prefs.theme else settings.theme
        textual_theme = _THEMES.get(_raw, _raw) if _raw else "textual-dark"
        if textual_theme and textual_theme != "textual-dark":
            def _apply_theme(t: str = textual_theme) -> None:
                try:
                    self.theme = t
                except Exception:
                    pass
            self.call_after_refresh(_apply_theme)
        # Restore saved layout preferences
        self._load_layout_prefs()
        self._setup_gateway()
        self.set_interval(_POLL_INTERVAL,    self._poll_tasks)
        self.set_interval(_SPINNER_INTERVAL, self._tick_spinners)
        self.query_one("#conv-pane", ConversationPane).focus_input()
        # Load MCP in background so the TUI opens immediately
        self.run_worker(self._load_mcp, exclusive=False, thread=True)

    def _load_mcp(self) -> None:
        """Load MCP servers in a background thread — keeps startup fast."""
        try:
            from birdclaw.tools.mcp.manager import mcp_manager, mcp_bridge
            mcp_manager.load_from_config()
            mcp_bridge.register_all()
        except Exception as exc:
            logger.warning("MCP load failed: %s", exc)

    def watch_theme(self, theme: str) -> None:
        """Persist theme and force buddy rerender on every theme change.

        Called when theme changes via Layout picker, command palette, or code.
        Saving here means all theme sources (including palette) persist across sessions.
        """
        # Persist — store the raw Textual theme name so any theme can be restored
        try:
            prefs = TuiPrefs.load()
            if prefs.theme != theme:
                prefs.theme = theme
                prefs.save()
        except Exception:
            pass
        self.refresh(layout=True)
        self.set_timer(0.15, self._trigger_buddy_rerender)

    def on_buddy_panel_zoom_toggle(self, _event: BuddyPanel.ZoomToggle) -> None:
        """Toggle buddy between compact and full display when ⊞ is clicked."""
        from birdclaw.tui.prefs import TuiPrefs
        new_full = not self._buddy_full
        self._set_buddy_full(new_full)
        prefs = TuiPrefs.load()
        prefs.buddy_full = new_full
        prefs.save()
        label = "Full (80 cols)" if new_full else "Compact (44 cols)"
        self.notify(f"Buddy: {label}  (saved)", timeout=1.5)

    def on_header_clicked(self, _event) -> None:  # type: ignore[override]
        """Intercept Header clicks — show status summary instead of default zoom."""
        _event.stop()
        parts = [
            f"Model    {self._st_model}" if self._st_model else "",
            f"Task     {self._st_session}" if self._st_session else "",
        ]
        if self._st_tokens:
            k = self._st_tokens / 1000
            if self._st_ctx_limit:
                pct = int(100 * self._st_tokens / self._st_ctx_limit)
                parts.append(f"Context  {k:.1f}k / {self._st_ctx_limit//1000}k  ({pct}%)")
            else:
                parts.append(f"Context  {k:.1f}k tokens")
        if self._st_stage:
            parts.append(f"Stage    {self._st_stage}")
        try:
            from birdclaw.agent.approvals import approval_queue
            n = len(approval_queue.list_pending())
            if n:
                parts.append(f"Approvals  {n} pending")
        except Exception:
            pass
        body = "\n".join(p for p in parts if p) or "ready"
        self.notify(body, title="Status", timeout=5)

    def _load_layout_prefs(self) -> None:
        """Restore user's saved pane sizes and arrangement from disk."""
        prefs = TuiPrefs.load()
        # Chat height
        if prefs.chat_height_pct in _CONV_HEIGHT_PRESETS:
            self._conv_height_idx = _CONV_HEIGHT_PRESETS.index(prefs.chat_height_pct)
            self._apply_conv_height()
        # Task list width
        if prefs.task_width_pct in _TASK_WIDTH_PRESETS:
            self._task_width_idx = _TASK_WIDTH_PRESETS.index(prefs.task_width_pct)
            self._apply_task_width()
        # Pane arrangement
        if prefs.layout_swapped:
            self._set_layout_swapped(True)
        # Buddy panel size
        if prefs.buddy_full:
            self._set_buddy_full(True)

    def _clear_sys_notice(self) -> None:
        self._st_sys_notice = ""
        self._redraw_subtitle()

    def update_status(self, *, model: str | None = None, session: str | None = None,
                      tokens: int | None = None, ctx_limit: int | None = None,
                      stage: str | None = None) -> None:
        """Update header status bar (model · session · ctx · stage · approvals)."""
        if model     is not None: self._st_model     = model
        if session   is not None: self._st_session   = session
        if tokens    is not None: self._st_tokens    = tokens
        if ctx_limit is not None: self._st_ctx_limit = ctx_limit
        if stage     is not None: self._st_stage     = stage
        self._redraw_subtitle()

    def _redraw_subtitle(self, tick: int = 0) -> None:
        """Build the header subtitle — shown as the status bar in the Textual Header."""
        import os
        parts: list[str] = []

        # CWD
        parts.append(str(Path(os.getcwd()).resolve()))

        # Model — dot blinks when agent is busy, solid when idle
        if self._st_model:
            dot = "●" if (not self._st_stage or tick % 8 < 5) else " "
            parts.append(f"{dot} {self._st_model}")

        # Focused task — show title if available, fall back to short ID
        if self._st_session:
            task = task_registry.get(self._st_session)
            if task and getattr(task, "title", ""):
                parts.append(task.title[:30])
            else:
                parts.append(f"#{self._st_session[:8]}")

        # Context / token usage
        if self._st_tokens:
            k = self._st_tokens / 1000
            if self._st_ctx_limit:
                used_pct = int(100 * self._st_tokens / self._st_ctx_limit)
                parts.append(f"ctx {k:.1f}k/{self._st_ctx_limit//1000}k ({used_pct}%)")
            else:
                parts.append(f"{k:.1f}k tok")

        # Current stage — animated spinner
        if self._st_stage:
            frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
            parts.append(f"{frame} {self._st_stage}")

        # Pending approvals
        try:
            from birdclaw.agent.approvals import approval_queue
            n_pending = len(approval_queue.list_pending())
            if n_pending:
                parts.append(f"⚠ {n_pending} approval{'s' if n_pending > 1 else ''}")
        except Exception:
            pass

        # Transient system notice (dream / cleanup events)
        if self._st_sys_notice:
            parts.append(f"✦ {self._st_sys_notice}")

        self.sub_title = "  ·  ".join(parts) if parts else "ready"

    def _setup_gateway(self) -> None:
        """Connect to the daemon TUI socket if running; otherwise start in-process gateway."""
        self._daemon_sock: "socket.socket | None" = None
        self._daemon_sock_lock = __import__("threading").Lock()
        self._daemon_session_id: str = ""   # assigned by daemon; used for reconnect

        if self._try_connect_daemon():
            logger.info("tui: connected to daemon socket — thin-client mode")
            return

        # Daemon not running — start everything in-process (standalone mode)
        from birdclaw.channels.tui_channel import TUIChannel
        from birdclaw.gateway.gateway import gateway
        from birdclaw.skills.cron import cron_service
        self._tui_channel = TUIChannel()
        self._tui_channel.on_deliver(
            lambda msg: self.call_from_thread(self._on_gateway_message, msg)
        )
        gateway.register(self._tui_channel)
        gateway.start()
        # Reap orphan tasks before first UI render so they don't flash as "running"
        gateway._reap_orphan_tasks()
        # Pre-create session so cross-notifications from test_socket/cron arrive immediately
        _sess = gateway._session_mgr.get_or_create("tui", self._tui_uuid)
        _sess.launch_cwd = str(Path.cwd().resolve())
        with gateway._lock:
            gateway._session_channel[self.tui_session_id] = "tui"
        cron_service.start()
        try:
            from birdclaw.gateway.test_socket import start as _start_test_sock
            _start_test_sock()
        except Exception as _e:
            logger.warning("test_socket: could not start: %s", _e)

    def on_unmount(self) -> None:
        """Send a clean-quit message to the daemon before the TUI closes."""
        import json as _json
        with self._daemon_sock_lock:
            sock = self._daemon_sock
        if sock is not None:
            try:
                sock.sendall((_json.dumps({"type": "quit"}) + "\n").encode())
            except OSError:
                pass

    def _try_connect_daemon(self) -> bool:
        """Attempt to connect to ~/.birdclaw/tui.sock. Returns True on success."""
        import json as _json
        import socket as _socket
        from pathlib import Path as _Path
        sock_path = _Path.home() / ".birdclaw" / "tui.sock"
        if not sock_path.exists():
            return False
        try:
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.connect(str(sock_path))
        except OSError as e:
            logger.info("tui: daemon socket not connectable: %s", e)
            return False

        # Send hello — include cwd (write target) and stored session_id for reconnect
        hello: dict = {"type": "hello", "cwd": str(Path.cwd().resolve())}
        if self._daemon_session_id:
            hello["session_id"] = self._daemon_session_id
        try:
            s.sendall((_json.dumps(hello) + "\n").encode())
        except OSError as e:
            logger.warning("tui: failed to send hello: %s", e)
            s.close()
            return False

        self._daemon_sock = s
        import threading as _threading
        t = _threading.Thread(
            target=self._daemon_reader,
            args=(s,),
            daemon=True,
            name="tui-daemon-reader",
        )
        t.start()
        return True

    def _daemon_reader(self, sock: "socket.socket") -> None:
        """Background thread: read JSON lines from the daemon and dispatch to UI."""
        import json as _json
        from birdclaw.gateway.channel import OutgoingMessage
        buf = b""
        try:
            while True:
                try:
                    chunk = sock.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue

                    # Session assignment from daemon — store and skip UI dispatch
                    if obj.get("type") == "session":
                        self._daemon_session_id = obj.get("session_id", "")
                        logger.info("tui: daemon assigned session %s", self._daemon_session_id)
                        continue

                    msg = OutgoingMessage(
                        session_id=obj.get("session_id", ""),
                        content=obj.get("content", ""),
                        msg_type=obj.get("msg_type", "reply"),
                        task_id=obj.get("task_id") or "",
                        metadata=obj.get("metadata") or {},
                    )
                    self.call_from_thread(self._on_gateway_message, msg)
        finally:
            logger.info("tui: daemon socket closed")
            with self._daemon_sock_lock:
                self._daemon_sock = None
            # Reconnect in background so messages aren't lost after daemon restart
            import threading as _th2
            _th2.Thread(target=self._daemon_reconnect_loop, daemon=True,
                        name="tui-daemon-reconnect").start()

    def _daemon_reconnect_loop(self) -> None:
        """Retry daemon socket connection after disconnect. Runs in a daemon thread."""
        import time as _time
        for attempt in range(24):   # retry for ~2 minutes (24 × 5s)
            _time.sleep(5)
            with self._daemon_sock_lock:
                if self._daemon_sock is not None:
                    return   # already reconnected by another path
            if self._try_connect_daemon():
                logger.info("tui: reconnected to daemon socket (attempt %d)", attempt + 1)
                return
        logger.warning("tui: daemon reconnect failed after 2 min — will retry on next message")

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll_tasks(self) -> None:
        """Poll output pane for new events. Task list rebuilds are driven by
        tick_spinner's snapshot check — not here — to prevent flicker."""
        # Sync tasks written by the daemon process (thin-client mode only).
        # In standalone mode tasks are in the same process — disk sync would
        # pull in tasks from other sessions (CLI, etc.) causing overflow.
        if self._daemon_sock is not None:
            try:
                task_registry.sync_from_disk()
            except Exception:
                pass
        # Drain cross-process system notifications (dream, cleanup, etc.)
        try:
            from birdclaw.gateway.notify import drain_notifications
            for n in drain_notifications():
                msg      = n.get("message", "")
                title    = n.get("title", "System")
                severity = n.get("severity", "information")
                self.notify(msg, title=title, severity=severity, timeout=8)
                # Also show in the header subtitle for visibility; auto-clear after 30s
                notice = f"{title}: {msg}" if title else msg
                self._st_sys_notice = notice[:60]
                self._redraw_subtitle()
                self.set_timer(30, self._clear_sys_notice)
        except Exception:
            pass

        output = self.query_one("#output-pane", OutputPane)
        output.refresh_output()
        tasks_pane = self.query_one("#tasks-pane", TaskListPane)
        task_ids = [tid for tid in tasks_pane._task_ids if tid]

        # Auto-focus: switch to newest running task when nothing focused,
        # or when the currently focused task has reached a terminal state.
        focused = output.focused_task_id
        focused_terminal = False
        if focused:
            t = task_registry.get(focused)
            focused_terminal = t is None or t.status in ("completed", "failed", "stopped")

        if (not focused or focused_terminal) and task_ids:
            running = [tid for tid in task_ids
                       if (t := task_registry.get(tid)) and t.status == "running"]
            if running:
                output.focused_task_id = running[0]

    def _tick_spinners(self) -> None:
        self._spinner_tick += 1
        t = self._spinner_tick
        self.query_one("#tasks-pane",  TaskListPane).tick_spinner(t)
        self.query_one("#output-pane", OutputPane).tick_spinner(t)
        conv = self.query_one("#conv-pane", ConversationPane)
        if self._soul_thinking:
            conv.show_thinking(t)
        # Tick buddy panel animation
        try:
            conv.query_one("#buddy-panel", BuddyPanel).tick(t)
        except NoMatches:
            pass
        # Refresh header subtitle every tick for spinner animation; approval count every ~2 s
        self._redraw_subtitle(tick=t)

    _soul_thinking:  bool = False

    # ── Ctrl+C — 3-state (openclaw pattern) ──────────────────────────────────

    def action_handle_ctrl_c(self) -> None:
        """
        State 1: input has text          → clear input
        State 2: input empty, first hit  → warn, set timer
        State 3: input empty, second hit within 1s → stop running task
        """
        now = time.time()
        conv = self.query_one("#conv-pane", ConversationPane)
        text = conv.get_input().strip()

        if text:
            conv.clear_input()
            return

        if now - self._last_ctrl_c <= 1.0:
            self._stop_focused_task()
        else:
            self._last_ctrl_c = now
            self.notify(
                "Press Ctrl+C again to stop the running task, or Ctrl+Q to quit.",
                title="Ctrl+C",
                severity="warning",
            )

    def _stop_focused_task(self) -> None:
        from birdclaw.agent.orchestrator import orchestrator
        tid = self.query_one("#output-pane", OutputPane).focused_task_id
        if not tid:
            self.notify("No task selected.", severity="information")
            return
        # Signal interrupt to the agent thread first, then mark registry
        found = orchestrator.interrupt_by_task(tid)
        if not found:
            try:
                task_registry.stop(tid)
            except (KeyError, ValueError) as e:
                self.notify(str(e), severity="error")
                return
        # Unblock any approval requests waiting for this task
        from birdclaw.agent.approvals import approval_queue
        approval_queue.deny_all_for_task(tid)
        self.notify(f"Task {tid[:8]}… stopping…", severity="warning")
        # Mark the running ToolCard as failed in the output pane
        try:
            output = self.query_one("#output-pane", OutputPane)
            if output.focused_task_id == tid and output._pending_card:
                output._pending_card.fail("aborted by user")
                output._pending_card = None
        except (NoMatches, Exception):
            pass

    def _handle_skills(self, args: str) -> None:
        """/skills [name]  — list skills or show one skill's details."""
        from birdclaw.skills.loader import load_skills
        skills = load_skills()
        if not skills:
            self.notify("No skills installed.", severity="information")
            return
        name = args.strip()
        if name:
            skill = next((s for s in skills if s.name == name), None)
            if skill is None:
                self.notify(f"Skill '{name}' not found.", severity="warning")
                return
            sched = f"  schedule: {skill.schedule}" if skill.schedule else "  (no schedule)"
            self.notify(
                f"{skill.name}\n{skill.description}\n{sched}",
                title=f"Skill: {skill.name}",
                severity="information",
            )
        else:
            lines = []
            for s in sorted(skills, key=lambda x: x.name):
                sched = f" [{s.schedule}]" if s.schedule else ""
                lines.append(f"{s.name}{sched}: {s.description[:50]}")
            self.notify("\n".join(lines), title="Skills", severity="information")

    def _handle_cron(self, args: str) -> None:
        """/cron [list|enable <id>|disable <id>|run <skill>]  — manage scheduled skills."""
        from birdclaw.skills.cron import cron_service
        parts = args.split(None, 1)
        sub   = parts[0].lower() if parts else "list"
        rest  = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("list", ""):
            entries = cron_service.list()
            if not entries:
                self.notify("No scheduled skills.", severity="information")
                return
            import time as _t
            lines = []
            for e in entries:
                status = "✓" if e.enabled else "✗"
                nxt = ""
                if e.enabled and e.next_run_at:
                    secs = max(0, int(e.next_run_at - _t.time()))
                    h, m = divmod(secs // 60, 60)
                    nxt = f" (next: {h}h{m:02d}m)" if h else f" (next: {m}m)"
                lines.append(f"{status} [{e.cron_id[:6]}] {e.skill_name}{nxt} — {e.schedule}")
            self.notify("\n".join(lines), title="Cron schedule", severity="information")

        elif sub == "run":
            ok = cron_service.trigger(rest)
            if ok:
                self.notify(f"Triggered skill '{rest}'.", severity="information")
            else:
                self.notify(f"Skill '{rest}' not found.", severity="warning")

        elif sub == "enable":
            ok = cron_service.enable(rest)
            self.notify(f"Enabled {rest}." if ok else f"Not found: {rest}",
                        severity="information" if ok else "warning")

        elif sub == "disable":
            ok = cron_service.disable(rest)
            self.notify(f"Disabled {rest}." if ok else f"Not found: {rest}",
                        severity="information" if ok else "warning")

        else:
            self.notify(
                "/cron list  /cron run <skill>  /cron enable <id>  /cron disable <id>",
                severity="information",
            )

    def _handle_approve(self, args: str) -> None:
        """/approve <id> [allow|always|deny]  — resolve a pending agent approval."""
        from birdclaw.agent.approvals import approval_queue
        parts = args.split()
        if not parts:
            pending = approval_queue.list_pending()
            if not pending:
                self.notify("No pending approvals.", severity="information")
            else:
                lines = [r.summary() for r in pending]
                self.notify("\n".join(lines), title="Pending approvals", severity="warning")
            return

        approval_id  = parts[0]
        decision_raw = parts[1].lower() if len(parts) > 1 else "allow"
        _map = {
            "allow":        "allow_once",
            "once":         "allow_once",
            "allow_once":   "allow_once",
            "always":       "allow_always",
            "allow_always": "allow_always",
            "deny":         "deny",
            "no":           "deny",
            "reject":       "deny",
        }
        decision = _map.get(decision_raw, "allow_once")
        ok = approval_queue.resolve(approval_id, decision)  # type: ignore[arg-type]
        if ok:
            self.notify(
                f"Approval {approval_id[:6]}: {decision}",
                severity="information",
            )
        else:
            self.notify(
                f"No pending approval matching '{approval_id}'.",
                severity="warning",
            )

    # ── Overlay actions ───────────────────────────────────────────────────────

    def action_open_shell(self) -> None: self.push_screen(ShellOverlay())
    def action_open_help(self)  -> None: self.push_screen(HelpOverlay())

    def action_open_model(self) -> None:
        from birdclaw.config import settings
        current = settings.llm_model
        # Common local + cloud models — user can add via BC_LLM_MODEL override
        items = [
            ("gemma-4-4b  (current)" if current == "gemma-4-4b"  else "gemma-4-4b",  "gemma-4-4b"),
            ("gemma-4-12b (current)" if current == "gemma-4-12b" else "gemma-4-12b", "gemma-4-12b"),
            ("llama-3-8b  (current)" if current == "llama-3-8b"  else "llama-3-8b",  "llama-3-8b"),
            ("gpt-4o      (current)" if current == "gpt-4o"      else "gpt-4o",       "gpt-4o"),
            ("claude-sonnet-4-6",                                                      "claude-sonnet-4-6"),
            (f"{current}  ← active",                                                  current),
        ]
        seen: set[str] = set()
        deduped = [(l, v) for l, v in items if not (v in seen or seen.add(v))]  # type: ignore[func-returns-value]

        def _apply(model: str | None) -> None:
            if model and model != current:
                import os
                os.environ["BC_LLM_MODEL"] = model
                self.notify(f"Model → {model}  (restart daemon to take effect)",
                            severity="information", timeout=5)

        self.push_screen(FuzzyPickerOverlay("Model", deduped), _apply)

    def action_open_agents(self) -> None:
        from birdclaw.agent.orchestrator import orchestrator
        handles = orchestrator.running_agents()
        if not handles:
            self.notify("No agents running.", severity="information")
            return
        items = [(f"[{h.agent_id[:8]}]  task:{h.task_id[:8]}", h.task_id)
                 for h in handles]

        def _focus(task_id: str | None) -> None:
            if task_id:
                try:
                    self.query_one("#output-pane", OutputPane).focused_task_id = task_id
                except NoMatches:
                    pass

        self.push_screen(FuzzyPickerOverlay(f"Agents ({len(handles)} running)", items),
                         _focus)

    def action_open_search(self) -> None:
        output = self.query_one("#output-pane", OutputPane)
        path   = output._session_log_path()
        if not path:
            self.notify("No task selected — select a task first.", severity="warning")
            return
        self.push_screen(SearchOverlay(str(path)))

    # ── Task tab shortcuts (global) ───────────────────────────────────────────

    def action_standing_tab(self) -> None:
        self.query_one("#tasks-pane", TaskListPane).tab = "standing"

    def action_active_tab(self) -> None:
        self.query_one("#tasks-pane", TaskListPane).tab = "active"

    # ── Conversation pane resize ──────────────────────────────────────────────

    def action_shrink_conv(self) -> None:
        self._conv_height_idx = max(0, self._conv_height_idx - 1)
        self._apply_conv_height()
        self._save_layout_prefs()

    def action_grow_conv(self) -> None:
        self._conv_height_idx = min(len(_CONV_HEIGHT_PRESETS) - 1,
                                    self._conv_height_idx + 1)
        self._apply_conv_height()
        self._save_layout_prefs()

    def _apply_conv_height(self) -> None:
        pct = _CONV_HEIGHT_PRESETS[self._conv_height_idx]
        try:
            self.query_one("#conv-pane", ConversationPane).styles.height = f"{pct}%"
        except NoMatches:
            pass
        self._trigger_buddy_rerender()

    def _trigger_buddy_rerender(self) -> None:
        """Ask the BuddyPanel to re-render after the next layout settle."""
        try:
            self.query_one("#buddy-panel", BuddyPanel).force_rerender()
        except NoMatches:
            pass

    def _save_layout_prefs(self) -> None:
        """Persist current layout state to disk."""
        try:
            prefs = TuiPrefs.load()
            prefs.chat_height_pct = _CONV_HEIGHT_PRESETS[self._conv_height_idx]
            prefs.task_width_pct  = _TASK_WIDTH_PRESETS[self._task_width_idx]
            prefs.layout_swapped  = self._layout_swapped
            prefs.buddy_full      = self._buddy_full
            prefs.save()
        except Exception:
            pass

    def action_cycle_layout(self) -> None:
        """Open the layout picker overlay (\\ key)."""
        self.action_open_layout()

    def action_open_layout(self) -> None:
        """Layout picker — adjust chat height, task list width, arrangement, and theme."""
        chat_pct  = _CONV_HEIGHT_PRESETS[self._conv_height_idx]
        tasks_pct = _TASK_WIDTH_PRESETS[self._task_width_idx]
        from birdclaw.config import settings
        prefs     = TuiPrefs.load()
        cur_theme = prefs.theme if prefs.theme else settings.theme

        def _mark(active: bool) -> str:
            return "▶ " if active else "  "

        items = [
            # ── Chat height ────────────────────────────────────────────────
            (f"{_mark(chat_pct==20)}Chat height 20%  compact",          "chat-20"),
            (f"{_mark(chat_pct==30)}Chat height 30%  default",          "chat-30"),
            (f"{_mark(chat_pct==40)}Chat height 40%",                    "chat-40"),
            (f"{_mark(chat_pct==50)}Chat height 50%  balanced",         "chat-50"),
            (f"{_mark(chat_pct==60)}Chat height 60%  chat focus",       "chat-60"),
            (f"{_mark(chat_pct==70)}Chat height 70%  max chat",         "chat-70"),
            # ── Task list width ────────────────────────────────────────────
            (f"{_mark(tasks_pct==25)}Task list width 25%  narrow",      "tasks-25"),
            (f"{_mark(tasks_pct==30)}Task list width 30%  default",     "tasks-30"),
            (f"{_mark(tasks_pct==35)}Task list width 35%",               "tasks-35"),
            (f"{_mark(tasks_pct==40)}Task list width 40%",               "tasks-40"),
            (f"{_mark(tasks_pct==50)}Task list width 50%  wide",        "tasks-50"),
            # ── Arrangement ────────────────────────────────────────────────
            (f"{_mark(not self._layout_swapped)}Arrangement: Tasks Left | Output Right",  "arr-default"),
            (f"{_mark(self._layout_swapped)}Arrangement: Output Left | Tasks Right",      "arr-swap"),
            # ── Buddy panel size ───────────────────────────────────────────
            (f"{_mark(not self._buddy_full)}Buddy: Compact  (44 cols, cropped)",          "buddy-compact"),
            (f"{_mark(self._buddy_full)}Buddy: Full  (80 cols, full GIF view)",           "buddy-full"),
            # ── Theme (all themes accessible via Ctrl+P palette; these are quick presets) ──
            (f"{_mark(cur_theme in ('dark','textual-dark'))}Theme: Dark",                "theme-dark"),
            (f"{_mark(cur_theme in ('light','textual-light'))}Theme: Light",             "theme-light"),
            (f"{_mark(cur_theme in ('solarized','solarized-light'))}Theme: Solarized",   "theme-solarized"),
            (f"{_mark(cur_theme in ('catppuccin','catppuccin-mocha'))}Theme: Catppuccin","theme-catppuccin"),
        ]

        def _apply(value: str | None) -> None:
            if not value:
                return
            prefs = TuiPrefs.load()
            if value.startswith("chat-"):
                pct = int(value[5:])
                if pct in _CONV_HEIGHT_PRESETS:
                    self._conv_height_idx = _CONV_HEIGHT_PRESETS.index(pct)
                    self._apply_conv_height()
                    prefs.chat_height_pct = pct
                    prefs.save()
                    self.notify(f"Chat height → {pct}%  (saved)", timeout=1.5)
            elif value.startswith("tasks-"):
                pct = int(value[6:])
                if pct in _TASK_WIDTH_PRESETS:
                    self._task_width_idx = _TASK_WIDTH_PRESETS.index(pct)
                    self._apply_task_width()
                    prefs.task_width_pct = pct
                    prefs.save()
                    self.notify(f"Task list → {pct}%  (saved)", timeout=1.5)
            elif value == "arr-swap":
                self._set_layout_swapped(True)
                prefs.layout_swapped = True
                prefs.save()
                self.notify("Layout: Output Left | Tasks Right  (saved)", timeout=2)
            elif value == "arr-default":
                self._set_layout_swapped(False)
                prefs.layout_swapped = False
                prefs.save()
                self.notify("Layout: Tasks Left | Output Right  (saved)", timeout=2)
            elif value == "buddy-full":
                self._set_buddy_full(True)
                prefs.buddy_full = True
                prefs.save()
                self.notify("Buddy: Full display (80 cols)  (saved)", timeout=2)
            elif value == "buddy-compact":
                self._set_buddy_full(False)
                prefs.buddy_full = False
                prefs.save()
                self.notify("Buddy: Compact (44 cols)  (saved)", timeout=2)
            elif value and value.startswith("theme-"):
                theme_key = value[6:]  # strip "theme-"
                textual_theme = _THEMES.get(theme_key, "textual-dark")
                try:
                    self.theme = textual_theme  # watch_theme auto-saves
                except Exception:
                    pass
                self.notify(f"Theme: {theme_key}  (saved)", timeout=1.5)

        self.push_screen(FuzzyPickerOverlay("Layout", items), _apply)

    def _apply_task_width(self) -> None:
        pct = _TASK_WIDTH_PRESETS[self._task_width_idx]
        try:
            self.query_one("#tasks-pane", TaskListPane).styles.width = f"{pct}%"
        except NoMatches:
            pass
        self._trigger_buddy_rerender()

    def _set_layout_swapped(self, swap: bool) -> None:
        self._layout_swapped = swap
        try:
            tasks  = self.query_one("#tasks-pane",  TaskListPane)
            output = self.query_one("#output-pane", OutputPane)
            if swap:
                tasks.styles.order  = 2
                output.styles.order = 1
            else:
                tasks.styles.order  = 1
                output.styles.order = 2
        except (NoMatches, Exception):
            pass
        self._trigger_buddy_rerender()

    def _set_buddy_full(self, full: bool) -> None:
        from birdclaw.tui.prefs import _BUDDY_COMPACT_WIDTH, _BUDDY_FULL_WIDTH
        self._buddy_full = full
        try:
            panel = self.query_one("#buddy-panel", BuddyPanel)
            if full:
                panel.add_class("buddy-full")
                panel._render_cols = max(4, _BUDDY_FULL_WIDTH - 2)
            else:
                panel.remove_class("buddy-full")
                panel._render_cols = max(4, _BUDDY_COMPACT_WIDTH - 2)
            panel.update_zoom_btn(full)
        except NoMatches:
            pass

    # ── Task list → output pane wiring ────────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        try:
            task_id = self.query_one("#tasks-pane", TaskListPane)._selected_task_id()
            if task_id:
                self.query_one("#output-pane", OutputPane).focused_task_id = task_id
        except NoMatches:
            pass

    # ── Conversation input ────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "conv-input":
            return
        text = event.value.strip()
        if not text:
            return

        # Help shortcut
        if text == "?":
            event.input.value = ""
            self.push_screen(HelpOverlay())
            return

        # Slash commands
        if text.startswith("/"):
            event.input.value = ""
            self._handle_slash(text)
            return

        event.input.value = ""
        self._submit_user_message(text)

    def _handle_slash(self, raw: str) -> None:
        parts = raw.lstrip("/").split(None, 1)
        cmd   = parts[0].lower() if parts else ""
        args  = parts[1] if len(parts) > 1 else ""

        if cmd in ("help", "h"):
            self.push_screen(HelpOverlay())
        elif cmd == "shell":
            self.push_screen(ShellOverlay())
        elif cmd == "clear":
            try:
                self.query_one("#conv-log", RichLog).clear()
            except NoMatches:
                pass
            # Also erase the persisted history so it doesn't reload on restart
            try:
                from birdclaw.gateway.gateway import gateway
                from birdclaw.gateway.session_manager import _make_session_id
                sid = _make_session_id("tui", "local")
                session = gateway._session_mgr.get(sid)
                if session:
                    session.history.clear()
            except Exception:
                pass
        elif cmd == "abort":
            self._stop_focused_task()
        elif cmd == "approve":
            self._handle_approve(args)
        elif cmd == "model":
            self.action_open_model()
        elif cmd == "agents":
            self.action_open_agents()
        elif cmd in ("tasks", "status"):
            tasks = task_registry.list()
            lines = [f"{t.status:10} {t.task_id[:8]} {t.prompt[:40]}" for t in tasks[:10]]
            msg = "\n".join(lines) if lines else "no tasks"
            self.notify(msg, title="Tasks", severity="information")
        elif cmd == "standing":
            tasks = [t for t in task_registry.list() if t.team_id.startswith("skill:")]
            lines = [f"{t.team_id:20} {t.status}" for t in tasks[:10]]
            msg = "\n".join(lines) if lines else "no standing tasks"
            self.notify(msg, title="Standing tasks", severity="information")
        elif cmd == "skills":
            self._handle_skills(args)
        elif cmd == "cron":
            self._handle_cron(args)
        elif cmd == "buddy":
            self._toggle_buddy()
        elif cmd in ("layout", "pane", "resize"):
            self.action_open_layout()
        elif cmd == "verbose":
            self._set_verbose(args.strip())
        else:
            self.notify(f"Unknown command: /{cmd}  — type ? for help", severity="warning")

    def _set_verbose(self, level: str) -> None:
        valid = ("off", "on", "full")
        if level not in valid:
            self.notify(f"/verbose {' | '.join(valid)}", severity="information")
            return
        try:
            output = self.query_one("#output-pane", OutputPane)
            output.verbose_mode = level
            output._rebuild()   # rebuild cards at new verbosity
        except NoMatches:
            pass
        self.notify(f"Verbose: {level}", severity="information", timeout=2)

    def _toggle_buddy(self) -> None:
        try:
            panel = self.query_one("#buddy-panel", BuddyPanel)
            panel.display = not panel.display
        except NoMatches:
            pass

    def _submit_user_message(self, text: str) -> None:
        """Submit a user message to the gateway. Gateway calls back via _on_gateway_message."""
        import json as _json
        conv = self.query_one("#conv-pane", ConversationPane)
        conv.append_turn("user", text)
        self._soul_thinking = True
        try:
            output = self.query_one("#output-pane", OutputPane)
            self._pending_parent_task_id = output.focused_task_id or ""
        except Exception:
            self._pending_parent_task_id = ""

        with self._daemon_sock_lock:
            sock = self._daemon_sock
        if sock is not None:
            payload = (_json.dumps({"type": "msg", "content": text}) + "\n").encode()
            try:
                sock.sendall(payload)
            except OSError as e:
                logger.warning("tui: daemon socket send failed: %s", e)
                with self._daemon_sock_lock:
                    self._daemon_sock = None
            return

        from birdclaw.gateway.channel import IncomingMessage
        from birdclaw.gateway.gateway import gateway
        gateway.submit(IncomingMessage(channel_id="tui", user_id=self._tui_uuid, content=text))

    def _on_gateway_message(self, msg: "OutgoingMessage") -> None:
        """Receive all gateway pushes: replies, task updates, approvals. Runs on UI thread."""
        from birdclaw.gateway.channel import OutgoingMessage  # noqa — for type hint
        conv = self.query_one("#conv-pane", ConversationPane)

        if msg.msg_type in ("reply", "task_started"):
            self._soul_thinking = False
            conv.append_turn("assistant", msg.content)
            if msg.task_id:
                # If there was a focused parent task when the message was submitted,
                # link the new task as a child (follow-up) so it shows in the tree.
                parent_id = getattr(self, "_pending_parent_task_id", "")
                if parent_id and parent_id != msg.task_id:
                    try:
                        task_registry.assign_team(msg.task_id, parent_id)
                    except Exception:
                        pass
                self._pending_parent_task_id = ""
                # Auto-focus new task in output pane only if nothing is currently running there
                try:
                    op = self.query_one("#output-pane", OutputPane)
                    currently_focused = op.focused_task_id
                    from birdclaw.memory.tasks import task_registry as _tr
                    focused_task = _tr.get(currently_focused) if currently_focused else None
                    if not focused_task or focused_task.status not in ("running", "created", "waiting"):
                        op.focused_task_id = msg.task_id
                except NoMatches:
                    pass
                # Refresh task list so the new task appears immediately
                try:
                    self.query_one("#tasks-pane", TaskListPane).refresh_tasks()
                except NoMatches:
                    pass

        elif msg.msg_type == "task_complete":
            # Show a styled completion card in the chat pane
            try:
                import time as _ti
                log = conv.query_one("#conv-log", RichLog)
                ts = _ti.strftime("%H:%M")
                log.write(Text("─" * 40, style="dim $primary-darken-3"))
                card = Text()
                card.append("✔ Done  ", style="bold green")
                card.append(ts, style="dim")
                log.write(card)
                if msg.content:
                    log.write(Text(msg.content.strip(), style="green"))
            except Exception:
                conv.append_turn("assistant", msg.content)
            try:
                self.query_one("#output-pane", OutputPane).finalize_pending(msg.task_id or "")
            except NoMatches:
                pass
            self.notify(
                f"Task finished.",
                severity="information",
                timeout=4,
            )

        elif msg.msg_type in ("task_failed", "task_stopped"):
            label = "failed" if msg.msg_type == "task_failed" else "stopped"
            try:
                self.query_one("#output-pane", OutputPane).finalize_pending(msg.task_id or "")
            except NoMatches:
                pass
            # Show a styled failure card in the chat pane
            try:
                import time as _ti
                log = conv.query_one("#conv-log", RichLog)
                ts = _ti.strftime("%H:%M")
                log.write(Text("─" * 40, style="dim $primary-darken-3"))
                card = Text()
                icon = "✗" if label == "failed" else "■"
                sty = "bold red" if label == "failed" else "bold yellow"
                card.append(f"{icon} Task {label}  ", style=sty)
                card.append(ts, style="dim")
                log.write(card)
                brief = msg.content.replace("[failed: ", "").rstrip("]")[:200]
                if brief:
                    log.write(Text(brief, style="dim red" if label == "failed" else "dim yellow"))
            except Exception:
                brief = msg.content.replace("[failed: ", "").rstrip("]")[:120]
                conv.append_turn("assistant", f"Task {label}: {brief}")
            self.notify(
                f"Task {label}.",
                severity="warning",
                timeout=6,
            )

        elif msg.msg_type == "approval_request":
            approval_id = msg.metadata.get("approval_id", "")
            task_id     = msg.task_id
            # Flash the task in the left pane
            try:
                self.query_one("#tasks-pane", TaskListPane).mark_approval_pending(task_id)
            except NoMatches:
                pass
            # Ring terminal bell to draw attention
            import sys as _sys
            _sys.stdout.write("\a")
            _sys.stdout.flush()
            # Auto-focus the output pane on the task needing approval
            output = self.query_one("#output-pane", OutputPane)
            output.focused_task_id = task_id
            output.focus()
            if approval_id:
                from birdclaw.agent.approvals import approval_queue
                req = approval_queue.get(approval_id)
                if req:
                    output.show_approval(req)
            self.notify(
                f"Task needs approval — review in output pane",
                title=f"⚠ Approval required",
                severity="warning",
                timeout=8,
            )
            # Clear flash once resolved (poll on next tick via approval check)
            def _clear_flash_when_resolved(tid: str = task_id, aid: str = approval_id) -> None:
                from birdclaw.agent.approvals import approval_queue
                import time as _t
                _t.sleep(0.5)
                while True:
                    req = approval_queue.get(aid)
                    if req is None:
                        self.call_from_thread(
                            lambda t=tid: (
                                self.query_one("#tasks-pane", TaskListPane)
                                    .clear_approval_pending(t)
                            )
                        )
                        break
                    if req.is_expired():
                        self.call_from_thread(
                            lambda t=tid: (
                                self.query_one("#tasks-pane", TaskListPane)
                                    .clear_approval_pending(t)
                            )
                        )
                        break
                    _t.sleep(0.5)
            import threading as _th
            _th.Thread(target=_clear_flash_when_resolved, daemon=True).start()

        elif msg.msg_type == "approval_flash":
            # Non-destructive operation auto-approved:
            #   - Flash (highlight) the task row in the task list
            #   - Show a brief note in the chat pane ("permissions pending")
            #   - If the task is focused in output pane, show inline approval card
            task_id = msg.task_id
            tool    = msg.metadata.get("tool_name", "tool")
            desc    = (msg.content or "")[:60]
            # Flash task row in the task list (same mechanism as approval_request)
            try:
                self.query_one("#tasks-pane", TaskListPane).mark_approval_pending(task_id)
            except NoMatches:
                pass
            # Chat pane: brief pending note (not a full toast — just inline text)
            try:
                conv.append_turn(
                    "system",
                    f"[auto-approved] {tool}: {desc}",
                )
            except Exception:
                pass
            # Output pane: if this task is focused, mount a lightweight approval card
            output = self.query_one("#output-pane", OutputPane)
            if output.focused_task_id == task_id:
                try:
                    scroll = output.query_one("#output-scroll", VerticalScroll)
                    from birdclaw.tui.cards import ApprovalCard
                    flash_card = ApprovalCard(
                        approval_id="auto",
                        tool_name=tool,
                        description=f"[Auto-approved] {desc}",
                        expires_at=0.0,
                    )
                    scroll.mount(flash_card)
                    scroll.scroll_end(animate=False)
                    # Auto-dismiss after 4 seconds
                    def _dismiss(card=flash_card) -> None:
                        import time as _t
                        _t.sleep(4)
                        try:
                            self.call_from_thread(card.remove)
                        except Exception:
                            pass
                    import threading as _th
                    _th.Thread(target=_dismiss, daemon=True).start()
                except Exception:
                    pass
            # Clear task list flash after a brief moment
            def _clear(tid: str = task_id) -> None:
                import time as _t
                _t.sleep(3)
                try:
                    self.call_from_thread(
                        lambda t=tid: self.query_one("#tasks-pane", TaskListPane)
                        .clear_approval_pending(t)
                    )
                except Exception:
                    pass
            import threading as _thr
            _thr.Thread(target=_clear, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    BirdClawApp().run()


if __name__ == "__main__":
    run()
