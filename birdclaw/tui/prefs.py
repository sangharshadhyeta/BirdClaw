"""TUI layout preferences — persisted to ~/.birdclaw/tui_prefs.json.

Stores the user's layout choices across restarts:
  chat_height_pct   int   30     conversation pane height %
  task_width_pct    int   30     task list pane width %
  layout_swapped    bool  False  True = output left, tasks right
  buddy_full        bool  False  True = buddy panel 80 cols (full), False = 44 cols (compact)

Usage:
    prefs = TuiPrefs.load()
    prefs.chat_height_pct = 40
    prefs.save()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BUDDY_COMPACT_WIDTH = 44
_BUDDY_FULL_WIDTH    = 80

_DEFAULTS: dict = {
    "chat_height_pct": 30,
    "task_width_pct":  30,
    "layout_swapped":  False,
    "buddy_full":      False,
    "theme":           "",
}


def _prefs_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "tui_prefs.json"


class TuiPrefs:
    """Lightweight JSON-backed TUI layout preferences."""

    def __init__(self, data: dict) -> None:
        self.chat_height_pct: int  = int(data.get("chat_height_pct", _DEFAULTS["chat_height_pct"]))
        self.task_width_pct:  int  = int(data.get("task_width_pct",  _DEFAULTS["task_width_pct"]))
        self.layout_swapped:  bool = bool(data.get("layout_swapped",  _DEFAULTS["layout_swapped"]))
        self.buddy_full:      bool = bool(data.get("buddy_full",      _DEFAULTS["buddy_full"]))
        self.theme:           str  = str(data.get("theme",            _DEFAULTS["theme"]))

    @classmethod
    def load(cls) -> "TuiPrefs":
        path = _prefs_path()
        try:
            if path.exists():
                return cls(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.debug("tui_prefs: load failed (%s) — using defaults", e)
        return cls({})

    def save(self) -> None:
        path = _prefs_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "chat_height_pct": self.chat_height_pct,
                "task_width_pct":  self.task_width_pct,
                "layout_swapped":  self.layout_swapped,
                "buddy_full":      self.buddy_full,
                "theme":           self.theme,
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("tui_prefs: save failed: %s", e)
