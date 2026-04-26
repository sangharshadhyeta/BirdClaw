"""Cross-process system notification queue backed by a JSONL file.

Any process (dream, cleanup, daemon) can write a notification.
The TUI drains it each poll cycle and shows toasts.

Usage (writer):
    from birdclaw.gateway.notify import push_notification
    push_notification("Dreaming complete — graph updated.", severity="information")

Usage (reader / TUI):
    from birdclaw.gateway.notify import drain_notifications
    for n in drain_notifications():
        self.notify(n["message"], title=n.get("title",""), severity=n.get("severity","information"))
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _notif_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "notifications.jsonl"


def push_notification(
    message: str,
    title: str = "",
    severity: str = "information",   # "information" | "warning" | "error"
) -> None:
    """Append a notification for the TUI to pick up."""
    record = {"message": message, "title": title, "severity": severity, "ts": time.time()}
    path = _notif_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("notify: write failed: %s", exc)


def drain_notifications() -> list[dict]:
    """Read and clear all pending notifications. Safe to call from TUI poll loop."""
    path = _notif_path()
    if not path.exists():
        return []
    try:
        with _lock:
            text = path.read_text()
            path.unlink(missing_ok=True)
    except Exception:
        return []
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return results
