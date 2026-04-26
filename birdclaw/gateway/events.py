"""Lightweight event bus for non-blocking TUI notifications.

Currently used by the approval fast path to flash a toast when a
non-destructive tool is auto-approved (instead of blocking the agent).

The TUI picks these up via the gateway's push loop. If no gateway is
running (CLI / test mode), the events are simply logged.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class ApprovalFlashEvent:
    task_id: str
    tool_name: str
    description: str


# ---------------------------------------------------------------------------
# Simple thread-safe event queue
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_queue: deque[ApprovalFlashEvent] = deque(maxlen=100)
_listeners: list[Callable[[ApprovalFlashEvent], None]] = []


def emit_approval_flash(task_id: str, tool_name: str, description: str) -> None:
    """Emit a non-blocking approval flash event.

    Stored in the queue for the TUI to drain, and forwarded to any
    registered listeners (e.g. gateway push loop).
    """
    evt = ApprovalFlashEvent(task_id=task_id, tool_name=tool_name, description=description)
    with _lock:
        _queue.append(evt)
        listeners = list(_listeners)
    logger.debug("approval flash: %s — %s", tool_name, description[:60])
    for cb in listeners:
        try:
            cb(evt)
        except Exception as e:
            logger.debug("approval flash listener error: %s", e)


def drain_flash_events() -> list[ApprovalFlashEvent]:
    """Drain and return all pending flash events (called by TUI poll loop)."""
    with _lock:
        events = list(_queue)
        _queue.clear()
    return events


def register_listener(cb: Callable[[ApprovalFlashEvent], None]) -> None:
    """Register a callback invoked synchronously on each flash event."""
    with _lock:
        _listeners.append(cb)
