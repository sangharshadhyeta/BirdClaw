"""Channel abstraction — unified send/receive interface for all transports.

Every transport (TUI, HTTP/WebSocket, future app) implements Channel.
The gateway owns routing; channels own delivery.

IncomingMessage  — user → gateway
OutgoingMessage  — gateway → user (reply, task update, approval request)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
import time


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class IncomingMessage:
    channel_id: str    # "tui", "http", …
    user_id:    str    # "local", uuid, …
    content:    str
    timestamp:  float = field(default_factory=time.time)


OutgoingType = Literal[
    "reply",             # direct soul answer
    "task_started",      # background task spawned
    "task_complete",     # task finished successfully
    "task_failed",       # task failed
    "task_stopped",      # task interrupted
    "approval_request",  # agent is blocked, needs user decision
    "approval_flash",    # flash indicator that an approval is pending
]


@dataclass
class OutgoingMessage:
    session_id: str
    content:    str
    msg_type:   OutgoingType = "reply"
    task_id:    str          = ""
    metadata:   dict         = field(default_factory=dict)
    timestamp:  float        = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Channel ABC
# ---------------------------------------------------------------------------

class Channel(ABC):
    """A transport adapter. Receives from gateway via deliver(); submits to
    gateway via gateway.submit(IncomingMessage(...))."""

    # Set to False for channels where tasks should survive disconnects
    # (e.g. Telegram — network drops are transient, tasks keep running).
    # Set to True for session-bound channels (e.g. TUI) where a clean quit
    # or expired reconnect window should stop the session's tasks.
    owns_tasks: bool = True

    @property
    @abstractmethod
    def channel_id(self) -> str:
        """Unique identifier for this channel instance (e.g. 'tui', 'http')."""
        ...

    @abstractmethod
    def deliver(self, msg: OutgoingMessage) -> None:
        """Push an outgoing message to the user. Called from gateway threads —
        implementations must be thread-safe (use call_from_thread in TUI)."""
        ...
