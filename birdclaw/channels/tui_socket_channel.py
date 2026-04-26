"""Socket-backed TUI channel — one instance per connected TUI process.

The daemon creates one of these per TUI connection.  Messages from the gateway
are serialised to JSON and written to the socket; if the write fails (broken
pipe) the delivery is silently dropped — the disconnect handler in
gateway/tui_socket.py will clean up the session.
"""

from __future__ import annotations

import json
import logging
import socket
import threading

from birdclaw.gateway.channel import Channel, OutgoingMessage

logger = logging.getLogger(__name__)


class TUISocketChannel(Channel):
    """Delivers gateway messages to a connected TUI process over a Unix socket."""

    owns_tasks = True  # tasks die when grace period expires after disconnect

    def __init__(self, session_id: str, conn: socket.socket) -> None:
        # channel_id must match the session_id so the gateway routes correctly.
        # session_id is e.g. "tui:a1b2c3d4"; channel_id matches.
        self._channel_id = session_id
        self._conn = conn
        self._lock = threading.Lock()

    @property
    def channel_id(self) -> str:
        return self._channel_id

    def deliver(self, msg: OutgoingMessage) -> None:
        payload = {
            "session_id": msg.session_id,
            "content":    msg.content,
            "msg_type":   msg.msg_type,
            "task_id":    msg.task_id,
            "metadata":   msg.metadata or {},
        }
        line = (json.dumps(payload) + "\n").encode()
        with self._lock:
            try:
                self._conn.sendall(line)
            except OSError as e:
                logger.debug("tui_socket_channel: send failed for %s: %s", self._channel_id, e)
