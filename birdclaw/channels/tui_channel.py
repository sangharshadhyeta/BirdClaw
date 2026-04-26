"""TUI channel — in-process gateway adapter for the Textual TUI.

The TUI submits messages via gateway.submit() and receives replies via
the deliver() callback, which uses Textual's call_from_thread() to push
updates safely to the UI thread.

Usage (in BirdClawApp.on_mount):
    from birdclaw.channels.tui_channel import TUIChannel
    from birdclaw.gateway.gateway import gateway

    self._tui_channel = TUIChannel()
    self._tui_channel.on_deliver(self._on_gateway_message)
    gateway.register(self._tui_channel)
    gateway.start()

Then on user input:
    from birdclaw.gateway.channel import IncomingMessage
    gateway.submit(IncomingMessage(channel_id="tui", user_id="local", content=text))
"""

from __future__ import annotations

import logging
from typing import Callable

from birdclaw.gateway.channel import Channel, OutgoingMessage

logger = logging.getLogger(__name__)


class TUIChannel(Channel):
    """In-process channel: delivers gateway messages via a registered callback."""

    _channel_id = "tui"

    def __init__(self) -> None:
        self._callback: Callable[[OutgoingMessage], None] | None = None

    @property
    def channel_id(self) -> str:
        return self._channel_id

    def on_deliver(self, callback: Callable[[OutgoingMessage], None]) -> None:
        """Register the delivery callback (called from gateway threads)."""
        self._callback = callback

    def deliver(self, msg: OutgoingMessage) -> None:
        """Called by the gateway. Forwards to the registered TUI callback."""
        if self._callback is None:
            logger.warning("tui_channel: no callback registered, dropping %s", msg.msg_type)
            return
        try:
            self._callback(msg)
        except Exception as e:
            logger.error("tui_channel: deliver callback raised: %s", e)
