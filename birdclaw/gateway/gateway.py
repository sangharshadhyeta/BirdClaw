"""Gateway — the persistent core of BirdClaw.

Receives messages from any channel, routes them through the soul layer,
and pushes replies + async updates (task completions, approval requests)
back to the originating channel.

                 ┌──────────────────────────────────┐
  TUI channel ──►│                                  │
  HTTP channel──►│  Gateway.submit(IncomingMessage) │
  (future)       │     ↓                            │
                 │  soul_respond(session context)   │
                 │     ↓                            │
                 │  channel.deliver(OutgoingMessage)│
                 └──────────────────────────────────┘

Push worker (background thread, 0.5s poll):
    - Task completions → task_complete / task_failed / task_stopped
    - Approval requests → approval_request  (per-session, deduped)

Usage:
    from birdclaw.gateway.gateway import gateway
    gateway.register(tui_channel)
    gateway.start()
    gateway.submit(IncomingMessage(channel_id="tui", user_id="local", content="hi"))
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from birdclaw.gateway.channel import Channel, IncomingMessage, OutgoingMessage
from birdclaw.gateway.session_manager import SessionManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_PUSH_INTERVAL   = 0.5    # seconds between push worker ticks
_PUSH_NOTIF_TTL  = 120.0  # seconds before we forget a seen approval_id


def _summarise_for_chat(output: str, prompt: str = "") -> str:
    """Call the 270M hands model to produce a 1-2 sentence chat summary.

    Falls back to the first two non-empty lines if the LLM call fails.
    """
    try:
        from birdclaw.llm.client import llm_client
        from birdclaw.llm.model_profile import main_profile
        from birdclaw.llm.scheduler import LLMPriority
        from birdclaw.llm.types import Message
        from birdclaw.tools.context_vars import set_llm_priority
        set_llm_priority(LLMPriority.BACKGROUND)
        excerpt = output[:1200]
        system = (
            "You are BirdClaw. Based on the task result, write a direct 1-2 sentence answer. "
            "Just give the answer or confirmation — no 'you asked', no 'here is', no preamble. "
            "No markdown, no bullet points."
        )
        user = (
            f"Task: {prompt[:150]}\n\n"
            f"Result:\n{excerpt}\n\n"
            f"Direct 1-2 sentence answer:"
        )
        result = llm_client.generate(
            [Message(role="system", content=system), Message(role="user", content=user)],
            thinking=False,
            profile=main_profile(),
        )
        summary = (result.content or "").strip()
        if summary:
            return summary
    except Exception as exc:
        logger.warning("gateway: summary call failed: %s", exc)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " ".join(lines[:2])[:200] if lines else "(no output)"


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class Gateway:
    """Central hub: channels in, soul routes, replies + updates out."""

    def __init__(self) -> None:
        self._lock:            threading.Lock            = threading.Lock()
        self._channels:        dict[str, Channel]        = {}   # channel_id → Channel
        self._session_channel: dict[str, str]            = {}   # session_id → channel_id
        self._session_mgr:     SessionManager            = SessionManager()
        self._notified_approvals: dict[str, float]       = {}   # approval_id → time
        self._completed_tasks:    set[str]               = set()
        self._worker_started:  bool                      = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def register(self, channel: Channel) -> None:
        """Register a channel. Must be called before start()."""
        with self._lock:
            self._channels[channel.channel_id] = channel
        logger.info("gateway: registered channel %r", channel.channel_id)

    def start(self) -> None:
        """Boot the push worker and memorise background worker."""
        from birdclaw.memory.memorise import start_worker as start_memorise
        start_memorise()

        with self._lock:
            if self._worker_started:
                return
            self._worker_started = True

        self._reap_orphan_tasks()

        t = threading.Thread(target=self._push_worker, daemon=True, name="gateway-push")
        t.start()
        logger.info("gateway: started")

    # ── Inbound ───────────────────────────────────────────────────────────────

    def submit(self, msg: IncomingMessage) -> None:
        """Accept an incoming message from a channel. Non-blocking."""
        threading.Thread(
            target=self._process,
            args=(msg,),
            daemon=True,
            name=f"gw-{msg.channel_id[:4]}-{msg.user_id[:4]}",
        ).start()

    # ── Processing ────────────────────────────────────────────────────────────

    def _process(self, msg: IncomingMessage) -> None:
        """Route one message through the soul layer. Runs in a daemon thread."""
        from birdclaw.agent.soul_loop import soul_respond

        session = self._session_mgr.get_or_create(msg.channel_id, msg.user_id)
        self._session_mgr.touch(session.session_id)
        self._session_mgr.save_turn(session.session_id, "user", msg.content)

        # Record session → channel mapping for delivery
        with self._lock:
            self._session_channel[session.session_id] = msg.channel_id

        logger.info(
            "gateway: %s → session %s: %s",
            msg.channel_id, session.session_id, msg.content[:60],
        )

        try:
            response = soul_respond(
                msg.content,
                history=session.history,
                session_id=session.session_id,
                session_task_ids=list(session.task_ids),
                launch_cwd=session.launch_cwd,
            )
        except Exception as e:
            logger.exception("gateway: soul_respond failed for session %s", session.session_id)
            self._deliver(session.session_id, OutgoingMessage(
                session_id=session.session_id,
                content=f"[error: {e}]",
            ))
            return

        if response.task_id:
            self._session_mgr.add_task(session.session_id, response.task_id)

        self._session_mgr.save_turn(session.session_id, "assistant", response.reply)

        msg_type = "task_started" if response.task_id else "reply"
        self._deliver(session.session_id, OutgoingMessage(
            session_id=session.session_id,
            content=response.reply,
            msg_type=msg_type,
            task_id=response.task_id,
        ))

        # Cross-notify all TUI sessions when a task is started from a non-TUI channel
        # (e.g. test_socket, cron) so it appears in the conversation pane.
        if response.task_id and not msg.channel_id.startswith("tui"):
            self._cross_notify_tui(
                exclude_session=session.session_id,
                content=response.reply,
                msg_type="task_started",
                task_id=response.task_id,
            )

    # ── Push worker ───────────────────────────────────────────────────────────

    def _push_worker(self) -> None:
        """Background loop: push task completions, approval requests, and flash events."""
        while True:
            try:
                self._push_task_updates()
                self._push_approval_requests()
                self._push_approval_flashes()
                self._expire_old_notifications()
            except Exception as e:
                logger.error("gateway push worker error: %s", e)
            time.sleep(_PUSH_INTERVAL)

    def _push_approval_flashes(self) -> None:
        """Drain auto-approval flash events and push as informational toasts."""
        try:
            from birdclaw.gateway.events import drain_flash_events
            events = drain_flash_events()
        except Exception:
            return
        for evt in events:
            # Push to first active session (flash notifications are not task-specific)
            for session in self._session_mgr.all_sessions():
                self._deliver(session.session_id, OutgoingMessage(
                    session_id=session.session_id,
                    task_id=evt.task_id,
                    content=evt.description[:80],
                    msg_type="approval_flash",
                    metadata={"tool_name": evt.tool_name},
                ))
                break  # only send to first session

    def _push_task_updates(self) -> None:
        """Check session-owned tasks for completion and push updates."""
        from birdclaw.memory.tasks import task_registry

        for session in self._session_mgr.all_sessions():
            for task_id in list(session.task_ids):
                if task_id in self._completed_tasks:
                    continue
                task = task_registry.get(task_id)
                if task is None or task.status not in ("completed", "failed", "stopped"):
                    continue

                self._completed_tasks.add(task_id)
                if len(self._completed_tasks) > 10_000:
                    live = {t.task_id for t in task_registry.list()}
                    self._completed_tasks &= live

                if task.status == "completed":
                    content  = _summarise_for_chat(task.output or "", task.prompt or "")
                    msg_type = "task_complete"
                elif task.status == "failed":
                    content  = f"[failed: {task.output or 'unknown error'}]"
                    msg_type = "task_failed"
                else:
                    content  = f"[stopped] {task.output or ''}"
                    msg_type = "task_stopped"

                self._session_mgr.save_turn(session.session_id, "assistant", content)
                self._deliver(session.session_id, OutgoingMessage(
                    session_id=session.session_id,
                    content=content,
                    msg_type=msg_type,
                    task_id=task_id,
                ))
                # Cross-notify TUI sessions so completion appears in chat pane.
                if not session.channel_id.startswith("tui"):
                    self._cross_notify_tui(
                        exclude_session=session.session_id,
                        content=content,
                        msg_type=msg_type,
                        task_id=task_id,
                    )
                logger.info(
                    "gateway: pushed %s for task %s to session %s",
                    msg_type, task_id[:8], session.session_id,
                )

    def _push_approval_requests(self) -> None:
        """Push first-seen approval requests to the session that owns the task."""
        from birdclaw.agent.approvals import approval_queue

        pending = approval_queue.list_pending()
        if not pending:
            return

        # Build task_id → session_id map from all sessions
        task_to_session: dict[str, str] = {}
        for session in self._session_mgr.all_sessions():
            for tid in session.task_ids:
                task_to_session[tid] = session.session_id

        for req in pending:
            if req.approval_id in self._notified_approvals:
                continue
            self._notified_approvals[req.approval_id] = time.time()

            session_id = task_to_session.get(req.task_id)
            if session_id is None:
                continue   # task not owned by any known session; TUI fallback handles it

            short    = req.short_id()
            secs_left = max(0, int(req.expires_at - time.time()))
            content = (
                f"Task `{req.task_id[:8]}` needs approval ({secs_left}s to respond):\n"
                f"{req.tool_name}: {req.description[:120]}\n\n"
                f"Reply: /approve {short} allow · /approve {short} always · "
                f"/approve {short} deny"
            )
            self._deliver(session_id, OutgoingMessage(
                session_id=session_id,
                content=content,
                msg_type="approval_request",
                task_id=req.task_id,
                metadata={"approval_id": req.approval_id, "short_id": short},
            ))

    def _expire_old_notifications(self) -> None:
        cutoff = time.time() - _PUSH_NOTIF_TTL
        with self._lock:
            expired = [aid for aid, ts in self._notified_approvals.items() if ts < cutoff]
            for aid in expired:
                del self._notified_approvals[aid]

    # ── Startup cleanup ───────────────────────────────────────────────────────

    def _reap_orphan_tasks(self) -> None:
        """Mark any tasks stuck in running/created with no live agent as failed.

        These are orphans from a previous crashed or killed session. Without this,
        they show forever in the Active task list.
        """
        try:
            from birdclaw.memory.tasks import task_registry
            from birdclaw.agent.orchestrator import orchestrator
            live_agents = set(orchestrator.running_agents())
            reaped = 0
            for task in task_registry.list(status="running") + task_registry.list(status="created"):
                if task.agent_id and task.agent_id in live_agents:
                    continue  # genuinely running
                task_registry.fail(task.task_id, reason="orphaned — process restarted")
                # Silence the push worker — these are old tasks, don't notify the user
                self._completed_tasks.add(task.task_id)
                reaped += 1
            if reaped:
                logger.info("gateway: reaped %d orphan task(s) on startup", reaped)
        except Exception as e:
            logger.warning("gateway: orphan reap failed: %s", e)

    # ── Delivery ──────────────────────────────────────────────────────────────

    def _cross_notify_tui(
        self,
        exclude_session: str,
        content: str,
        msg_type: str,
        task_id: str,
    ) -> None:
        """Deliver a message to every registered TUI session except exclude_session."""
        for session in self._session_mgr.all_sessions():
            if session.channel_id.startswith("tui") and session.session_id != exclude_session:
                self._deliver(session.session_id, OutgoingMessage(
                    session_id=session.session_id,
                    content=content,
                    msg_type=msg_type,
                    task_id=task_id,
                ))

    def _deliver(self, session_id: str, msg: OutgoingMessage) -> None:
        with self._lock:
            channel_id = self._session_channel.get(session_id)
            if not channel_id:
                # Fall back to session record (survives daemon restart)
                sess = self._session_mgr.get(session_id)
                channel_id = sess.channel_id if sess else None
            channel = self._channels.get(channel_id) if channel_id else None
        if channel is None:
            logger.warning("gateway: no channel for session %s", session_id)
            return
        try:
            channel.deliver(msg)
        except Exception as e:
            logger.error("gateway: deliver failed for %s: %s", session_id, e)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

gateway = Gateway()
