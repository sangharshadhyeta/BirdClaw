"""LLM request scheduler — priority queue in front of the llama.cpp server.

Why this exists
---------------
llama.cpp runs with a fixed number of parallel KV-cache slots (--parallel N,
default 4 in BirdClaw).  Without a scheduler every agent thread calls
llm_client.generate() directly; when more than N threads are in-flight the
extras block inside httpx waiting for the server.  This gives equal treatment
to a background cron job and the user's interactive message — the interactive
reply can be delayed by however long the cron task takes.

The scheduler gates all generate() calls behind a semaphore of size N (read
from settings.llamacpp_parallel).  A single daemon worker drains a
PriorityQueue, ensuring higher-priority requests always reach the server first.

Priority levels (lower number = higher priority)
------------------------------------------------
  INTERACTIVE = 0   user typed a message → soul loop
  AGENT       = 1   active foreground task the user is watching
  BACKGROUND  = 2   active task not currently in the TUI focus
  CRON        = 3   scheduled skill run (background)
  MEMORY      = 4   dreaming / memorise / ingest workers

Usage
-----
Priority is set once per agent thread via context_vars:

    from birdclaw.tools.context_vars import set_llm_priority, LLMPriority
    set_llm_priority(LLMPriority.INTERACTIVE)

The scheduler reads it automatically inside generate().  No other code changes
needed at call sites — LLMClient.generate() calls scheduler.submit() internally.

Config
------
    BC_LLM_SCHEDULER_ENABLED=false   bypass the scheduler entirely (dev / testing)
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

class LLMPriority:
    INTERACTIVE = 0
    AGENT       = 1
    BACKGROUND  = 2
    CRON        = 3
    MEMORY      = 4


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class LLMScheduler:
    """Priority-ordered semaphore gate in front of the LLM server.

    submit(fn, priority) → Future[T]
        Enqueues fn.  Returns a Future that resolves when the call completes.
        The caller can block on future.result() to get the return value.
    """

    def __init__(self, slots: int) -> None:
        self._slots     = max(1, slots)
        self._semaphore = threading.Semaphore(self._slots)
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq       = 0               # tiebreaker — FIFO within same priority
        self._lock      = threading.Lock()
        self._enabled   = True

        # Single drain worker — more would risk exceeding the slot count
        self._worker = threading.Thread(
            target=self._drain, daemon=True, name="llm-scheduler"
        )
        self._worker.start()
        logger.debug("LLMScheduler started: %d slots", self._slots)

    # ── Public ────────────────────────────────────────────────────────────────

    def submit(self, fn: Callable[[], Any], priority: int = LLMPriority.AGENT) -> "Future[Any]":
        """Enqueue a generate() call and return a Future for its result."""
        fut: Future = Future()
        with self._lock:
            seq = self._seq
            self._seq += 1
        self._queue.put((priority, seq, fn, fut))
        return fut

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _drain(self) -> None:
        """Worker loop: pop item → acquire slot → call fn → release slot."""
        while True:
            try:
                priority, _seq, fn, fut = self._queue.get()
                # Acquire a server slot — blocks if all N slots are busy
                self._semaphore.acquire()
                # Run the LLM call in a fresh thread so the drain loop can
                # immediately pick up the next queued item (up to slot_count
                # items run concurrently, matching the server's parallel count).
                threading.Thread(
                    target=self._call,
                    args=(fn, fut),
                    daemon=True,
                    name=f"llm-slot-{priority}",
                ).start()
            except Exception as e:
                logger.error("scheduler drain error: %s", e)

    def _call(self, fn: Callable[[], Any], fut: "Future[Any]") -> None:
        """Execute fn, resolve future, then release the semaphore slot."""
        try:
            result = fn()
            if not fut.done():
                fut.set_result(result)
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
        finally:
            self._semaphore.release()


# ---------------------------------------------------------------------------
# Module singleton — created lazily so settings are loaded first
# ---------------------------------------------------------------------------

_scheduler: LLMScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> LLMScheduler:
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                from birdclaw.config import settings
                _scheduler = LLMScheduler(slots=settings.llamacpp_parallel)
    return _scheduler
