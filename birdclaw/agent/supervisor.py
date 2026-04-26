"""StepSupervisor — overlaps LLM generation with write/verify.

The subtask executor normally runs sequentially:
    write_item(N) → verify(N) → write_item(N+1) → verify(N+1) → ...

The supervisor pipelines work so that verification of item N happens
while the LLM is generating item N+1, hiding verification latency:

    write_item(N) → [submit_llm(N+1) to slot 2]
                 → verify(N)          ← runs in main thread
                 → collect_llm(N+1)   ← result is ready (or nearly)
                 → write_item(N+1)
                 → [submit_llm(N+2)]
                 ...

File ordering is preserved: item N is written to disk before item N+1's
LLM context is read, so file-as-memory correctness is maintained.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StepSupervisor:
    """Submit LLM calls to a background slot; collect results when ready."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="supervisor")
        self._pending: Future | None = None
        self._pending_tag: str = ""

    def submit(self, fn: Callable, *args: Any, tag: str = "") -> None:
        """Submit fn(*args) to the background slot.

        Only one pending call is supported. Collect the previous result
        before submitting again.
        """
        if self._pending is not None and not self._pending.done():
            logger.warning("[supervisor] submit called with pending uncollected — collecting first")
            self.collect()

        self._pending = self._pool.submit(fn, *args)
        self._pending_tag = tag
        logger.debug("[supervisor] submitted  tag=%r", tag)

    def collect(self) -> Any:
        """Block until the pending call completes and return its result.

        Returns None if no call was pending.
        """
        if self._pending is None:
            return None
        try:
            result = self._pending.result()
            logger.debug("[supervisor] collected  tag=%r", self._pending_tag)
            return result
        except Exception as e:
            logger.warning("[supervisor] call failed  tag=%r  error=%s", self._pending_tag, e)
            return None
        finally:
            self._pending = None
            self._pending_tag = ""

    def has_pending(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
