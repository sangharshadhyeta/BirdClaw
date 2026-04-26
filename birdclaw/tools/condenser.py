"""Web content condenser — two-phase processing of raw web content.

Phase 1 — Sync fast-path (no LLM, instant):
    Strip remaining noise from already-cleaned text.
    Returns first ~1000 chars as an immediate context snippet.
    This is what goes into the agent's context for the current step.

Phase 2 — Async LLM cleaning (background thread):
    Full LLM pass: read clean text + optional task goal.
    Produces { "cleaned": "...", "notes": "..." }
      cleaned  — condensed markdown for page store (general, reusable)
      notes    — task-focused extract for context supplement (injected next step)
    Stored in page store. Notes surfaced via pending_notes queue.

The two phases decouple immediate context quality (fast, good enough) from
long-term storage quality (thorough, async). The agent never waits for LLM
cleaning; Ollama's inference capacity fills the gap between agent steps.

Usage (called by web.py):
    from birdclaw.tools.condenser import condense_async, drain_pending_notes
    immediate = condense_async(raw_text, url, source_tool="web_fetch")
    # → immediate snippet injected into context now
    # → LLM cleaning queued in background

    # At the start of each loop step:
    notes = drain_pending_notes()   # list of ready notes to inject
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

logger = logging.getLogger(__name__)

# Max chars kept in the immediate fast-path snippet
_FAST_PATH_CHARS = 1200

# Max chars fed to the LLM condenser (protect context window)
_LLM_INPUT_CHARS = 4000

# Max workers in the condenser thread pool (1 = serialised behind Ollama)
_MAX_WORKERS = 1


# ---------------------------------------------------------------------------
# Pending notes queue — notes ready to inject into agent context
# ---------------------------------------------------------------------------

@dataclass
class PendingNote:
    url:        str
    notes:      str
    source_tool: str
    ready_at:   float


_pending: Deque[PendingNote] = deque()
_pending_lock = threading.Lock()


def drain_pending_notes() -> list[PendingNote]:
    """Return and clear all notes that have been prepared since last drain.

    Called at the start of each agent loop step to inject ready notes.
    """
    with _pending_lock:
        items = list(_pending)
        _pending.clear()
    return items


def _push_pending(note: PendingNote) -> None:
    with _pending_lock:
        _pending.append(note)


# ---------------------------------------------------------------------------
# Condenser thread pool
# ---------------------------------------------------------------------------

_work_queue: Deque[tuple] = deque()
_work_lock   = threading.Lock()
_work_event  = threading.Event()
_worker_started = False
_worker_lock    = threading.Lock()


def _worker_loop() -> None:
    """Background worker: drain work queue, run LLM cleaning between agent steps."""
    while True:
        _work_event.wait()
        _work_event.clear()

        while True:
            with _work_lock:
                if not _work_queue:
                    break
                item = _work_queue.popleft()

            url, clean_text, task_goal, source_tool = item
            try:
                _run_llm_cleaning(url, clean_text, task_goal, source_tool)
            except Exception as e:
                logger.warning("condenser LLM cleaning failed for %s: %s", url, e)


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True, name="condenser-worker")
            t.start()
            _worker_started = True


def _enqueue(url: str, clean_text: str, task_goal: str, source_tool: str) -> None:
    _ensure_worker()
    with _work_lock:
        _work_queue.append((url, clean_text, task_goal, source_tool))
    _work_event.set()


# ---------------------------------------------------------------------------
# LLM cleaning pass
# ---------------------------------------------------------------------------

def _run_llm_cleaning(url: str, clean_text: str, task_goal: str, source_tool: str) -> None:
    """Run the LLM condenser, store result in page store, push notes to pending."""
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.types import Message
    from birdclaw.memory import page_store

    truncated = clean_text[:_LLM_INPUT_CHARS]

    goal_line = f"Current task goal: {task_goal}" if task_goal else ""
    prompt = f"""\
You are a content condenser. Read the web content below and produce a JSON object.

Rules:
- "cleaned": full condensed markdown — remove boilerplate, keep facts, code, data, structure. Max 1500 chars.
- "notes": if a task goal is given, extract only what is directly relevant to it. If no goal, copy the first 400 chars of cleaned. Max 500 chars.
- Output ONLY the JSON object. No explanation.

{goal_line}
Source: {url}

Content:
{truncated}

Output format:
{{"cleaned": "...", "notes": "..."}}"""

    result = llm_client.generate(
        [Message(role="user", content=prompt)],
        format_schema={"type": "json_object"},
        thinking=False,
    )

    parsed = _parse_json(result.content or "")
    if not parsed:
        # Fallback: use fast-path text as cleaned, first 400 chars as notes
        cleaned = clean_text[:1500]
        notes   = clean_text[:400]
    else:
        cleaned = parsed.get("cleaned", clean_text[:1500])
        notes   = parsed.get("notes", cleaned[:400])

    # Store in page store
    entry = page_store.put(url, cleaned, source_tool=source_tool)
    logger.debug("condenser stored page: %s (%d chars cleaned)", url, len(cleaned))

    # Push notes to pending queue for next agent step
    _push_pending(PendingNote(
        url=url,
        notes=notes,
        source_tool=source_tool,
        ready_at=time.time(),
    ))


def _parse_json(text: str) -> dict | None:
    import json
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Sync fast-path
# ---------------------------------------------------------------------------

def _fast_path(text: str) -> str:
    """Instant clean-up for immediate context injection. No LLM."""
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that look like pure navigation/menu noise
    lines = [
        ln for ln in text.splitlines()
        if len(ln.strip()) > 3 and not re.match(r"^[\s\W]{0,3}$", ln)
    ]
    return "\n".join(lines)[:_FAST_PATH_CHARS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def condense_async(
    raw_text: str,
    url: str,
    source_tool: str = "web_fetch",
) -> str:
    """Process web content for immediate context injection, queue LLM cleaning.

    Returns the fast-path snippet immediately (sync, no LLM).
    Queues a background LLM cleaning job that will populate the page store
    and push task-focused notes to drain_pending_notes().

    Args:
        raw_text:    Already HTML-stripped text (from BeautifulSoup in web.py).
        url:         Source URL (used as page store key).
        source_tool: "web_fetch" or "web_search".

    Returns:
        Immediate context snippet (first ~1200 chars of cleaned text).
    """
    from birdclaw.tools.context_vars import get_stage_goal

    task_goal = get_stage_goal()
    clean_text = _fast_path(raw_text)

    # Enqueue async LLM cleaning
    _enqueue(url, raw_text[:_LLM_INPUT_CHARS], task_goal, source_tool)

    return clean_text
