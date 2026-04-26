"""Memorise phase — curated per-unit knowledge ingestion into the graph.

The model reads one content unit at a time and decides what is worth writing
to the knowledge graph. It is offered only the four graph tools plus `done`.

Content unit sources:
    page_store   — LLM-cleaned web pages produced by the condenser
    session_log  — stage_done (research), assistant_message

The page store is the primary source — condenser already cleaned the content.
Session log entries (research summaries, answers) are secondary.

Background worker:
    Runs post-task via a daemon thread. Pauses when any task is running
    (checked via task_registry). Resumes automatically when tasks complete.

    memorise_pause()   — called when a task starts
    memorise_resume()  — called when a task ends
    start_worker()     — called once at startup (idempotent)

Per-unit tracking: ~/.birdclaw/memory/memorised.json
    { "<source_key>": true, ... }
    Idempotent — re-runs skip already-processed units.

Manual run:
    from birdclaw.memory.memorise import run_memorise
    count = run_memorise()
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from birdclaw.config import settings

logger = logging.getLogger(__name__)

# Tools offered per memorise mini-loop
_MEMORISE_TOOL_NAMES = {"graph_search", "graph_add", "graph_relate", "done"}

# Maximum steps per content unit
_MAX_STEPS = 6

# Session log event types to process
_SESSION_UNIT_TYPES = {"stage_done", "assistant_message", "plan"}

# stage_done types worth indexing (research always; verify adds facts too)
_STAGE_DONE_TYPES = {"research", "verify"}

# Idle sleep between drain passes (seconds)
_IDLE_SLEEP = 5.0

# Poll interval when paused waiting for tasks to finish
_PAUSED_POLL = 2.0


# ---------------------------------------------------------------------------
# Pause / resume signal + priority queue
# ---------------------------------------------------------------------------

_paused = threading.Event()   # set = paused, clear = running
_paused.clear()               # start in running state

# Priority queue: session IDs that just finished and should be indexed ASAP.
# Worker drains this before doing the idle pass.
_priority_queue: queue.Queue[str] = queue.Queue()

# Event set when a new session is pushed — wakes the sleeping worker immediately
_wake_event = threading.Event()

_worker_started = False
_worker_lock    = threading.Lock()


def memorise_pause() -> None:
    """Pause the background memorise worker (call when a task starts)."""
    _paused.set()
    logger.debug("memorise: paused")


def memorise_resume() -> None:
    """Resume the background memorise worker (call when a task ends)."""
    _paused.clear()
    logger.debug("memorise: resumed")


def notify_session(session_id: str) -> None:
    """Signal that session_id just completed and should be indexed immediately.

    Safe to call from any thread. The worker will wake within _PAUSED_POLL
    seconds and drain this session before continuing the idle pass.
    """
    _priority_queue.put(session_id)
    _wake_event.set()
    logger.debug("memorise: priority session queued: %s", session_id)


# ---------------------------------------------------------------------------
# Tracking file
# ---------------------------------------------------------------------------

def _tracking_path() -> Path:
    return settings.data_dir / "memory" / "memorised.json"


def _load_tracking() -> dict[str, bool]:
    p = _tracking_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("memorised.json unreadable — starting fresh")
    return {}


def _save_tracking(seen: dict[str, bool]) -> None:
    p = _tracking_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Merge with on-disk version before writing so concurrent callers
    # (worker thread + synchronous run_memorise()) don't overwrite each other.
    if p.exists():
        try:
            on_disk = json.loads(p.read_text(encoding="utf-8"))
            for k, v in on_disk.items():
                seen.setdefault(k, v)
        except Exception:
            pass
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(seen, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Done schema
# ---------------------------------------------------------------------------

def _build_done_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that you have finished extracting knowledge from this content unit. "
                "Call when you have written everything useful, or when there is nothing worth storing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Brief note on what was extracted or why nothing was written."},
                },
                "required": ["reason"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Per-unit memorise loop
# ---------------------------------------------------------------------------

def _memorise_unit(text: str, source_label: str) -> int:
    """Run the memorise mini-loop for one content unit. Returns graph write count."""
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.scheduler import LLMPriority
    from birdclaw.llm.types import Message
    from birdclaw.tools.context_vars import set_llm_priority
    from birdclaw.tools.executor import execute
    from birdclaw.tools.registry import registry
    set_llm_priority(LLMPriority.MEMORY)

    graph_schemas = [
        t.to_compact_schema()
        for t in registry.all_tools()
        if t.name in _MEMORISE_TOOL_NAMES - {"done"}
    ]
    offered = graph_schemas + [_build_done_schema()]

    system = (
        "You are the memorise phase of BirdClaw. "
        "Read the content unit and write genuinely useful facts, entities, "
        "or relationships to the knowledge graph. "
        "Use graph_search first to avoid duplicates. "
        "Only store specific, reusable knowledge — skip errors, generic statements, transient data. "
        "Call done() when finished or when there is nothing worth storing."
    )

    messages = [
        Message(role="system", content=system),
        Message(role="user", content=f"Source: {source_label}\n\nContent:\n{text[:3000]}"),
    ]

    writes = 0
    for _ in range(_MAX_STEPS):
        result = llm_client.generate(messages, tools=offered, thinking=False)
        if not result.tool_calls:
            break
        tc = result.tool_calls[0]
        if tc.name == "done":
            logger.debug("memorise done: %s", tc.arguments.get("reason", ""))
            break
        if tc.name in ("graph_add", "graph_relate"):
            writes += 1
        obs = execute(tc)
        logger.debug("memorise: %s → %s", tc.name, obs[:120])
        messages = messages + [
            Message(role="assistant", content=result.content or ""),
            Message(role="tool", content=obs, tool_call_id=tc.id, name=tc.name),
        ]

    return writes


# ---------------------------------------------------------------------------
# Page store drain
# ---------------------------------------------------------------------------

def _drain_page_store(seen: dict[str, bool], stop_fn=None) -> int:
    """Process unmemorised page store entries one at a time. Returns count processed.

    stop_fn: optional callable → bool; drain stops after the current page if it
    returns True, so the caller can handle the interruption cleanly.
    """
    from birdclaw.memory import page_store

    processed = 0
    for entry in page_store.list_recent(n=100):
        key = f"page:{entry.url}"
        if seen.get(key):
            continue
        if _paused.is_set() or (stop_fn and stop_fn()):
            break

        source_label = f"page_store:{entry.url}"
        logger.info("memorise page: %s", entry.url[:80])
        try:
            writes = _memorise_unit(entry.cleaned, source_label)
            logger.info("memorise page %s → %d write(s)", entry.url[:60], writes)
        except Exception as e:
            logger.error("memorise page failed %s: %s", entry.url[:60], e)
            continue

        seen[key] = True
        processed += 1
        _save_tracking(seen)  # persist after each page so early stop is safe

    return processed


# ---------------------------------------------------------------------------
# Session log drain
# ---------------------------------------------------------------------------

def _unit_text(event_type: str, data: dict[str, Any]) -> str | None:
    if event_type == "stage_done":
        stage_type = data.get("stage_type", "")
        if stage_type not in _STAGE_DONE_TYPES:
            return None
        summary = data.get("summary", "").strip()
        goal    = data.get("goal", "").strip()
        if not summary:
            return None
        return f"[{stage_type} stage: {goal}]\n{summary}"

    if event_type == "assistant_message":
        content = data.get("content", "").strip()
        return f"[assistant answer]\n{content}" if len(content) >= 100 else None

    if event_type == "plan":
        outcome = data.get("outcome", "").strip()
        steps   = data.get("steps", [])
        if not outcome:
            return None
        steps_text = "\n".join(f"  - {s}" for s in steps) if steps else ""
        return f"[task plan]\nOutcome: {outcome}\n{steps_text}".strip()

    return None


def _drain_sessions(seen: dict[str, bool], session_id: str | None = None, stop_fn=None) -> int:
    """Process unmemorised session log entries. Returns count processed."""
    from birdclaw.memory.session_log import SessionLog

    processed = 0
    candidates = (
        [settings.sessions_dir / f"{session_id}.jsonl"]
        if session_id
        else sorted(settings.sessions_dir.glob("*.jsonl"))
    )

    for session_path in candidates:
        if not session_path.exists():
            continue
        sid = session_path.stem
        log = SessionLog.load(sid)

        for idx, event in enumerate(log.all_events()):
            key = f"session:{sid}/{idx}"
            if seen.get(key):
                continue
            if event.type not in _SESSION_UNIT_TYPES:
                seen[key] = True
                continue

            text = _unit_text(event.type, event.data)
            if text is None:
                seen[key] = True
                continue

            if _paused.is_set() or (stop_fn and stop_fn()):
                _save_tracking(seen)
                return processed

            source_label = f"session:{sid} event:{idx}"
            logger.info("memorise session unit %s", key)
            try:
                writes = _memorise_unit(text, source_label)
                logger.info("memorise session %s → %d write(s)", key, writes)
            except Exception as e:
                logger.error("memorise session %s failed: %s", key, e)
                continue

            # For research stage_done events, also run the full two-pass
            # proposition+entity extraction pipeline to build richer graph nodes.
            if event.type == "stage_done" and event.data.get("stage_type") == "research":
                try:
                    from birdclaw.memory.ingest import ingest_text
                    from birdclaw.memory.graph import knowledge_graph
                    ingest_text(text, source=source_label, graph=knowledge_graph)
                except Exception as ie:
                    logger.debug("memorise: ingest pipeline failed for %s: %s", key, ie)

            seen[key] = True
            processed += 1

        _save_tracking(seen)

    return processed


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _dream_reflect(seen: dict[str, bool]) -> None:
    """Dreaming pass: merge session_graph → knowledge_graph, then surface patterns.

    Two steps:
      1. Merge ephemeral session_graph into persistent knowledge_graph (+ save).
      2. One LLM call over recent knowledge nodes to surface patterns/insights
         as new [dream]-tagged nodes.
    """
    try:
        from birdclaw.memory.graph import knowledge_graph, session_graph

        # Step 1 — merge session_graph into knowledge_graph
        n_session = session_graph.node_count()
        if n_session > 0:
            knowledge_graph.merge_from(session_graph)
            knowledge_graph.save()
            logger.info("dream: merged %d session node(s) → knowledge graph", n_session)

        # Step 2 — LLM reflection over recent knowledge nodes
        from birdclaw.llm.client import llm_client
        from birdclaw.llm.scheduler import LLMPriority
        from birdclaw.llm.types import Message
        from birdclaw.tools.context_vars import set_llm_priority
        from birdclaw.tools.executor import execute
        from birdclaw.tools.registry import registry
        set_llm_priority(LLMPriority.MEMORY)

        graph_schemas = [
            t.to_compact_schema()
            for t in registry.all_tools()
            if t.name in {"graph_search", "graph_add", "graph_relate", "done"}
        ]

        # Use internal _graph attribute (NetworkX DiGraph)
        recent_nodes = list(knowledge_graph._graph.nodes(data=True))[-20:]
        if len(recent_nodes) < 3:
            return  # not enough to reflect on

        node_text = "\n".join(
            f"- {name}: {data.get('summary', data.get('description', ''))[:120]}"
            for name, data in recent_nodes
        )

        system = (
            "You are the dream reflection phase of BirdClaw. "
            "Look at these recent knowledge graph nodes and identify:\n"
            "  1. Patterns that repeat across multiple entries\n"
            "  2. Contradictions or outdated facts that need updating\n"
            "  3. Higher-level insights worth storing as new graph nodes\n\n"
            "Write only genuinely new insights — tag them with [dream] in the description. "
            "Use graph_add to store insights, graph_relate to link them. "
            "Call done() when finished. Be concise."
        )
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=f"Recent knowledge graph nodes:\n{node_text}"),
        ]

        for _ in range(8):
            result = llm_client.generate(messages, tools=graph_schemas, thinking=False)
            if not result.tool_calls:
                break
            tc = result.tool_calls[0]
            if tc.name == "done":
                break
            obs = execute(tc)
            messages = messages + [
                Message(role="assistant", content=result.content or ""),
                Message(role="tool", content=obs, tool_call_id=tc.id, name=tc.name),
            ]

        logger.info("memorise: dream reflection pass complete")

        # Synthesise raw task reflections into inner_life.md
        try:
            from birdclaw.memory.inner_life import update_from_reflections
            n = update_from_reflections()
            if n:
                logger.info("dream: inner_life updated from %d reflections", n)
        except Exception as ie:
            logger.warning("dream: inner_life update failed: %s", ie)

        # Retroactive user knowledge extraction from recent task outputs
        try:
            _extract_user_knowledge_from_tasks()
        except Exception as ue:
            logger.warning("dream: user knowledge extraction failed: %s", ue)

    except Exception as e:
        logger.warning("memorise: dream reflection failed: %s", e)


def _extract_user_knowledge_from_tasks() -> None:
    """Scan recent completed task prompts for user facts and remember them.

    Uses a fast heuristic: look for first-person statements about the user
    in task prompts, then ask the model to extract structured facts.
    Only processes tasks not yet seen (tracked via a simple stamp file).
    """
    from birdclaw.memory.tasks import task_registry
    from birdclaw.memory.user_knowledge import remember
    from birdclaw.config import settings
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.scheduler import LLMPriority
    from birdclaw.llm.types import Message
    from birdclaw.tools.context_vars import set_llm_priority
    import json as _json

    # Track which tasks have been processed
    stamp_path = settings.data_dir / "memory" / "uk_seen_tasks.json"
    seen: set[str] = set()
    if stamp_path.exists():
        try:
            seen = set(_json.loads(stamp_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    completed = [t for t in task_registry.list(status="completed") if t.task_id not in seen]
    if not completed:
        return

    # Build a digest of recent task prompts
    digest_lines = []
    for t in completed[-30:]:  # limit to 30 most recent unseen
        digest_lines.append(f"- {t.prompt[:120]}")
        seen.add(t.task_id)

    if not digest_lines:
        return

    set_llm_priority(LLMPriority.BACKGROUND)
    messages = [
        Message(role="system", content=(
            "Extract facts about the USER (not the AI) from these task prompts. "
            "Look for: role, occupation, what they're building, preferences, tools they use, "
            "recurring interests. Each fact on its own line as: FACT: <one sentence> | CAT: <facts|preferences|interests>. "
            "Only extract clear, explicit facts — no inferences. "
            "If there are no user facts, respond with NONE."
        )),
        Message(role="user", content="Task prompts:\n" + "\n".join(digest_lines)),
    ]

    result = llm_client.generate(messages, thinking=False)
    content = (result.content or "").strip()

    if content and content.upper() != "NONE":
        for line in content.splitlines():
            line = line.strip()
            if not line.startswith("FACT:"):
                continue
            parts = line.split("|", 1)
            fact = parts[0][5:].strip()
            category = "facts"
            if len(parts) > 1 and "CAT:" in parts[1]:
                category = parts[1].split("CAT:", 1)[1].strip().lower()
                if category not in ("facts", "preferences", "interests"):
                    category = "facts"
            if fact:
                remember(fact, category)

    # Save updated seen set
    try:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(_json.dumps(sorted(seen)), encoding="utf-8")
    except Exception:
        pass

    logger.info("dream: user knowledge extraction checked %d tasks", len(completed))


def _worker_body() -> None:
    """Daemon thread: drain page store + session log when no tasks are running."""
    logger.debug("memorise worker started")
    while True:
        if _paused.is_set():
            _wake_event.wait(timeout=_PAUSED_POLL)
            _wake_event.clear()
            continue

        # Also check task registry — pause if any task running
        try:
            from birdclaw.memory.tasks import task_registry
            if task_registry.list(status="running"):
                _wake_event.wait(timeout=_PAUSED_POLL)
                _wake_event.clear()
                continue
        except Exception:
            pass

        seen = _load_tracking()
        total = 0
        try:
            # Priority queue — drain immediate sessions first
            while not _priority_queue.empty():
                try:
                    sid = _priority_queue.get_nowait()
                except queue.Empty:
                    break
                if _paused.is_set():
                    break
                n = _drain_sessions(seen, session_id=sid)
                total += n
                if n:
                    _save_tracking(seen)
                    logger.info("memorise worker: priority session %s → %d unit(s)", sid, n)

            # Idle drain — pages + all sessions
            if not _paused.is_set():
                n_pages    = _drain_page_store(seen)
                n_sessions = _drain_sessions(seen)
                total += n_pages + n_sessions
                if n_pages + n_sessions:
                    _save_tracking(seen)

            if total:
                logger.info("memorise worker: %d unit(s) processed", total)
                # Dream reflection after a productive pass
                _dream_reflect(seen)

        except Exception as e:
            logger.error("memorise worker error: %s", e)

        # Sleep interruptibly so notify_session() wakes us up immediately
        _wake_event.wait(timeout=_IDLE_SLEEP)
        _wake_event.clear()


def start_worker() -> None:
    """Start the memorise background worker (idempotent — safe to call multiple times)."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_body, daemon=True, name="memorise-worker")
        t.start()
        _worker_started = True
        logger.debug("memorise worker thread started")


# ---------------------------------------------------------------------------
# Manual / CLI run
# ---------------------------------------------------------------------------

def run_memorise(session_id: str | None = None, stop_fn=None) -> int:
    """Drain unprocessed units synchronously (for CLI use or dream cycle).

    Args:
        session_id: Process only this session. None = all sessions.
        stop_fn:    Optional callable → bool. Drain stops after the current unit
                    if it returns True (e.g. a task just started).

    Returns:
        Total content units processed.
    """
    import birdclaw.tools.graph_tools  # noqa — register graph tools

    settings.ensure_dirs()
    seen = _load_tracking()
    n_pages    = _drain_page_store(seen, stop_fn=stop_fn)
    n_sessions = _drain_sessions(seen, session_id=session_id, stop_fn=stop_fn)
    _save_tracking(seen)
    return n_pages + n_sessions
