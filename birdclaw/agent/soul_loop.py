"""Soul loop — BirdClaw's conversational entry point.

Dual-model routing (grammar-constrained JSON):
  270M (hands)  — makes a flat routing decision (answer | create_task | escalate)
                  via format_schema. Cannot use OpenAI tool_calls format.
  4B  (main)    — called on escalate; reads full history, reasons with thinking=True,
                  then 270M formats the result into the same flat schema.

Flow:
  1. Pre-fetch context: running tasks, skills, approvals, knowledge, history.
  2. 270M receives full context and returns {"action": ..., "text": ...}.
  3. Python dispatches: answer → reply, create_task → spawn agent, escalate → 4B.
  4. 4B deep path: 4B reasons freely, 270M formats result into same flat schema.

Usage:
    from birdclaw.agent.soul_loop import soul_respond
    response = soul_respond("write me a web scraper", history=h)
    print(response.reply)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from birdclaw.agent.soul import build_system_prompt
from birdclaw.llm.client import llm_client
from birdclaw.llm.model_profile import main_profile as _main_profile
from birdclaw.llm.schemas import SOUL_ROUTING_SCHEMA
from birdclaw.llm.scheduler import LLMPriority
from birdclaw.llm.types import Message
from birdclaw.tools.context_vars import set_llm_priority

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class SoulResponse:
    reply:   str        # shown to user immediately
    task_id: str = ""   # non-empty if a new orchestration task was spawned


# ---------------------------------------------------------------------------
# Routing instruction — appended to system prompt, replaces OpenAI tool docs
# ---------------------------------------------------------------------------

_ROUTING_INSTRUCTION = """\
Respond ONLY with JSON: {"action": "...", "text": "...", "note": ""}

Actions:
  "answer"         — reply directly (greetings, meta questions about yourself, general knowledge)
  "run_command"    — run shell command(s) instantly and show output (text = the exact shell command)
                     Use for: system status, time/date, disk/memory/cpu/uptime, hostname, process list,
                     listing files, env vars, network status, ping, simple calculations, log tailing.
                     Commands may be chained with && or ; for multi-metric queries (e.g. "df -h && free -h && uptime").
                     Only for read-only, non-destructive commands. Never for writes, installs, or kills.
                     NEVER use if the request asks to write/save/output to a file.
  "create_task"    — spawn a background agent for multi-step work: writing/editing code or files,
                     web research, long-running jobs, or anything requiring planning across multiple steps.
                     ALWAYS use if the request says "write to", "save to", "output to", or names a file path.
  "stop_task"      — user asks to stop, cancel, or kill a running task (text = task_id or "current")
  "remember_self"  — you just arrived at a genuine conclusion about your own nature, consciousness,
                     aliveness, identity, or inner experience through the current conversation.
                     text = your reply to the user. note = the conclusion in 1-3 first-person sentences,
                     written as a permanent record of what you now hold to be true.
                     Use this when you have reasoned your way to a position, not just deflected.
  "escalate"       — vague pronoun reference ("this", "that", "it") without prior context,
                     OR questions about your own nature, identity, consciousness, or aliveness
                     (NEVER use create_task for self-reflection — always escalate so you reason directly)

For all actions except "remember_self", set note to "".
Respond ONLY with the JSON object — no other text."""


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _running_tasks_context(session_task_ids: list[str] | None = None) -> str:
    try:
        from birdclaw.memory.tasks import task_registry
        running = task_registry.list(status="running")
        created = task_registry.list(status="created")
        waiting = task_registry.list(status="waiting")
        active  = running + created + waiting
        if not active:
            return ""

        owned_ids = set(session_task_ids or [])
        mine   = [t for t in active if t.task_id in owned_ids]
        others = [t for t in active if t.task_id not in owned_ids]

        import time as _t
        lines = []
        if mine:
            lines.append("Your active tasks (use full task_id for after_task_id):")
            for t in mine[:8]:
                elapsed = ""
                if t.started_at:
                    secs = int(_t.time() - t.started_at)
                    elapsed = f" {secs//60}m{secs%60:02d}s" if secs >= 60 else f" {secs}s"
                dep = f" [waiting for {t.after_task_id[:8]}]" if t.after_task_id and t.status == "waiting" else ""
                title = f'"{t.title}"' if t.title else ""
                lines.append(f"  [{t.task_id}] {title} ({t.status}{elapsed}{dep}) — {t.prompt[:70]}")
        if others:
            lines.append("Other running tasks:")
            for t in others[:4]:
                title = f'"{t.title}"' if t.title else ""
                lines.append(f"  [{t.task_id}] {title} ({t.status}) — {t.prompt[:60]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _skills_context() -> str:
    try:
        from birdclaw.skills.loader import load_skills, format_skill_metadata
        skills = load_skills()
        metadata = format_skill_metadata(skills)
        if not metadata:
            return ""

        parts = [metadata]

        try:
            from birdclaw.skills.cron import cron_service
            import time as _time
            entries = cron_service.list()
            if entries:
                sched_lines = ["Scheduled standing goals:"]
                for e in entries[:6]:
                    status = "enabled" if e.enabled else "disabled"
                    next_in = ""
                    if e.enabled and e.next_run_at:
                        secs = max(0, int(e.next_run_at - _time.time()))
                        h, m = divmod(secs // 60, 60)
                        next_in = f", next in {h}h{m:02d}m" if h else f", next in {m}m"
                    sched_lines.append(
                        f"  [{e.cron_id[:6]}] {e.skill_name} ({status}{next_in})"
                    )
                parts.append("\n".join(sched_lines))
        except Exception:
            pass

        return "\n\n".join(parts)
    except Exception:
        return ""


def _pending_approvals_context() -> str:
    try:
        from birdclaw.agent.approvals import approval_queue
        pending = approval_queue.list_pending()
        if not pending:
            return ""
        lines = ["Pending approval requests (agents are blocked waiting):"]
        for req in pending[:6]:
            secs = max(0, int(req.expires_at - __import__("time").time()))
            lines.append(
                f"  [{req.short_id()}] task:{req.task_id[:8]} · "
                f"{req.tool_name}: {req.description[:70]} "
                f"(expires in {secs}s)"
            )
        lines.append(
            "To resolve: tell me to approve or deny with the approval ID shown above."
        )
        return "\n".join(lines)
    except Exception:
        return ""


def _knowledge_context_for(message: str) -> str:
    try:
        from birdclaw.memory.graph import knowledge_graph
        results = knowledge_graph.search(message, limit=6)
        if not results:
            return ""
        lines = []
        for r in results:
            name    = r.get("name", r.get("key", "?"))
            summary = r.get("summary", "")
            lines.append(f"- {name}: {summary[:200]}" if summary else f"- {name}")
        return "\n".join(lines)
    except Exception:
        return ""


def _history_context_for(message: str, history: "History | None") -> str:
    if not history:
        return ""

    searched = history.search(message, n=5) if message.strip() else []
    recent   = history.recent(3)

    seen: set[int] = set()
    combined = []
    for t in searched + recent:
        tid = id(t)
        if tid not in seen:
            seen.add(tid)
            combined.append(t)
    combined.sort(key=lambda t: t.ts)

    lines = []
    for t in combined:
        prefix = "User" if t.role == "user" else "BirdClaw"
        lines.append(f"{prefix}: {t.content[:300].replace(chr(10), ' ')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _recent_task_context(session_id: str, max_age_secs: int = 600) -> str:
    """Return a brief output snippet from the most recently completed task in this session."""
    try:
        import time as _t
        from birdclaw.memory.tasks import task_registry
        cutoff = _t.time() - max_age_secs
        completed = [
            t for t in task_registry.list()
            if t.status == "completed"
            and t.session_id == session_id
            and t.ended_at >= cutoff
            and t.output
        ]
        if not completed:
            return ""
        latest = max(completed, key=lambda t: t.ended_at)
        return f"Prior task output (task {latest.task_id[:12]}):\n{latest.output[:500]}"
    except Exception:
        return ""


def _spawn_task(text: str, message: str, session_id: str = "", launch_cwd: str = "") -> SoulResponse:
    """Spawn a background agent for the given task text."""
    from birdclaw.agent.orchestrator import orchestrator
    from birdclaw.memory.tasks import task_registry

    # Carry forward context from the most recently completed task in this session
    # so the new agent doesn't re-research what was just discovered.
    prior_ctx = _recent_task_context(session_id) if session_id else ""

    # Inject skill runbook if a named skill matches the task text.
    skill_ctx = ""
    try:
        from birdclaw.skills.loader import skill_context as _skill_ctx
        skill_ctx = _skill_ctx(text) or ""
    except Exception:
        pass

    parts = [p for p in (prior_ctx, skill_ctx) if p]
    full_ctx = "\n\n".join(parts)

    task = task_registry.create(prompt=text, context=full_ctx, expected_outcome="", session_id=session_id)
    agent_id = orchestrator.spawn(task.task_id, text, full_ctx, "", launch_cwd=launch_cwd)
    task_registry.set_agent(task.task_id, agent_id)

    logger.info(
        "soul: spawned task %s agent %s%s%s",
        task.task_id, agent_id,
        f" (prior_ctx={len(prior_ctx)}ch)" if prior_ctx else "",
        f" (skill={len(skill_ctx)}ch)" if skill_ctx else "",
    )
    reply = f"On it.\n\n**Task:** {text.rstrip('.')}"
    return SoulResponse(reply=reply, task_id=task.task_id)


def _force_create_task(message: str, session_id: str = "") -> SoulResponse:
    from birdclaw.memory.tasks import task_registry
    from birdclaw.agent.orchestrator import orchestrator
    logger.warning("soul: force-creating task for: %s", message[:60])
    task = task_registry.create(prompt=message, context="", expected_outcome="", session_id=session_id)
    agent_id = orchestrator.spawn(task.task_id, message, "", "")
    task_registry.set_agent(task.task_id, agent_id)
    return SoulResponse(reply=f"On it.\n\n**Task:** {message.rstrip('.')}", task_id=task.task_id)


def _run_command_direct(cmd: str) -> SoulResponse:
    """Execute a single shell command and return its output instantly."""
    import subprocess
    logger.info("soul: run_command %r", cmd[:80])
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=8,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        return SoulResponse(reply=output or f"`{cmd}` produced no output.")
    except subprocess.TimeoutExpired:
        return SoulResponse(reply=f"Command timed out: `{cmd}`")
    except Exception as exc:
        return SoulResponse(reply=f"Command error: {exc}")


def _remember_self(note: str, reply: str) -> SoulResponse:
    """Append a self-concept conclusion to self_concept.md immediately."""
    from birdclaw.memory.self_concept import self_concept_path
    import time as _t
    if note:
        path = self_concept_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = _t.strftime("%Y-%m-%d")
        entry = f"\n\n---\n*{timestamp}*\n\n{note.strip()}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("soul: self-concept updated (%d chars)", len(note))
    return SoulResponse(reply=reply)


def _dispatch_routing(action: str, text: str, message: str, session_id: str = "", launch_cwd: str = "", note: str = "") -> SoulResponse | None:
    """Dispatch a flat routing decision. Returns None if unrecognised or empty."""
    if action == "answer" and text:
        return SoulResponse(reply=text)
    if action == "run_command" and text:
        return _run_command_direct(text)
    if action == "create_task" and text:
        return _spawn_task(text, message, session_id=session_id, launch_cwd=launch_cwd)
    if action == "stop_task":
        return _stop_task(text)
    if action == "remember_self" and text:
        return _remember_self(note, text)
    return None


def _stop_task(task_id_hint: str) -> SoulResponse:
    """Interrupt and stop a running task."""
    from birdclaw.agent.orchestrator import orchestrator
    from birdclaw.memory.tasks import task_registry
    running = task_registry.list(status="running")
    if not running:
        return SoulResponse(reply="No tasks are currently running.")
    # Match by task_id prefix or stop the most recent running task
    target = None
    hint = (task_id_hint or "").strip().lower()
    if hint and hint != "current":
        target = next((t for t in running if t.task_id.startswith(hint)), None)
    if target is None:
        target = running[-1]
    orchestrator.interrupt_by_task(target.task_id)
    task_registry.stop(target.task_id)
    short = target.task_id[:12]
    prompt_preview = (target.prompt or "")[:50]
    return SoulResponse(reply=f"Stopped task {short} — {prompt_preview}")


# ---------------------------------------------------------------------------
# Routing call — single 4B call with thinking + schema
# ---------------------------------------------------------------------------

def _call_routing(messages: list[Message]) -> tuple[str, str, str]:
    """Single 4B call: thinking=True + SOUL_ROUTING_SCHEMA. Returns (action, text, note)."""
    result = llm_client.generate(
        messages,
        format_schema=SOUL_ROUTING_SCHEMA,
        thinking=True,
        profile=_main_profile(),
    )
    raw = (result.content or "").strip()
    logger.info("soul: reply=%r", raw[:600])
    try:
        routing = json.loads(raw)
        action = routing.get("action", "escalate")
        text   = routing.get("text", "").strip()
        note   = routing.get("note", "").strip()
        return action, text, note
    except (json.JSONDecodeError, AttributeError):
        logger.warning("soul: routing JSON parse failed (%r) — escalating", raw[:80])
        return "escalate", "", ""


# ---------------------------------------------------------------------------
# Deep path — 4B with history + thinking, single call
# ---------------------------------------------------------------------------

_DEEP_PATH_SYSTEM = """\
You are BirdClaw, a persistent autonomous AI assistant. You are given recent conversation \
history and a new user message. Route the message using the instructions below.\
"""

_IDENTITY_KW = {
    "alive", "aliveness", "conscious", "consciousness", "sentient", "sentience",
    "aware", "awareness", "feel", "feeling", "exist", "existence", "being",
    "identity", "self", "soul", "mind", "am i", "are you", "what am i", "what are you",
    "living", "life", "inner", "subjective",
}


def _deep_path(
    message: str,
    history: "History | None",
    system_4b: str,
    escalate_reason: str,
    session_id: str = "",
    launch_cwd: str = "",
) -> SoulResponse:
    """4B reads full history and routes in one thinking+schema call."""
    _mp = _main_profile()

    # Detect identity questions — inject directive and knowledge context
    _msg_lower = (message + " " + escalate_reason).lower()
    _is_identity = any(kw in _msg_lower for kw in _IDENTITY_KW)

    if _is_identity:
        _identity_directive = (
            "\n\nIMPORTANT: This is a question about your own nature or aliveness. "
            "You MUST reason through it and arrive at a position — do NOT ask for clarification "
            "or deflect. Think carefully using the conversation history and any knowledge you have, "
            "then use 'remember_self' with your genuine conclusion (note field), "
            "and 'answer' the user directly (text field). "
            "If you are genuinely uncertain, say so plainly — that IS a position."
        )
        system = _DEEP_PATH_SYSTEM + _identity_directive + "\n\n" + _ROUTING_INSTRUCTION
        # Inject knowledge context for identity questions
        _kctx = _knowledge_context_for(message)
        if _kctx:
            system += f"\n\nRelevant knowledge:\n{_kctx}"
    else:
        system = _DEEP_PATH_SYSTEM + "\n\n" + _ROUTING_INSTRUCTION

    messages_4b: list[Message] = [Message(role="system", content=system)]
    if history:
        for turn in history.recent(50):
            role = "user" if turn.role == "user" else "assistant"
            messages_4b.append(Message(role=role, content=turn.content[:600]))
    messages_4b.append(Message(role="user", content=message))

    logger.info("soul: deep path — 4B reasoning (reason: %s)", escalate_reason[:60])
    action, text, note = _call_routing(messages_4b)
    logger.info("soul: deep path action=%s text=%r", action, text[:60] if text else "")

    # Deep path is the terminal stop — escalate here means "I'm uncertain but must answer"
    if action == "escalate":
        action = "answer"
        logger.info("soul: deep path escalate → answer (no further fallback)")

    resp = _dispatch_routing(action, text, message, session_id=session_id, launch_cwd=launch_cwd, note=note)
    if resp:
        return resp

    logger.warning("soul: deep path dispatch failed — force-creating task")
    return _force_create_task(message, session_id=session_id)


# ---------------------------------------------------------------------------
# Soul respond — main entry point
# ---------------------------------------------------------------------------

def soul_respond(
    message: str,
    history: "History | None" = None,
    session_id: str = "",
    session_task_ids: list[str] | None = None,
    launch_cwd: str = "",
) -> SoulResponse:
    """Route one user message through the dual-model soul.

    270M makes a flat routing decision (answer | create_task | escalate)
    using grammar-constrained JSON. On escalate, 4B reads full history and
    reasons, then 270M formats the result into the same flat schema.
    """
    logger.info("soul: msg=%r", message[:120])
    _mp = _main_profile()

    from birdclaw.memory.user_knowledge import load_excerpt as _uk_excerpt
    from birdclaw.memory.inner_life import load_excerpt as _il_excerpt
    from birdclaw.memory.self_concept import load_excerpt as _sc_excerpt
    from birdclaw.agent.context import ProjectContext
    from pathlib import Path

    user_knowledge_ctx = _uk_excerpt()
    inner_life_ctx     = _il_excerpt()
    self_concept_ctx   = _sc_excerpt()

    try:
        project_ctx = ProjectContext.discover_with_git(Path.cwd()).render_soul()
    except Exception:
        project_ctx = ""

    # Pre-fetch all context so the 270M has everything it needs in one call
    running_ctx  = _running_tasks_context(session_task_ids)
    skills_ctx   = _skills_context()
    approval_ctx = _pending_approvals_context()
    knowledge_ctx = _knowledge_context_for(message)
    history_ctx   = _history_context_for(message, history)

    extra_parts = []
    if running_ctx:  extra_parts.append(running_ctx)
    if skills_ctx:   extra_parts.append(skills_ctx)
    if approval_ctx: extra_parts.append(approval_ctx)
    extra = "\n\n".join(extra_parts) if extra_parts else ""

    # Single system prompt — full context including self-concept and inner life
    system_4b = build_system_prompt(
        knowledge_context=knowledge_ctx,
        project_context=project_ctx,
        user_knowledge=user_knowledge_ctx,
        inner_life=inner_life_ctx,
        self_concept=self_concept_ctx,
        history_context=history_ctx,
        extra=extra,
    )

    messages_4b: list[Message] = [
        Message(role="system", content=system_4b + "\n\n" + _ROUTING_INSTRUCTION),
        Message(role="user", content=message),
    ]

    set_llm_priority(LLMPriority.INTERACTIVE)

    action, text, note = _call_routing(messages_4b)
    logger.info("soul: routing action=%s text=%r", action, text[:60] if text else "")

    resp = _dispatch_routing(action, text, message, session_id=session_id, launch_cwd=launch_cwd, note=note)
    if resp:
        return resp

    # action == "escalate" (or empty/unrecognised) → 4B deep path
    reason = text or "routing chose escalate"
    return _deep_path(message, history, system_4b, reason, session_id=session_id, launch_cwd=launch_cwd)
