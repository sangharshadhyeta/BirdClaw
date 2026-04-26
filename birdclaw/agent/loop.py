"""Agent loop — plan → stage queue → execute → answer.

Design principles:
  A1 — one tool call processed per turn
  A2 — only tools for the current stage type are shown
  A3 — format adapter normalises all response formats
  A5 — per-step message carries outcome + progress + current goal every turn
  A7 — compact schemas only

Flow
----
  PRE-LOOP  planner.generate_plan() — HANDS model format call → { outcome, stages }
            Does not consume a step slot. Falls back to free-tool mode on failure.

  LOOP      Pop stage from queue → inject GraphRAG knowledge context for stage goal
            → execute until stage advances → record summary → pop next.

  Format stages  (write_code, write_doc, edit_file):
    HANDS model handles schema-constrained output. MAIN model thinks freely.
    thinking=False in format stages — llama.cpp constraint.

  Tool-call stages (research, verify, reflect):
    MAIN model with thinking enabled when think tool is offered.
    - research: ends when model calls think()
    - verify:   ends when bash() returns no error
    - reflect:  ends when model calls think()

  Queue exhausted → forced answer() with outcome + completed summaries.

State separation
----------------
  GraphRAG    — knowledge only: research findings, entities, code facts
  session_log — orchestration: plan decisions, stage summaries, tool calls
  in-memory   — live stage progress (completed_stages list within _run_loop)

Per-step context (A5)
---------------------
  "[3/8] Outcome: <criteria> | Done: <s1> → <s2> | Now (research): <goal>"
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from birdclaw.agent.budget import (
    REQUEST_BUDGET_SCHEMA as _REQUEST_BUDGET_SCHEMA,
    historical_budget as _historical_budget,
    log_stage as _log_stage_history,
)
from birdclaw.agent.planner import (
    FORMAT_STAGE_TYPES as _FORMAT_STAGE_TYPES,
    THINK_ADVANCES_TYPES as _THINK_ADVANCES_TYPES,
    REFLECT_GATE_TYPES as _REFLECT_GATE_TYPES,
    STAGE_TYPE_TOOLS as _STAGE_TYPE_TOOLS,
    generate_plan as _generate_plan,
    rewrite_output_paths as _rewrite_output_paths,
    planning_context as _planning_context,
    answering_context as _answering_context,
    parse_format_response as _parse_format_response,
    reflect_on_stage as _reflect_on_stage,
    tools_for_stage as _tools_for_stage,
    tools_for_step as _tools_for_step,
    infer_stage_type as _infer_stage_type,
)
from birdclaw.agent.prompts import ANSWER_SCHEMA, CONTROL_TOOLS, SYSTEM, dynamic_context
from birdclaw.agent.router import select
from birdclaw.agent import subtask_executor as _subtask_executor
from birdclaw.config import settings
from birdclaw.llm.client import llm_client
from birdclaw.llm.model_profile import main_profile as _main_profile
from birdclaw.llm.schemas import EDIT_PATCH_SCHEMA as _EDIT_PATCH_SCHEMA
from birdclaw.llm.scheduler import LLMPriority
from birdclaw.llm.types import Message, ToolCall
from birdclaw.memory.compact import CompactionConfig, compact, should_compact
from birdclaw.tools import bash, code_index, files, graph_tools, skills, web  # noqa: F401 — triggers registration
import birdclaw.tools.task_tools  # noqa: F401 — registers search_tasks / get_task_output
from birdclaw.tools.context_vars import get_llm_priority, set_llm_priority
from birdclaw.tools.executor import execute
from birdclaw.tools.registry import registry

if TYPE_CHECKING:
    from birdclaw.memory.session_log import SessionLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# O2: Fuzzy match helper for edit_file "old text not found" retries
# ---------------------------------------------------------------------------

def _fuzzy_match_hint(old_text: str, content: str, window: int = 120) -> str:
    """Find the closest substring in content to old_text using difflib.

    Returns a one-line hint string: "Closest match found: '<snippet>'\n"
    or an empty string if nothing useful is found.
    """
    import difflib
    needle = old_text.strip()
    if not needle or len(needle) < 10:
        return ""
    best_ratio = 0.0
    best_snippet = ""
    step = max(10, len(needle) // 4)
    for i in range(0, max(1, len(content) - len(needle) + 1), step):
        chunk = content[i: i + len(needle) + window]
        ratio = difflib.SequenceMatcher(None, needle, chunk, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_snippet = chunk[: len(needle) + 30].replace("\n", "\\n")
    if best_ratio > 0.5:
        return f"Closest match in file (ratio={best_ratio:.2f}): '{best_snippet[:120]}'\n"
    return "No similar text found — the text may have already been changed.\n"


# Budget helpers, planner, and stage type constants are imported at the top.
# _REQUEST_BUDGET_SCHEMA, _historical_budget, _log_stage_history → budget.py
# _generate_plan, _reflect_on_stage, _tools_for_stage, etc. → planner.py

# Goals that should NOT trigger a stall-guard web_search because they don't
# need new data — they synthesise what's already in context.
_SYNTHESIS_KW: frozenset[str] = frozenset({
    "summarise", "summarize", "synthesize", "synthesise",
    "combine", "consolidate", "outline", "structure",
    "digest", "integrate", "collate",
    "review", "analyse", "analyze", "retrieved", "gathered",
    "findings", "compile", "organize", "organise",
    "ask user", "user input", "request from user", "prompt user",
    "gather user", "collect user", "from user", "user provide",
})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: int = 0
    thinking: list[str] = field(default_factory=list)


# Context builders and format response parser live in planner.py


def _append_to_notes(notes_path: str, tool_name: str, args: dict, result: str) -> None:
    """Append a tool result entry to the task notes file for future search_relevant lookups."""
    try:
        label = args.get("query") or args.get("url") or args.get("reasoning", "")[:60]
        header = f"## [{tool_name}] {label[:80]}" if label else f"## [{tool_name}]"
        with open(notes_path, "a", encoding="utf-8") as _f:
            _f.write(f"\n{header}\n{result[:500]}\n")
        logger.debug("[notes] appended  tool=%s  path=%s  chars=%d", tool_name, notes_path, len(result[:500]))
    except Exception:
        pass


def _format_write(path_str: str, content: str, append: bool = False) -> str:
    from birdclaw.agent.subtask_executor import _resolve_output_path
    resolved = _resolve_output_path(path_str)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(resolved, mode, encoding="utf-8") as f:
        f.write(content)
    action = "Appended" if append else "Created"
    return f"{action} {resolved} ({len(content)} chars)"


# Stage type system, tool selection, plan generation → planner.py


def _build_messages(
    question: str,
    history: list[Message] | None,
    extra_system: str | None,
) -> list[Message]:
    system_content = (extra_system + "\n\n" if extra_system else "") + SYSTEM
    if history:
        if history[0].role != "system":
            history = [Message(role="system", content=system_content)] + history
        else:
            history[0].content = system_content
        if history[-1].content != question or history[-1].role != "user":
            history.append(Message(role="user", content=question))
        return history
    return [
        Message(role="system", content=system_content),
        Message(role="user", content=question),
    ]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

_MAX_THINK_STEPS = 3   # nudge model away from think after this many consecutive calls

# Reflection gate → planner.py (_reflect_on_stage)


def _parse_edit_target(goal: str, last_written_path: str) -> str:
    """Extract a file path from an edit_file stage goal, or fall back to last_written_path."""
    # Look for an absolute path in the goal text
    m = _re.search(r"(/[\w./\-]+\.\w+)", goal)
    if m:
        return m.group(1)
    return last_written_path


def _run_loop(
    messages: list[Message],
    question: str,
    session_log: "SessionLog | None",
    interrupt_event: "threading.Event | None" = None,
    write_dir: str = "",
    skill_hint: str = "",
) -> tuple[str, list[str], int, list[str], list[dict]]:
    """Run the agent loop. Returns (answer, sources, steps_taken, thinking_traces, completed_stages)."""
    # Default to AGENT priority — callers may override via set_llm_priority() before
    # entering this function (e.g. cron sets CRON, soul sets INTERACTIVE).
    if get_llm_priority() == LLMPriority.INTERACTIVE:
        # Soul already bumped priority; keep it for the first tool call then drop to AGENT
        set_llm_priority(LLMPriority.AGENT)

    thinking_traces: list[str] = []
    max_steps = settings.max_agent_steps

    # Inject planning-stage memory context as a user message — NOT into the system
    # message. Modifying the system message busts the llama.cpp KV cache on every
    # call (7b: SYSTEM_PROMPT_DYNAMIC_BOUNDARY — static system = cache-stable).
    mem_ctx = _planning_context(question, session_log)
    if mem_ctx:
        messages.insert(1, Message(role="user", content=f"[Context]\n{mem_ctx}"))
        messages.insert(2, Message(role="assistant", content="Understood."))

    # --- Pre-loop: generate plan ---
    outcome, stage_queue = _generate_plan(question, session_log, skill_hint=skill_hint)

    # Rewrite generic 'output.md' / 'output.py' to task-specific filenames so
    # concurrent tasks don't collide on the same file.
    try:
        from birdclaw.tools.context_vars import get_task_id as _get_tid_plan
        _plan_task_id = _get_tid_plan()
    except Exception:
        _plan_task_id = ""
    if _plan_task_id:
        try:
            import re as _re
            from birdclaw.memory.tasks import task_registry as _tr
            _task_title = (_tr.get(_plan_task_id) or object()).__dict__.get("title", "") or ""
            _task_slug = _re.sub(r"[^a-z0-9]+", "_", _task_title.lower()).strip("_")[:40] or _plan_task_id[:12]
        except Exception:
            _task_slug = _plan_task_id[:12]
        stage_queue = _rewrite_output_paths(stage_queue, _task_slug)
    else:
        _task_slug = f"task_{int(__import__('time').time())}"
    def _tui_add_phase(task_id: str, goal: str) -> None:
        """Register a dynamically-inserted stage with the task registry so TUI shows it."""
        if not task_id:
            return
        try:
            from birdclaw.memory.tasks import task_registry as _tr_dyn
            _tr_dyn.add_phase_after_current(task_id, goal)
        except Exception:
            pass

    completed_stages: list[dict] = []  # {"type", "goal", "summary"} — in-memory only
    last_written_path: str = ""        # most recently written file path (v2 learning)
    edit_target_path: str = ""         # file being edited in an edit_file stage
    edit_not_found: int = 0            # consecutive "old text not found" retries

    # Per-stage budget state — reset by _pop_next_stage on each stage advance
    steps_this_stage: int = 0
    current_stage_budget: int = 0

    if outcome:
        stage_summary = ", ".join(f"{s['type']}:{s['goal'][:20]}" for s in stage_queue)
        messages.append(Message(
            role="assistant",
            content=f"Plan: outcome={outcome!r}  stages=[{stage_summary}]",
        ))

    # Mutable stage state
    current_stage: dict | None = None
    write_loop_initialized = False
    think_count = 0
    answering_mode = False   # True once completion message injected; restrict to answer only
    stage_started_at: float = 0.0
    sources: list[str] = []   # collected from web tool results
    stage_real_calls: int = 0  # non-think tool calls in current research/reflect stage
    consecutive_no_tool: int = 0  # consecutive steps with no tool call during a stage
    deepen_counts: dict[str, int] = {}  # per stage-type: how many times gate said "deepen"
    _last_think_reasoning: str = ""  # O6: detect repeated think() reasoning (research loop)

    def _pop_next_stage() -> dict | None:
        """Pop next stage, reset message history, inject GraphRAG context."""
        nonlocal write_loop_initialized, think_count, stage_started_at, stage_real_calls
        nonlocal steps_this_stage, current_stage_budget, consecutive_no_tool, _last_think_reasoning
        if not stage_queue:
            return None
        if interrupt_event and interrupt_event.is_set():
            stage_queue.clear()
            return None
        cfg = stage_queue.pop(0)
        write_loop_initialized = False
        think_count = 0
        stage_started_at = _time.time()
        stage_real_calls = 0
        steps_this_stage = 0
        consecutive_no_tool = 0
        _last_think_reasoning = ""
        current_stage_budget = max(1, cfg.get("budget") or _historical_budget(cfg["type"]))
        logger.info("stage start: %s — %s (budget=%d)", cfg["type"], cfg["goal"], current_stage_budget)
        if session_log:
            session_log.stage_start(cfg["type"], cfg["goal"])

        # Stage reset: rebuild messages from scratch on every stage transition.
        # Prior stage summaries are in completed_stages — no need to keep raw history.
        # First stage keeps the original context (question + workspace + plan).
        if completed_stages:
            try:
                done_str = " → ".join(s["goal"][:40] for s in completed_stages)
                stage_ctx = (
                    f"Task: {question}\n"
                    f"Outcome required: {outcome}\n"
                    f"Completed: {done_str}\n"
                    f"Now: [{cfg['type']}] {cfg['goal']}"
                )
                # For write stages continuing a file — inject last 20 lines so the
                # model can continue from exactly where the prior stage left off.
                if cfg["type"] in ("write_doc", "write_code") and last_written_path:
                    try:
                        _fc = Path(last_written_path).read_text(encoding="utf-8", errors="replace")
                        _tail_lines = _fc.splitlines()[-20:]
                        if _tail_lines:
                            stage_ctx += (
                                f"\n\n[{last_written_path} — last {len(_tail_lines)} lines]\n"
                                + "\n".join(_tail_lines)
                            )
                    except OSError:
                        pass
                _old = len(messages)
                messages[:] = [
                    Message(role="system", content=SYSTEM),
                    Message(role="user",   content=stage_ctx),
                ]
                logger.info("stage reset: %d → %d messages for [%s]", _old, len(messages), cfg["type"])
            except Exception as _err:
                logger.warning("stage reset failed (%s) — keeping existing messages", _err)

        # GraphRAG knowledge context for this stage's specific goal
        try:
            from birdclaw.memory.retrieval import retrieve
            ctx = retrieve(cfg["goal"])
            if ctx:
                logger.info("[graphrag] retrieved  stage=%s  goal=%r  chars=%d", cfg["type"], cfg["goal"][:50], len(ctx))
                messages.append(Message(
                    role="user",
                    content=f"[knowledge context for '{cfg['goal']}'] {ctx}",
                ))
            else:
                logger.debug("[graphrag] miss  stage=%s  goal=%r", cfg["type"], cfg["goal"][:50])
        except Exception:
            pass
        return cfg

    def _complete_stage(summary: str = "") -> None:
        nonlocal current_stage
        if current_stage:
            duration_ms = int((_time.time() - stage_started_at) * 1000) if stage_started_at else 0
            record = {
                "type": current_stage["type"],
                "goal": current_stage["goal"],
                "summary": summary or current_stage["goal"],
                "duration_ms": duration_ms,
            }
            completed_stages.append(record)
            # Persist empirical step count for future _historical_budget queries
            _log_stage_history(current_stage["type"], steps_this_stage, current_stage["goal"])
            # Log summary to session (orchestration — not GraphRAG)
            if session_log:
                _stage_ok = not any(w in summary.lower() for w in ("partial", "not found", "error", "failed", "skipping", "skip"))
                session_log.stage_done(
                    current_stage["type"],
                    current_stage["goal"],
                    summary,
                    duration_ms=duration_ms,
                    ok=_stage_ok,
                )
            logger.info("stage done: %s — %s", current_stage["type"], summary[:60])

            # Phase 9 — research ingest loop injection:
            # Immediately push research findings into the knowledge graph so
            # subsequent stages in this same task can retrieve them via GraphRAG.
            # The memorise worker does richer extraction post-session; this gives
            # instant searchability at zero extra LLM calls.
            if current_stage["type"] == "research" and summary:
                try:
                    from birdclaw.memory.graph import knowledge_graph
                    from birdclaw.tools.context_vars import get_task_id as _get_tid_inj
                    _inj_tid = _get_tid_inj() or "task"
                    knowledge_graph.upsert_node(
                        name=current_stage["goal"][:80],
                        node_type="fact",
                        summary=summary[:600],
                        sources=[f"research:{_inj_tid}"],
                    )
                    logger.info("[graphrag] research injected  goal=%r  chars=%d",
                                current_stage["goal"][:50], len(summary))
                except Exception as _ie:
                    logger.debug("[graphrag] research inject failed: %s", _ie)

        current_stage = None

    # Pop the first stage
    current_stage = _pop_next_stage()

    # Free-tool schemas: used when there are no stages (plan failed or simple Q&A).
    # router.select() picks the most relevant tools for this specific question
    # rather than offering the same fixed set every time.
    _router_tools = select(question)
    free_tool_schemas = CONTROL_TOOLS + [t.to_compact_schema() for t in _router_tools]

    # Write directory hint for format-mode stages — always the launch cwd.
    # config.py adds cwd to workspace_roots automatically, so it is always allowed.
    if write_dir:
        _write_dir = str(Path(write_dir).resolve())
    else:
        from birdclaw.config import settings as _settings
        _write_dir = str(_settings.workspace_dir.resolve())

    # Per-task subfolder — keeps notes, output files, and artifacts off the cwd root.
    #   {task_slug}/notes.md       ← research notes (search results + think reasoning)
    #   {task_slug}/{task_slug}.md  ← produced document (write_doc fallback)
    #   {task_slug}/{task_slug}.py  ← produced code    (write_code fallback)
    _task_dir        = Path(_write_dir) / _task_slug
    _task_notes_path = str(_task_dir / "notes.md")

    # 7b: inject date + dirs as a user message (not into system — keeps SYSTEM cache-stable)
    _dyn = dynamic_context(write_dir=_write_dir, task_dir=str(_task_dir))
    messages.insert(1, Message(role="user", content=f"[{_dyn}]"))
    messages.insert(2, Message(role="assistant", content="Noted."))
    logger.info("[init] task=%s  dir=%s  steps=%d", _task_slug, _write_dir, max_steps)

    for step in range(1, max_steps + 1):

        # Interrupt check — soul layer or user requested stop
        if interrupt_event and interrupt_event.is_set():
            logger.info("loop interrupted at step %d", step)
            summary = "\n".join(
                f"- {s['type']}: {s['summary']}" for s in completed_stages
            ) or "No stages completed."
            return (
                f"Task interrupted at step {step}.\nCompleted so far:\n{summary}",
                [], step, thinking_traces, completed_stages,
            )

        # ── Per-stage budget tracking ─────────────────────────────────────────
        _budget_warn = False
        if current_stage and not answering_mode:
            steps_this_stage += 1
            _budget_remaining = current_stage_budget - steps_this_stage

            if _budget_remaining < 0:
                # Budget exhausted — force-advance to next stage
                logger.warning(
                    "stage '%s' budget exhausted (%d/%d steps) — forcing advance",
                    current_stage["type"], steps_this_stage, current_stage_budget,
                )
                _complete_stage(
                    f"budget exhausted after {steps_this_stage} steps "
                    f"(budget={current_stage_budget}) — partial output"
                )
                current_stage = _pop_next_stage()
                continue

            _budget_warn = (_budget_remaining <= 2)

        # Auto-compact if history is growing too large
        _in_fmt = bool(current_stage and current_stage.get("type") in _FORMAT_STAGE_TYPES)
        if should_compact(messages, in_format_stage=_in_fmt):
            _next_goal = stage_queue[0]["goal"] if stage_queue else (current_stage or {}).get("goal", "")
            result_c = compact(messages, in_format_stage=_in_fmt, current_goal=_next_goal)
            if result_c.removed_message_count > 0:
                messages = result_c.compacted_messages
                logger.info("compacted %d messages", result_c.removed_message_count)
                if session_log:
                    session_log.compaction(result_c.removed_message_count, result_c.formatted_summary)
                # Post-compaction manifest re-injection (injection point C)
                if _in_fmt:
                    from birdclaw.tools.context_vars import get_task_id as _get_tid2
                    from birdclaw.memory.tasks import task_registry as _tr2
                    from birdclaw.agent import subtask_manifest as _sm_mod, subtask_verifier as _sv_mod
                    import dataclasses as _dc
                    _tid2 = _get_tid2()
                    _mdict = _tr2.get_manifest(_tid2) if _tid2 else None
                    if _mdict:
                        try:
                            _items = [_sm_mod.SubtaskItem(**i) for i in _mdict.get("items", [])]
                            _m = _sm_mod.SubtaskManifest(
                                stage_goal=_mdict["stage_goal"],
                                file_path=_mdict["file_path"],
                                file_type=_mdict["file_type"],
                                items=_items,
                                file_content_hash=_mdict.get("file_content_hash", ""),
                            )
                            _fc = _subtask_executor._read_file(_m.file_path)
                            _rdiff = _sv_mod.run(_m, _fc)
                            messages.append(Message(role="user", content=_rdiff.resume_context))
                            logger.info("post-compaction: re-injected manifest resume context")
                        except Exception as _ce:
                            logger.warning("post-compaction manifest re-injection failed: %s", _ce)

        # Queue exhausted — inject completion context and restrict to answer() only.
        if current_stage is None and outcome and not answering_mode:
            progress = "\n".join(f"  - {s['type']}: {s['summary']}" for s in completed_stages)
            file_hint = f" File written: {last_written_path}." if last_written_path else ""
            last = messages[-1]
            completion_msg = (
                f"[{step}/{max_steps}] All stages complete.{file_hint}\n"
                f"Outcome: {outcome}\n"
                f"Completed:\n{progress}\n"
                f"Call answer() to report what was accomplished."
            )
            if last.role == "user" and last.content.startswith("[") and "/" in last.content[:8]:
                messages[-1] = Message(role="user", content=completion_msg)
            else:
                messages.append(Message(role="user", content=completion_msg))
            answering_mode = True

        # Determine active schemas for this step
        if answering_mode:
            active_schemas = [ANSWER_SCHEMA]
        elif current_stage:
            stage_type = current_stage["type"]
            if stage_type in _FORMAT_STAGE_TYPES:
                active_schemas = []  # format mode — no tool schemas
            else:
                active_schemas = _tools_for_stage(stage_type)
                # Offer request_budget when approaching the budget limit
                if _budget_warn:
                    active_schemas = active_schemas + [_REQUEST_BUDGET_SCHEMA]
        else:
            active_schemas = free_tool_schemas

        # --- Per-step context message (A5) ---
        if current_stage:
            done_parts = " → ".join(s["goal"][:28] for s in completed_stages)
            done_str = done_parts if done_parts else "none yet"
            stage_type = current_stage["type"]

            if stage_type in _FORMAT_STAGE_TYPES:
                # Two explicit options — avoids model generating prose "I'm done" instead of JSON.
                # The "DONE" sentinel path is a unique token the model won't generate for real files.
                # Always include the workspace dir so the model writes to the right place.
                import re as _re_fmt
                if stage_type == "write_code":
                    # Extract target .py path from goal; fallback is cwd (not task subfolder —
                    # output artifacts belong in the project dir, not the agent memory dir).
                    _py_match = _re_fmt.search(r'(/[\w./\-]+\.py)', current_stage['goal'])
                    _example_path = _py_match.group(1) if _py_match else f"{_write_dir}/{_task_slug}.py"
                    fmt_hint = (
                        f'Write directory: {_write_dir}/\n'
                        'Two choices — pick one:\n'
                        f'  Write next function: {{"path": "{_example_path}", "content": "import requests\\n\\ndef fetch_url(url):\\n    pass"}}\n'
                        '  All functions written: {"path": "DONE", "content": ""}'
                    )
                elif stage_type == "write_doc":
                    # Extract target file path from goal if an absolute path is present.
                    # Fallback to cwd so multi-file projects land together in the project dir.
                    _doc_match = _re_fmt.search(r'(/[\w./\-]+\.\w+)', current_stage['goal'])
                    _example_path = _doc_match.group(1) if _doc_match else f"{_write_dir}/{_task_slug}.md"
                    fmt_hint = (
                        f'Write directory: {_write_dir}/\n'
                        'Two choices — pick one:\n'
                        f'  Write content: {{"path": "{_example_path}", "content": "Full text here..."}}\n'
                        '  All content written: {"path": "DONE", "content": ""}'
                    )
                else:  # edit_file
                    _et = edit_target_path or last_written_path or "unknown"
                    fmt_hint = (
                        f'File being edited: {_et}\n'
                        'Two choices — pick one:\n'
                        f'  Patch a section: {{"old": "exact text to replace", "new": "replacement text"}}\n'
                        '  All edits done: {"old": "DONE", "new": ""}'
                    )
                _bw = f" | BUDGET: {_budget_remaining} steps left — output DONE/done=true to wrap up" if _budget_warn else ""
                step_msg = f"[{step}/{max_steps}] Goal: {current_stage['goal']} | {fmt_hint}{_bw}"
            elif stage_type == "verify":
                # Tailor bash hint to the goal: py_compile only for Python file checks.
                _goal_lower = current_stage['goal'].lower()
                if any(w in _goal_lower for w in ('.py', 'python', 'compile', 'syntax', 'script')):
                    _bash_hint = "Use bash, e.g. bash('python -m py_compile <file>') to check Python syntax."
                else:
                    _bash_hint = "Use bash to execute and verify the result directly."
                step_msg = (
                    f"[{step}/{max_steps}] {current_stage['goal']} | "
                    f"{_bash_hint}"
                )
            elif outcome:
                _bw = f" | BUDGET WARNING: {_budget_remaining} steps left — call request_budget if more work remains" if _budget_warn else ""
                step_msg = (
                    f"[{step}/{max_steps}] Outcome: {outcome} | "
                    f"Done: {done_str} | "
                    f"Now ({stage_type}): {current_stage['goal']}{_bw}"
                )
                # Inject notes into research/reflect steps — read the full file up to 2000 chars
                if stage_type in _THINK_ADVANCES_TYPES and Path(_task_notes_path).is_file():
                    try:
                        _notes_raw = Path(_task_notes_path).read_text(encoding="utf-8", errors="replace")
                        _notes_ctx = _notes_raw[-2000:] if len(_notes_raw) > 2000 else _notes_raw
                        if _notes_ctx.strip():
                            logger.debug("[notes] injected into step_msg  stage=%s  chars=%d", stage_type, len(_notes_ctx))
                            step_msg = f"[Notes from earlier in this task]\n{_notes_ctx}\n\n{step_msg}"
                    except Exception:
                        pass
            else:
                _bw = f" | BUDGET WARNING: {_budget_remaining} steps left" if _budget_warn else ""
                step_msg = f"[{step}/{max_steps}] {stage_type}: {current_stage['goal']}{_bw}"
        else:
            tool_names = [t["function"]["name"] for t in active_schemas]
            step_msg = f"[{step}/{max_steps}] Call one: {', '.join(tool_names)}"

        last = messages[-1]
        if last.role == "user" and last.content.startswith("[") and "/" in last.content[:8]:
            messages[-1] = Message(role="user", content=step_msg)
        elif last.role != "user":
            messages.append(Message(role="user", content=step_msg))

        logger.debug("step %d/%d  stage=%s  tools=%s",
                     step, max_steps,
                     current_stage["type"] if current_stage else "free",
                     [s["function"]["name"] for s in active_schemas])

        # ── Format mode ──────────────────────────────────────────────────────
        if current_stage and current_stage["type"] in _FORMAT_STAGE_TYPES:
            stage_type = current_stage["type"]

            # ── edit_file: surgical patch loop ────────────────────────────────
            if stage_type == "edit_file":
                # Resolve target file on first iteration of this stage
                if not edit_target_path:
                    edit_target_path = _parse_edit_target(
                        current_stage["goal"], last_written_path
                    )
                    edit_not_found = 0

                if not edit_target_path or not Path(edit_target_path).exists():
                    logger.warning("edit_file: target not found (%r) — skipping stage", edit_target_path)
                    _complete_stage("edit_file: target file not found")
                    edit_target_path = ""
                    current_stage = _pop_next_stage()
                    continue

                # Always re-read so model sees latest content after each patch
                try:
                    current_content = Path(edit_target_path).read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning("edit_file: cannot read %s: %s", edit_target_path, exc)
                    _complete_stage("edit_file: read error")
                    edit_target_path = ""
                    current_stage = _pop_next_stage()
                    continue

                # Minimal context: system + current file + goal. No full conversation history.
                cap = 3000
                content_preview = current_content[:cap] + ("…" if len(current_content) > cap else "")

                # Step 1: 4B identifies exactly what to find and replace (thinking=True).
                edit_messages = [
                    Message(role="system", content=(
                        "You are editing a file. Think carefully, then output JSON only.\n"
                        "Two choices — pick one:\n"
                        '  Patch a section: {"old": "exact text to replace", "new": "replacement text"}\n'
                        '  All edits done: {"old": "DONE", "new": ""}\n'
                        "The old text must match character-for-character."
                    )),
                    Message(
                        role="user",
                        content=(
                            f"Goal: {current_stage['goal']}\n\n"
                            f"File: {edit_target_path}\n"
                            "```\n" + content_preview + "\n```"
                        ),
                    ),
                ]

                result = llm_client.generate(
                    edit_messages, format_schema=_EDIT_PATCH_SCHEMA, thinking=True,
                    profile=_main_profile(),
                )
                if result.thinking:
                    thinking_traces.append(result.thinking)

                parsed = _parse_format_response(result.content)
                if not parsed:
                    logger.warning("edit_file: bad parse (content=%r) — nudging", (result.content or "")[:80])
                    # Inject an explicit JSON example so the model corrects itself
                    edit_messages.append(Message(
                        role="assistant", content=result.content or ""
                    ))
                    edit_messages.append(Message(
                        role="user",
                        content=(
                            'Output ONLY valid JSON. Example:\n'
                            '{"old": "exact text to replace", "new": "replacement"}\n'
                            'or {"old": "DONE", "new": ""}'
                        ),
                    ))
                    result2 = llm_client.generate(
                        edit_messages, format_schema=_EDIT_PATCH_SCHEMA, thinking=True,
                        profile=_main_profile(),
                    )
                    parsed = _parse_format_response(result2.content)
                    if not parsed:
                        logger.warning("edit_file: bad parse after retry — skipping step")
                        continue

                old_text = parsed.get("old", "")
                new_text = parsed.get("new", "")

                if old_text == "DONE":
                    logger.info("edit_file: done — %s", edit_target_path)
                    _complete_stage(f"edited {edit_target_path}")
                    edit_target_path = ""
                    current_stage = _pop_next_stage()
                    continue

                if old_text not in current_content:
                    edit_not_found += 1
                    logger.warning("edit_file: old text not found (attempt %d)", edit_not_found)
                    if edit_not_found >= 3:
                        logger.warning("edit_file: too many not-found retries — advancing")
                        _complete_stage("edit_file: partial (text not found)")
                        edit_target_path = ""
                        current_stage = _pop_next_stage()
                        continue
                    # O2: fuzzy-match to give the model a corrective hint
                    fuzzy_hint = _fuzzy_match_hint(old_text, current_content)
                    edit_messages.append(Message(
                        role="user",
                        content=(
                            f"The exact text was not found in the file.\n{fuzzy_hint}"
                            "Output corrected JSON with the exact matching text."
                        ),
                    ))
                    result_retry = llm_client.generate(
                        edit_messages, format_schema=_EDIT_PATCH_SCHEMA,
                        thinking=True, profile=_main_profile(),
                    )
                    parsed = _parse_format_response(result_retry.content) or {}
                    old_text = parsed.get("old", "")
                    new_text = parsed.get("new", "")
                    if old_text == "DONE":
                        _complete_stage(f"edited {edit_target_path}")
                        edit_target_path = ""
                        current_stage = _pop_next_stage()
                    elif old_text and old_text in current_content:
                        # Retry produced a valid match — apply it now rather than
                        # letting the loop advance a step without writing anything.
                        edit_not_found = 0
                        patched = current_content.replace(old_text, new_text, 1)
                        try:
                            Path(edit_target_path).write_text(patched, encoding="utf-8")
                            last_written_path = edit_target_path
                            logger.info("edit_file: patched (retry) %s", edit_target_path)
                        except OSError as _exc:
                            logger.error("edit_file: retry write failed: %s", _exc)
                    continue

                # Apply the patch
                edit_not_found = 0
                patched = current_content.replace(old_text, new_text, 1)
                try:
                    Path(edit_target_path).write_text(patched, encoding="utf-8")
                    last_written_path = edit_target_path
                    logger.info("edit_file: patched %s (%d→%d chars)", edit_target_path, len(current_content), len(patched))
                    if session_log:
                        session_log.tool_result("edit_file", f"Patched {edit_target_path}: replaced {len(old_text)} chars")
                except OSError as exc:
                    logger.error("edit_file: write failed: %s", exc)
                continue

            # ── write_code / write_doc: subtask executor ─────────────────────
            completed_type = current_stage["type"]
            file_type = "doc" if completed_type == "write_doc" else "code"

            # O8: extract all plausible paths from goal; use the first absolute one,
            # fall back to relative paths, then default. Multiple candidates are
            # logged so the verifier can check the right file.
            import re as _re_fmt2
            _all_path_candidates = _re_fmt2.findall(
                r'(?:(?:to|into|at|file[: ]+|path[: ]+)\s+)?'
                r'([~/]?[\w./\-]+\.(?:py|md|txt|json|yaml|yml|toml|js|ts|sh|html|css))',
                current_stage["goal"],
            )
            _abs_paths = [p for p in _all_path_candidates if p.startswith(("/", "~"))]
            _rel_paths = [p for p in _all_path_candidates if not p.startswith(("/", "~"))]
            _ext = 'md' if file_type == 'doc' else 'py'
            # Fallback filename: task_id + stage index, guaranteed unique across all
            # write stages in a task regardless of how many files the task produces.
            _write_stage_n = sum(1 for s in completed_stages if s["type"] in ("write_code", "write_doc"))
            _fallback_name = f"{_task_slug}_s{_write_stage_n + 1}" if _write_stage_n > 0 else _task_slug
            _fallback = f"{_write_dir}/{_fallback_name}.{_ext}"
            _inferred_path = (_abs_paths or _rel_paths or [_fallback])[0]
            if len(_all_path_candidates) > 1:
                logger.debug("write stage: multiple path candidates %s — using %s", _all_path_candidates, _inferred_path)

            from birdclaw.tools.context_vars import get_task_id as _get_task_id
            from birdclaw.memory.tasks import task_registry as _task_registry

            _task_id = _get_task_id()

            def _store_manifest(m: Any) -> None:
                import dataclasses
                m_dict = dataclasses.asdict(m)
                if _task_id:
                    _task_registry.set_manifest(_task_id, m_dict)
                if session_log:
                    session_log.subtask_manifest(
                        stage_index=len(completed_stages),
                        manifest_dict=m_dict,
                    )

            def _on_item_start(title: str, path: str) -> None:
                if session_log:
                    session_log.tool_call("write_item", {"title": title, "path": path})

            def _on_item_done(title: str, success: bool, duration_ms: int) -> None:
                if session_log:
                    session_log.tool_result(
                        "write_item",
                        f"{title}: {'complete' if success else 'partial'}",
                        duration_ms=duration_ms,
                    )

            stage_result = _subtask_executor.run_stage(
                llm_client=llm_client,
                stage=current_stage,
                file_path=_inferred_path,
                file_type=file_type,
                step=step,
                max_steps=max_steps,
                store_manifest=_store_manifest,
                on_item_start=_on_item_start,
                on_item_done=_on_item_done,
                interrupt_event=interrupt_event,
            )

            last_written_path = stage_result.written_path
            _was_dynamic = bool((current_stage or {}).get("_dynamic"))
            _complete_stage(stage_result.summary)

            if outcome and last_written_path and not _was_dynamic:
                gate = _reflect_on_stage(outcome, completed_type, stage_result.summary, last_written_path, steps_remaining=max_steps - step, notes_path=_task_notes_path)
                decision = gate.get("decision", "continue")
                if decision == "done":
                    stage_queue.clear()
                    logger.info("reflect gate: outcome met — clearing queue")
                elif decision in ("deepen", "insert"):
                    if decision == "deepen" and completed_type in ("write_doc", "write_code"):
                        deepen_goal = f"{last_written_path}: {gate['goal']}"
                        stage_queue.insert(0, {"type": "edit_file", "goal": deepen_goal, "_dynamic": True})
                        _tui_add_phase(_plan_task_id, deepen_goal)
                        logger.info("reflect gate: deepen %s → edit_file: %s", completed_type, gate["goal"][:60])
                    else:
                        insert_type = gate.get("type", completed_type) if decision == "insert" else completed_type
                        stage_queue.insert(0, {"type": insert_type, "goal": gate["goal"], "_dynamic": True})
                        _tui_add_phase(_plan_task_id, gate["goal"])
                        logger.info("reflect gate: %s %s → %s", decision, insert_type, gate["goal"][:60])

            edit_target_path = ""
            current_stage = _pop_next_stage()
            continue

        # ── Tool call mode ────────────────────────────────────────────────────
        # Enable thinking only for stages that offer the think tool.
        # Gemma4+Ollama: thinking + tool_choice causes reasoning-only output (no tool call JSON).
        enable_thinking = "think" in {s["function"]["name"] for s in active_schemas}
        # Use "required" only when answer is the sole tool (completion step).
        # Gemma4+Ollama bug: required + thinking = reasoning-only output; but here
        # thinking is False (no think tool), so required works correctly.
        schema_names = {s["function"]["name"] for s in active_schemas}
        result = llm_client.generate(
            messages,
            tools=active_schemas,
            tool_choice="auto",
            thinking=enable_thinking,
            profile=_main_profile(),
        )

        if result.thinking:
            thinking_traces.append(result.thinking)

        # No tool call — handle depending on context
        if not result.tool_calls:
            logger.debug("no tool call at step %d (content_len=%d)", step, len(result.content or ""))
            if current_stage:
                consecutive_no_tool += 1
                stage_type = current_stage["type"]

                # Force-advance research/reflect after 2 consecutive non-tool responses.
                # These stages end when think() is called — if the model keeps writing text,
                # it has gathered what it needs but hasn't signalled completion.
                if consecutive_no_tool >= 2 and stage_type in _THINK_ADVANCES_TYPES:
                    logger.warning(
                        "force-advancing %s stage after %d consecutive non-tool responses",
                        stage_type, consecutive_no_tool,
                    )
                    consecutive_no_tool = 0
                    _complete_stage(f"(auto-advanced after stall) {(result.content or '')[:120]}")
                    current_stage = _pop_next_stage()
                    next_info = (
                        f"{current_stage['type']}: {current_stage['goal']}"
                        if current_stage else "All stages complete."
                    )
                    messages.append(Message(
                        role="user",
                        content=f"[{step}/{max_steps}] Stage auto-advanced. Next: {next_info}",
                    ))
                    continue

                # Force-advance other stages (verify, free mode) after 4 non-tool responses.
                if consecutive_no_tool >= 4:
                    logger.warning(
                        "force-advancing %s stage after %d non-tool responses",
                        stage_type, consecutive_no_tool,
                    )
                    consecutive_no_tool = 0
                    _complete_stage(f"(stalled — {(result.content or '')[:120]})")
                    current_stage = _pop_next_stage()
                    continue

                # Stronger nudge — include a concrete example call
                tool_names = _STAGE_TYPE_TOOLS.get(stage_type, ["answer"])
                example_tool = tool_names[0] if tool_names else "answer"
                nudge = (
                    f"[{step}/{max_steps}] You must call a tool now — do NOT write prose. "
                    f"Available: {', '.join(tool_names + ['answer'])}. "
                    f"Example: call {example_tool}() with the required arguments."
                )
                messages.append(Message(role="user", content=nudge))
                continue
            if schema_names == {"answer"}:
                # Answer-only step: accept text content as answer (model may not call tool
                # due to Gemma4+Ollama thinking+required bug). If content is empty,
                # nudge once with the completion context still in history.
                if result.content.strip():
                    return result.content.strip(), sources, step, thinking_traces, completed_stages
                messages.append(Message(
                    role="user",
                    content=f"[{step}/{max_steps}] Call answer() to summarize what was done.",
                ))
                continue
            return result.content.strip(), sources, step, thinking_traces, completed_stages

        # A1: process only the first tool call per step
        tc = result.tool_calls[0]
        consecutive_no_tool = 0
        logger.debug("tool_call: %s  args=%s", tc.name, str(tc.arguments)[:120])

        # Reject out-of-schema calls — must still include tool_calls in assistant
        # message so llama.cpp can match it to the following role=tool message.
        offered = {s["function"]["name"] for s in active_schemas}
        call_id = tc.id or f"call_{step}"
        if tc.name not in offered:
            logger.warning("out-of-schema %r (offered: %s)", tc.name, sorted(offered))
            messages.append(Message(
                role="assistant",
                content=result.content or "",
                tool_calls=[{
                    "id": call_id, "type": "function",
                    "function": {"name": tc.name, "arguments": _json.dumps(tc.arguments)},
                }],
            ))
            messages.append(Message(
                role="tool",
                content=f"'{tc.name}' not available here. Call one of: {', '.join(sorted(offered))}.",
                tool_call_id=call_id,
                name=tc.name,
            ))
            continue

        if session_log:
            session_log.tool_call(tc.name, tc.arguments)

        # Append assistant turn with tool_calls so llama.cpp can match the tool response.
        messages.append(Message(
            role="assistant",
            content=result.content or "",
            tool_calls=[{
                "id": call_id, "type": "function",
                "function": {"name": tc.name, "arguments": _json.dumps(tc.arguments)},
            }],
        ))

        # ── answer ────────────────────────────────────────────────────────────
        if tc.name == "answer":
            content = tc.arguments.get("content", "").strip()
            # JSON may be truncated (long answers hit token limit mid-string).
            # Fall back to the raw text content the model emitted alongside the call.
            if not content:
                content = (result.content or "").strip()
            answer_sources = tc.arguments.get("sources", [])
            return content, list(dict.fromkeys(sources + answer_sources)), step, thinking_traces, completed_stages

        # ── think ─────────────────────────────────────────────────────────────
        if tc.name == "think":
            reasoning = tc.arguments.get("reasoning", "")
            logger.debug("think: %s", reasoning[:200])
            thinking_traces.append(reasoning)
            think_count += 1
            # Save reasoning to notes so future stages can recall what was concluded
            if current_stage and current_stage["type"] in _THINK_ADVANCES_TYPES:
                _task_dir.mkdir(parents=True, exist_ok=True)
                _append_to_notes(_task_notes_path, "think", tc.arguments, reasoning)

            # Research stages must use at least one real tool before think() can advance.
            # Research stages must fetch real data before think() can advance.
            # If the model calls think() with no real tool calls, it is hallucinating
            # prior searches from compacted context. On the first occurrence, nudge it.
            # On the second occurrence: if the goal is a synthesis task (summarise/
            # synthesize/etc.) just advance — no new data needed. Otherwise force a
            # web_search so the model has real data to reason from.
            if (
                current_stage
                and current_stage["type"] == "research"
                and stage_real_calls == 0
            ):
                _is_synthesis = any(kw in current_stage["goal"].lower() for kw in _SYNTHESIS_KW)
                if think_count == 1:
                    if _is_synthesis:
                        # Synthesis goals have no data to fetch — let the model advance
                        # immediately using the notes already in context.
                        messages.append(Message(
                            role="tool",
                            content="Synthesise the findings already in your notes and call think() to complete this stage.",
                            tool_call_id=call_id,
                            name="think",
                        ))
                    else:
                        # First offence on a fetch goal: nudge clearly
                        messages.append(Message(
                            role="tool",
                            content=(
                                f"No web search has been done yet in this stage. "
                                f"Call web_search now with a query for: {current_stage['goal'][:120]}"
                            ),
                            tool_call_id=call_id,
                            name="think",
                        ))
                    continue
                else:
                    if _is_synthesis:
                        # Synthesis still stalling — force-advance with what we have.
                        logger.warning(
                            "research stage stall: synthesis goal, no real tool needed — "
                            "force-advancing: %s", current_stage["goal"][:80],
                        )
                        messages.append(Message(
                            role="tool",
                            content="Orchestrator: advancing synthesis stage.",
                            tool_call_id=call_id,
                            name="think",
                        ))
                        _complete_stage(reasoning[:120])
                        current_stage = _pop_next_stage()
                        continue
                    else:
                        # Second+ offence on a fetch goal: orchestrator force-invokes
                        # the right tool — web_fetch for URL goals, web_search otherwise.
                        import re as _re_url
                        _url_match = _re_url.search(r'https?://\S+', current_stage["goal"])
                        if _url_match:
                            _forced_url = _url_match.group(0).rstrip(".,)")
                            logger.warning(
                                "research stage stall: model called think() %d times with no real "
                                "tool — force-invoking web_fetch for: %s",
                                think_count, _forced_url[:80],
                            )
                            from birdclaw.tools.web import web_fetch as _web_fetch_fn
                            try:
                                forced_result = _web_fetch_fn(_forced_url)
                            except Exception as _fe:
                                forced_result = f'{{"error": "{_fe}"}}'
                            _force_label = f"web_fetch({_forced_url[:60]})"
                        else:
                            logger.warning(
                                "research stage stall: model called think() %d times with no real "
                                "tool — force-invoking web_search for: %s",
                                think_count, current_stage["goal"][:80],
                            )
                            forced_query = current_stage["goal"][:120]
                            from birdclaw.tools.web import web_search as _web_search_fn
                            try:
                                forced_result = _web_search_fn(forced_query)
                            except Exception as _fe:
                                forced_result = f'{{"error": "{_fe}"}}'
                            _force_label = f"web_search({forced_query[:60]})"
                        stage_real_calls += 1
                        messages.append(Message(
                            role="tool",
                            content="Noted. Orchestrator is fetching data for you.",
                            tool_call_id=call_id,
                            name="think",
                        ))
                        messages.append(Message(
                            role="user",
                            content=(
                                f"[orchestrator {_force_label}]\n"
                                f"{forced_result[:1200]}\n\n"
                                f"Real data injected above. "
                                f"Now call think() to summarise what you found."
                            ),
                        ))
                        continue

            # O6: detect research loop — model repeats same think() reasoning after forced search
            if (
                current_stage
                and current_stage["type"] == "research"
                and stage_real_calls > 0
                and _last_think_reasoning
            ):
                import difflib as _dl
                _loop_ratio = _dl.SequenceMatcher(
                    None, _last_think_reasoning[:300], reasoning[:300], autojunk=False
                ).ratio()
                if _loop_ratio > 0.75:
                    logger.warning(
                        "research stage loop detected (similarity=%.2f) — force-advancing", _loop_ratio
                    )
                    messages.append(Message(
                        role="tool",
                        content="Orchestrator: loop detected — advancing to next stage.",
                        tool_call_id=call_id,
                        name="think",
                    ))
                    _complete_stage(reasoning[:120])
                    current_stage = _pop_next_stage()
                    continue
            _last_think_reasoning = reasoning

            feedback = (
                "Plan noted. Act now."
                if think_count < _MAX_THINK_STEPS
                else "Act now — no more thinking."
            )
            messages.append(Message(role="tool", content=feedback, tool_call_id=call_id, name="think"))
            # think() advances stages where it signals "done" (research, reflect)
            if current_stage and current_stage["type"] in _THINK_ADVANCES_TYPES:
                completed_type = current_stage["type"]
                completed_goal = current_stage["goal"]
                _complete_stage(reasoning[:120])

                # Reflection gate — evaluate quality, possibly insert follow-up stage.
                # No insertion cap: the model decides when enough is enough.
                # max_agent_steps is the only backstop.
                if outcome:
                    gate = _reflect_on_stage(outcome, completed_type, reasoning[:200], last_written_path, steps_remaining=max_steps - step, notes_path=_task_notes_path)
                    logger.debug("reflect gate: %s", gate)
                    decision = gate.get("decision", "continue")
                    if decision == "done":
                        stage_queue.clear()
                        logger.info("reflect gate: outcome met — clearing queue")
                    elif decision == "deepen":
                        deepen_counts[completed_type] = deepen_counts.get(completed_type, 0) + 1
                        if deepen_counts[completed_type] > 2:
                            logger.warning(
                                "reflect gate: deepen cap hit for %s (%d times) — forcing continue",
                                completed_type, deepen_counts[completed_type],
                            )
                        else:
                            deepen_stage = {"type": completed_type, "goal": gate["goal"]}
                            stage_queue.insert(0, deepen_stage)
                            logger.info("reflect gate: deepen → %s", gate["goal"][:60])
                    elif decision == "insert":
                        insert_stage = {"type": gate.get("type", "research"), "goal": gate["goal"]}
                        stage_queue.insert(0, insert_stage)
                        logger.info("reflect gate: insert %s → %s", insert_stage["type"], gate["goal"][:60])

                edit_target_path = ""  # reset between stages
                current_stage = _pop_next_stage()
                logger.debug("think → next stage: %s", current_stage)
            continue

        # ── request_budget ────────────────────────────────────────────────────
        if tc.name == "request_budget":
            additional = tc.arguments.get("additional_steps", 10)
            reason = tc.arguments.get("reason", "")
            try:
                additional = int(additional)
            except (TypeError, ValueError):
                additional = 10
            # Cap a single grant at the configured maximum
            additional = max(1, min(additional, settings.stage_budget_max_grant))
            current_stage_budget += additional
            logger.info(
                "budget extended: +%d → %d for stage '%s' (%s)",
                additional, current_stage_budget,
                current_stage["type"] if current_stage else "?",
                reason[:80],
            )
            messages.append(Message(
                role="tool",
                content=(
                    f"Budget extended by {additional} steps. "
                    f"New budget: {current_stage_budget}. Continue."
                ),
                tool_call_id=call_id,
                name="request_budget",
            ))
            continue

        # ── use_skill (legacy compat — planner has replaced skill selection) ──
        if tc.name == "use_skill":
            messages.append(Message(
                role="tool",
                content="Skill selection is handled by the planner. Continue with your current stage.",
                tool_call_id=call_id,
                name="use_skill",
            ))
            continue

        # ── all other tools ───────────────────────────────────────────────────
        from birdclaw.tools.context_vars import set_stage_goal
        if current_stage:
            set_stage_goal(current_stage["goal"])

        # Count real tool calls per stage (used to enforce think() gating)
        stage_real_calls += 1

        try:
            obs = execute(tc)
        except Exception as exc:
            obs = f"[tool error: {exc}]"
            logger.error("[tool] execute failed  tool=%s  error=%s", tc.name, exc)
        logger.info("[tool] result  tool=%s  len=%d  preview=%r", tc.name, len(obs), obs[:120])

        # Append to task notes file for future search_relevant lookups
        if current_stage and current_stage["type"] in _THINK_ADVANCES_TYPES:
            _task_dir.mkdir(parents=True, exist_ok=True)
            _append_to_notes(_task_notes_path, tc.name, tc.arguments, obs)

        # Inject any pending condenser notes from previous web fetches
        from birdclaw.tools.condenser import drain_pending_notes
        pending = drain_pending_notes()
        if pending:
            notes_text = "\n\n".join(
                f"[Notes from {n.url[:60]}]\n{n.notes}" for n in pending
            )
            obs = obs + f"\n\n[Condenser notes ready]\n{notes_text}"

        if session_log:
            _dur = 0
            if tc.name == "bash":
                try:
                    _dur = _json.loads(obs).get("duration_ms", 0)
                except Exception:
                    pass
            session_log.tool_result(tc.name, obs, duration_ms=_dur)

        # Track last written path for completion message hints (v2 learning)
        if tc.name == "write_file":
            try:
                _wr = _json.loads(obs)
                if "path" in _wr:
                    last_written_path = _wr["path"]
            except Exception:
                pass

        # Collect web sources + extract entities into knowledge graph
        if tc.name in ("web_search", "web_fetch"):
            try:
                _wr = _json.loads(obs)
                for _r in _wr.get("results", []):
                    if _url := _r.get("url"):
                        sources.append(_url)
                # NER: mine entities from the raw result text
                try:
                    from birdclaw.memory.retrieval import extract_and_index, retrieve
                    _context_hint = f"{tc.name}: {str(tc.arguments.get('query', tc.arguments.get('url', '')))[:60]}"
                    logger.debug("[graphrag] extract_and_index  source=%s", _context_hint[:60])
                    extract_and_index(obs, context=_context_hint)
                    # Re-inject fresh context based on what we just learned
                    if current_stage:
                        _fresh_ctx = retrieve(current_stage.get("goal", ""))
                        if _fresh_ctx:
                            logger.debug("[graphrag] re-inject after web tool  chars=%d", len(_fresh_ctx))
                            messages.append(Message(
                                role="user",
                                content=f"[updated knowledge context] {_fresh_ctx}",
                            ))
                except Exception:
                    pass
            except Exception:
                pass

        messages.append(Message(role="tool", content=obs, tool_call_id=call_id, name=tc.name))

        # Research stage: after enough real searches, push the model toward think().
        # Without this, the model loops on web_search until budget exhaustion.
        if current_stage and current_stage["type"] == "research" and stage_real_calls >= 3:
            if stage_real_calls >= 5:
                # Hard stop: force-complete the stage with what we have rather than
                # letting it burn all remaining budget with more redundant searches.
                logger.warning(
                    "research stage: %d real calls without think() — force-completing",
                    stage_real_calls,
                )
                _complete_stage(f"force-completed after {stage_real_calls} searches (no think called)")
                current_stage = _pop_next_stage()
                messages.append(Message(
                    role="user",
                    content=(
                        f"[orchestrator] Research stage ended after {stage_real_calls} searches. "
                        f"Next: {current_stage['type'] + ': ' + current_stage['goal'][:60] if current_stage else 'All stages done.'}"
                    ),
                ))
                continue
            else:
                # Soft nudge: append a user-turn after the tool result
                messages.append(Message(
                    role="user",
                    content=(
                        f"[{stage_real_calls} searches done] You have enough data. "
                        f"Call think() now to synthesize what you found and complete this research stage."
                    ),
                ))

        # verify stage: bash result drives next step
        # Success → advance. Failure → push a repair write_code stage to fix errors.
        # This implements the micro-generation + verify loop:
        #   write function → verify → if error: fix → verify → ... → next function
        if current_stage and current_stage["type"] == "verify" and tc.name == "bash":
            # Determine success purely from exit code — never scan stdout content,
            # which legitimately contains words like "error", "failed", "exception".
            import re as _re
            _exit_match = _re.search(r'"exit_code[_\s:]*(\d+)"', obs)
            _exit_code = int(_exit_match.group(1)) if _exit_match else (
                0 if '"exit_code:0"' in obs or "exit_code: 0" in obs else 1
            )
            if _exit_code == 0:
                _complete_stage(f"bash ok: {obs[:60]}")
                current_stage = _pop_next_stage()
                logger.debug("verify ok → next stage: %s", current_stage)
            else:
                # Push a fix stage to the front — model gets write_code tools to repair
                # Build a human-readable goal instead of dumping raw JSON
                try:
                    import json as _jfix
                    _d = _jfix.loads(obs)
                    _cmd = (current_stage or {}).get("goal", "command")[:60]
                    _rc_str = _d.get("return_code_interpretation", f"exit {_exit_code}")
                    _stderr = (_d.get("stderr") or "")[:80].strip()
                    _fix_goal = f"Fix: {_rc_str} running {_cmd}"
                    if _stderr:
                        _fix_goal += f" — {_stderr}"
                except Exception:
                    _fix_goal = f"Fix: exit {_exit_code} in previous command"
                fix_stage = {
                    "type": "write_code",
                    "goal": _fix_goal,
                    "_dynamic": True,
                }
                stage_queue.insert(0, fix_stage)
                _tui_add_phase(_plan_task_id, _fix_goal)
                _complete_stage(f"verify failed (exit {_exit_code}) — queued fix: {obs[:60]}")
                current_stage = _pop_next_stage()  # pops the fix_stage
                logger.info("verify failed → queued repair stage")

    # --- Queue exhausted or max steps reached: forced answer ---
    logger.info(
        "loop end: forcing answer (outcome=%r, %d stages completed)",
        (outcome or "")[:40],
        len(completed_stages),
    )

    if session_log:
        ans_ctx = _answering_context(session_log)
        if ans_ctx and messages[0].role == "system":
            messages[0] = Message(
                role="system",
                content=ans_ctx + "\n\n" + messages[0].content,
            )

    if outcome or completed_stages:
        progress = "\n".join(
            f"  - {s['type']}: {s['summary']}" for s in completed_stages
        )
        messages.append(Message(
            role="user",
            content=(
                f"All stages complete.\n"
                f"Outcome target: {outcome}\n"
                f"Completed:\n{progress}\n"
                f"Call answer() to report what was accomplished."
            ),
        ))

    forced = llm_client.generate(
        messages,
        tools=[ANSWER_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "answer"}},
        thinking=True,
        profile=_main_profile(),
    )
    if forced.tool_calls:
        args = forced.tool_calls[0].arguments
        return args.get("content", "").strip(), args.get("sources", []), max_steps, thinking_traces, completed_stages

    answer = (forced.content or "").strip()
    if not answer:
        # Model returned blank (truncated tool call or empty response).
        # Build a deterministic summary from what was actually completed.
        parts = [f"- {s['type']}: {s['summary']}" for s in completed_stages if s.get("summary")]
        answer = "Task completed.\n" + "\n".join(parts) if parts else "Task completed."
        if last_written_path:
            answer += f"\n\nFile written: {last_written_path}"
        logger.warning("forced-answer was blank — using stage summary fallback")

    return answer, [], max_steps, thinking_traces, completed_stages


# ---------------------------------------------------------------------------
# [v2 removed — _run_loop is the single executor; see git history for v2]
# ---------------------------------------------------------------------------

# Sentinel so old call sites importing _run_loop_v2 raise a clear error.
def _run_loop_v2(*_a, **_kw):  # type: ignore[override]
    raise RuntimeError("_run_loop_v2 removed — use _run_loop")





# ---------------------------------------------------------------------------
# Document routing — save long write_doc results to ~/outputs/ instead of chat
# ---------------------------------------------------------------------------

def _maybe_save_doc(answer: str, completed_stages: list[dict], question: str) -> str:
    """If any completed stage was write_doc and the answer is long, save to file.

    Returns a short announcement pointing to the file path instead of the
    full content, so the conversation pane stays readable.  If the answer is
    short or no write_doc stage ran, returns the original answer unchanged.
    """
    _DOC_THRESHOLD = 500  # chars — below this just show inline

    has_doc_stage = any(s.get("type") == "write_doc" for s in completed_stages)
    if not has_doc_stage or len(answer) <= _DOC_THRESHOLD:
        return answer

    from birdclaw.tools.context_vars import get_task_id
    import re as _re2
    import time as _time2

    task_id = get_task_id() or f"task_{int(_time2.time())}"
    # Build a safe slug from the first few words of the question
    slug = _re2.sub(r"[^a-z0-9]+", "_", question[:40].lower()).strip("_") or "doc"

    outputs_dir = settings.data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    doc_path = outputs_dir / f"{task_id}_{slug}.md"

    try:
        doc_path.write_text(answer, encoding="utf-8")
        logger.info("write_doc output saved to %s (%d chars)", doc_path, len(answer))
        return (
            f"Document saved to `{doc_path}`\n\n"
            f"({len(answer):,} characters written)\n\n"
            f"Preview:\n{answer[:300]}{'…' if len(answer) > 300 else ''}"
        )
    except OSError as exc:
        logger.warning("could not save write_doc output: %s", exc)
        return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent_loop(
    question: str,
    history: list[Message] | None = None,
    extra_system: str | None = None,
    session_log: "SessionLog | None" = None,
    interrupt_event: "threading.Event | None" = None,
    write_dir: str = "",
) -> AgentResult:
    """Run the agent loop and return a complete AgentResult (blocking).

    Args:
        question:        The user's question or task.
        history:         Prior conversation turns for multi-turn context.
        extra_system:    Additional system prompt text prepended before SYSTEM.
        session_log:     Optional SessionLog — enables memory injection and event
                         recording. Pass one in to activate the full memory layer.
        interrupt_event: Optional threading.Event; if set mid-run the loop stops
                         at the next step boundary and returns what it has so far.
    """
    from birdclaw.memory.memorise import memorise_pause, memorise_resume, start_worker
    from birdclaw.memory.session_log import SessionLog
    from birdclaw.tools.context_vars import clear_stage_goal

    # Always have a session log — create one if the caller didn't supply one.
    if session_log is None:
        session_log = SessionLog.new()

    # Ensure memorise background worker is running
    start_worker()

    # Pause memorise while this task runs
    memorise_pause()
    try:
        session_log.user_message(question)

        messages = _build_messages(question, history, extra_system)

        # ── Inject workspace context BEFORE loop starts ──
        try:
            from birdclaw.agent.project_notes import workspace_context_for_task

            _ws_ctx = workspace_context_for_task(question)

            if _ws_ctx:
                logger.info("[ws-ctx] injected workspace context  chars=%d", len(_ws_ctx))
                messages.insert(1, Message(
                    role="user",
                    content=f"[Workspace context — read before beginning]\n{_ws_ctx}",
                ))
                messages.insert(2, Message(
                    role="assistant",
                    content="Understood. I have read the workspace files and project history.",
                ))
            else:
                logger.debug("[ws-ctx] no workspace context found")
        except Exception as _ws_err:
            logger.debug("workspace context injection failed: %s", _ws_err)

        # Extract skill hint from extra_system so the planner can follow the runbook.
        _skill_hint = ""
        if extra_system:
            for _line in extra_system.splitlines():
                if _line.startswith("## Active Skill:"):
                    _skill_hint = extra_system[extra_system.index(_line):]
                    break

        answer, sources, steps, thinking, completed_stages = _run_loop(
            messages, question, session_log,
            interrupt_event=interrupt_event,
            write_dir=write_dir,
            skill_hint=_skill_hint,
        )

        # Route long write_doc outputs to a file instead of the conversation.
        answer = _maybe_save_doc(answer, completed_stages, question)

        session_log.assistant_message(answer)

        # Append work-log entry to project BIRDCLAW.md
        try:
            from birdclaw.agent.project_notes import update_project_notes
            update_project_notes(
                cwd=_task_dir,   # per-task subfolder — no cwd-root bleed between tasks
                question=question,
                completed_stages=completed_stages,
            )
        except Exception:
            pass  # never block task completion on this

    finally:
        clear_stage_goal()
        memorise_resume()

    return AgentResult(answer=answer, sources=sources, steps=steps, thinking=thinking)


def run_agent_loop_stream(
    question: str,
    history: list[Message] | None = None,
    extra_system: str | None = None,
    session_log: "SessionLog | None" = None,
) -> "Iterator[str]":
    """Run the loop (blocking), then stream the answer in small chunks."""
    from typing import Iterator
    result = run_agent_loop(question, history, extra_system, session_log)
    chunk_size = 4
    for i in range(0, len(result.answer), chunk_size):
        yield result.answer[i: i + chunk_size]
