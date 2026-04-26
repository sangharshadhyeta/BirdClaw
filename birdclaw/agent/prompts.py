"""System prompts and control tool schemas.

Prompts are kept compact — every token matters with a 4B model.
The think/answer control tools are defined here (not in the registry)
because they are always present and handled specially by the loop.

7b (parity port — SYSTEM_PROMPT_DYNAMIC_BOUNDARY):
  SYSTEM is the static portion — never modified, always the same token sequence.
  This lets llama.cpp reuse its KV cache for the system prompt on every call.
  Dynamic context (date, working dir, memory) is injected as a user message,
  NOT by prepending to the system message (which would bust the cache).
"""

from __future__ import annotations

import time as _time

# ---------------------------------------------------------------------------
# System prompt — STATIC (never modified — cache-stable across all calls)
# ---------------------------------------------------------------------------

# Task-execution system prompt — used by agent/loop.py (not the soul layer).
# For conversational system prompts use birdclaw.agent.soul.build_system_prompt().
SYSTEM = """\
You are BirdClaw, an AI agent. Always call a tool — never plain text.
Call answer() when done.
Tip: to run commands in a conda env use `conda run -n <env> <cmd>` (not `conda activate`).
"""


# ---------------------------------------------------------------------------
# Dynamic context — injected as a user message, NOT appended to SYSTEM
# ---------------------------------------------------------------------------

def dynamic_context(write_dir: str = "", task_dir: str = "") -> str:
    """One-line context injected as a user message before the main conversation.

    Kept separate from SYSTEM so the static prompt's KV cache is never
    invalidated. Called once per task at loop start.
    """
    from birdclaw.config import settings as _s
    parts = [f"Date: {_time.strftime('%Y-%m-%d %H:%M')}"]
    if write_dir:
        parts.append(f"Working dir: {write_dir}")
    if task_dir:
        parts.append(f"Task folder: {task_dir}")
    parts.append(f"Skills dir: {_s.skills_dir} (write skill files here)")
    parts.append(f"Source dir: {_s.src_dir} (read-only; use note_improvement() to log needed changes)")
    return " | ".join(parts)

# ---------------------------------------------------------------------------
# Control tool schemas (always injected, not in registry)
# ---------------------------------------------------------------------------

THINK_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Record your reasoning before acting. "
            "Use this to plan which tool to call next or to evaluate results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your current reasoning and next action plan.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

ANSWER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Deliver the final response. Call when the task is done.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
        },
    },
}

# Both think and answer are always present.
# think() gives the model a structured reasoning step before acting — critical
# when tool_choice="required" forces a call every turn.
CONTROL_TOOLS = [THINK_SCHEMA, ANSWER_SCHEMA]
