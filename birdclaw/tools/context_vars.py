"""Thread-local context variables shared across the tool call chain.

The agent loop sets these before each tool execution so tools (especially
the condenser) can access the current task context without needing it passed
as an explicit parameter.

Usage:
    # In the agent loop, before execute(tc):
    from birdclaw.tools.context_vars import set_stage_goal
    set_stage_goal(current_stage["goal"])

    # In any tool or helper:
    from birdclaw.tools.context_vars import get_stage_goal
    goal = get_stage_goal()   # "" if not set
"""

from __future__ import annotations

import threading

_local = threading.local()


def set_stage_goal(goal: str) -> None:
    """Set the current stage goal for this thread."""
    _local.stage_goal = goal


def get_stage_goal() -> str:
    """Return the current stage goal, or empty string if not set."""
    return getattr(_local, "stage_goal", "")


def clear_stage_goal() -> None:
    """Clear the stage goal (call when a task ends)."""
    _local.stage_goal = ""


# ---------------------------------------------------------------------------
# Task / agent identity (set by orchestrator per agent thread)
# ---------------------------------------------------------------------------

def set_task_context(task_id: str, agent_id: str) -> None:
    """Bind the running task and agent IDs to this thread."""
    _local.task_id  = task_id
    _local.agent_id = agent_id


def get_task_id() -> str:
    return getattr(_local, "task_id", "")


def get_agent_id() -> str:
    return getattr(_local, "agent_id", "")


def clear_task_context() -> None:
    _local.task_id  = ""
    _local.agent_id = ""


# ---------------------------------------------------------------------------
# LLM priority (set per agent thread for scheduler)
# ---------------------------------------------------------------------------

def set_llm_priority(priority: int) -> None:
    """Set the LLM scheduling priority for this thread (use LLMPriority constants)."""
    _local.llm_priority = priority


def get_llm_priority() -> int:
    """Return the LLM scheduling priority for this thread (default: AGENT = 1)."""
    return getattr(_local, "llm_priority", 1)  # 1 = AGENT
