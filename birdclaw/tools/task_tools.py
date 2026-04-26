"""Task query tools — search and inspect the task registry.

Used by:
  - The soul loop (internal tool calls to find related work before answering)
  - The MCP server (external clients querying task state)

Tools registered here:
  search_tasks(query, status="any") → matching tasks with titles, status, output excerpts
  get_task_output(task_id)          → full details + output + saved document path
  note_improvement(description, priority) → append an item to the self-update backlog
"""

from __future__ import annotations

import json
import logging
import time

from birdclaw.tools.registry import registry, Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def search_tasks(query: str, status: str = "any") -> str:
    """Search the task registry by keyword across prompt, title, and output."""
    from birdclaw.memory.tasks import task_registry

    all_tasks = task_registry.list()
    q = query.lower()

    matches = [
        t for t in all_tasks
        if (status == "any" or t.status == status)
        and q in f"{t.prompt} {t.title} {t.output[:300]}".lower()
    ]

    if not matches:
        logger.debug("[tasks] search miss  query=%r  status=%s  total=%d", query[:40], status, len(all_tasks))
        return f"No tasks found matching {query!r} (status={status})."

    logger.info("[tasks] search hit  query=%r  status=%s  matches=%d", query[:40], status, len(matches))
    lines = [f"Found {len(matches)} task(s) matching {query!r}:"]
    for t in matches[:12]:
        age = ""
        if getattr(t, "created_at", None):
            secs = int(time.time() - t.created_at)
            h, m = divmod(secs // 60, 60)
            age = f" ({h}h{m:02d}m ago)" if h else f" ({m}m ago)"
        label = t.title if getattr(t, "title", "") else t.prompt[:55]
        lines.append(f"  [{t.task_id[:8]}] [{t.status}]{age}  {label}")
        if t.output:
            preview = t.output[:100].replace("\n", " ")
            lines.append(f"    Output: {preview}{'…' if len(t.output) > 100 else ''}")
    return "\n".join(lines)


def get_task_output(task_id: str) -> str:
    """Get full details, output, and any saved document path for a task."""
    from birdclaw.memory.tasks import task_registry
    from birdclaw.config import settings

    task = task_registry.get(task_id)
    if task is None:
        # Prefix match
        candidates = [t for t in task_registry.list() if t.task_id.startswith(task_id)]
        if len(candidates) == 1:
            task = candidates[0]
        elif len(candidates) > 1:
            return (
                f"Ambiguous prefix {task_id!r} — matches {len(candidates)} tasks. "
                "Provide more characters."
            )
        else:
            return f"Task {task_id!r} not found."

    lines = [
        f"task_id : {task.task_id}",
        f"title   : {task.title or '(untitled)'}",
        f"status  : {task.status}",
        f"prompt  : {task.prompt[:250]}",
    ]
    if getattr(task, "context", ""):
        lines.append(f"context : {task.context[:120]}")
    if getattr(task, "expected_outcome", ""):
        lines.append(f"outcome : {task.expected_outcome[:120]}")

    if task.output:
        lines.append(f"\noutput ({len(task.output)} chars):")
        lines.append(task.output[:3000])
        if len(task.output) > 3000:
            lines.append(f"… [{len(task.output) - 3000} chars truncated]")
    else:
        lines.append("\noutput: (none yet)")

    # Saved document from a write_doc task
    outputs_dir = settings.data_dir / "outputs"
    if outputs_dir.exists():
        docs = sorted(outputs_dir.glob(f"{task.task_id}*"))
        if docs:
            lines.append(f"\nsaved document: {docs[0]}")

    return "\n".join(lines)


def note_improvement(description: str, priority: str = "normal") -> str:
    """Log an improvement idea to the self-update backlog.

    Called during normal task execution when the agent notices a capability
    gap, a failure pattern, a fallback it hit, or a feature it wishes existed.
    The self-update cycle reads this backlog to pick the next improvement target.

    Args:
        description: What to improve and why — be specific about the file/behaviour.
        priority:    "high" | "normal" | "low"  (default: normal)
    """
    from birdclaw.config import settings as _s
    entry = {
        "ts": time.time(),
        "description": description[:500],
        "priority": priority if priority in ("high", "normal", "low") else "normal",
    }
    try:
        path = _s.self_update_todo_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("[note_improvement] logged: %s", description[:80])
        return f"Logged improvement ({priority}): {description[:120]}"
    except OSError as e:
        logger.warning("[note_improvement] write failed: %s", e)
        return f"Could not write to backlog: {e}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="search_tasks",
    description=(
        "Search past and current tasks by keyword. "
        "Use when the user references previous work, asks about a task status or result, "
        "or when context from an earlier session is needed to answer properly. "
        "Returns task IDs, titles, statuses, ages, and output excerpts."
    ),
    handler=search_tasks,
    input_schema={
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search across task prompts, titles, and outputs",
            },
            "status": {
                "type": "string",
                "enum": ["any", "running", "completed", "failed", "created"],
                "description": "Optional status filter (default: any)",
            },
        },
        "required": ["query"],
    },
    tags=["tasks", "search", "history", "status", "output", "find"],
))

registry.register(Tool(
    name="note_improvement",
    description=(
        "Log an improvement idea to BirdClaw's self-update backlog. "
        "Call this whenever you hit a capability gap, encounter a fallback, notice a bug, "
        "or wish a tool worked differently. Be specific: name the file and behaviour. "
        "The self-update cycle reads this backlog to choose what to fix next."
    ),
    handler=note_improvement,
    input_schema={
        "properties": {
            "description": {
                "type": "string",
                "description": "What to improve and why — include file name and specific behaviour",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "normal", "low"],
                "description": "Urgency of this improvement (default: normal)",
            },
        },
        "required": ["description"],
    },
    tags=["self-update", "improvement", "backlog", "meta", "feedback"],
))

registry.register(Tool(
    name="get_task_output",
    description=(
        "Get the full output, context, and saved document path for a specific task. "
        "Use the task_id (or prefix) from search_tasks results."
    ),
    handler=get_task_output,
    input_schema={
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Full task ID or unique prefix (e.g. 'task_1a2b')",
            },
        },
        "required": ["task_id"],
    },
    tags=["tasks", "output", "result", "details", "document"],
))
