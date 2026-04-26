"""Persistent task list — one per user request.

Mirrors what the plan document does for the whole project, but scoped to a
single request. The orchestrator decomposes complex requests here; the loop
works through steps one at a time.

Storage: ~/.birdclaw/tasks/<request_id>.json

Usage:
    tl = decompose("write hello.py and run it")
    # → TaskList with steps: [write file, run file, answer user]

    for step in tl.pending():
        tl.mark_running(step.id)
        result = run_agent_loop(step.instruction, ...)
        tl.mark_done(step.id, result=result)

    tl.save()
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from birdclaw.config import settings

logger = logging.getLogger(__name__)

StepStatus = Literal["pending", "running", "done", "failed"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskStep:
    id: str
    description: str          # human-readable label
    instruction: str          # what to pass to the agent loop
    status: StepStatus = "pending"
    result: str = ""          # tool output or answer from this step
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "instruction": self.instruction,
            "status": self.status,
            "result": self.result,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskStep":
        return cls(
            id=d["id"],
            description=d["description"],
            instruction=d["instruction"],
            status=d.get("status", "pending"),
            result=d.get("result", ""),
            started_at=d.get("started_at", 0.0),
            ended_at=d.get("ended_at", 0.0),
            duration_ms=d.get("duration_ms", 0),
        )


@dataclass
class TaskList:
    request_id: str
    original_request: str
    steps: list[TaskStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""

    # ── Step access ──────────────────────────────────────────────────────────

    def pending(self) -> list[TaskStep]:
        return [s for s in self.steps if s.status == "pending"]

    def is_complete(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)

    def context_so_far(self) -> str:
        """Compact summary of completed steps — injected into each step's prompt."""
        lines = []
        for s in self.steps:
            if s.status == "done" and s.result:
                lines.append(f"- {s.description}: {s.result[:200]}")
        return "\n".join(lines)

    # ── Mutations ────────────────────────────────────────────────────────────

    def mark_running(self, step_id: str) -> None:
        s = self._get(step_id)
        s.status = "running"
        s.started_at = time.time()

    def mark_done(self, step_id: str, result: str = "") -> None:
        s = self._get(step_id)
        s.status = "done"
        s.result = result
        s.ended_at = time.time()
        if s.started_at:
            s.duration_ms = int((s.ended_at - s.started_at) * 1000)
        if self.is_complete():
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, step_id: str, reason: str = "") -> None:
        s = self._get(step_id)
        s.status = "failed"
        s.result = reason
        s.ended_at = time.time()
        if s.started_at:
            s.duration_ms = int((s.ended_at - s.started_at) * 1000)

    def _get(self, step_id: str) -> TaskStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"step {step_id!r} not found")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> Path:
        settings.ensure_dirs()
        tasks_dir = settings.data_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        path = tasks_dir / f"{self.request_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "original_request": self.original_request,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskList":
        tl = cls(
            request_id=d["request_id"],
            original_request=d["original_request"],
            created_at=d.get("created_at", ""),
            completed_at=d.get("completed_at", ""),
        )
        tl.steps = [TaskStep.from_dict(s) for s in d.get("steps", [])]
        return tl

    @classmethod
    def load(cls, request_id: str) -> "TaskList":
        path = settings.data_dir / "tasks" / f"{request_id}.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Complexity check
# ---------------------------------------------------------------------------

# Verbs that indicate the request requires action (not just knowledge retrieval)
_ACTION_VERBS = frozenset({
    "write", "create", "make", "build", "generate",
    "run", "execute", "install", "start", "launch",
    "edit", "modify", "change", "update", "fix", "refactor",
    "delete", "remove",
    "find", "search", "grep", "list",
    "read", "open", "show",
    "test", "check", "verify",
    "send", "post", "fetch", "download",
})


def is_complex(request: str) -> bool:
    """Return True if the request likely needs more than one tool call.

    Heuristic: count distinct action verbs. Two or more → decompose.
    Single-sentence factual questions → answer directly.
    """
    words = set(request.lower().split())
    action_count = len(words & _ACTION_VERBS)
    return action_count >= 3 or len(request.split()) > 30


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _make_step(description: str, instruction: str) -> TaskStep:
    return TaskStep(
        id=str(uuid.uuid4())[:8],
        description=description,
        instruction=instruction,
    )


def decompose(request: str) -> TaskList:
    """Break a request into atomic steps using the LLM.

    For simple requests, returns a single-step TaskList immediately
    without calling the model.

    For complex requests, asks the model to produce a numbered step list,
    then parses that into TaskSteps.
    """
    request_id = str(uuid.uuid4())[:12]

    if not is_complex(request):
        tl = TaskList(request_id=request_id, original_request=request)
        tl.steps = [_make_step("answer", request)]
        logger.debug("simple request — 1 step")
        return tl

    # Ask the model to decompose
    tl = _decompose_with_model(request, request_id)
    logger.debug("decomposed into %d steps", len(tl.steps))
    return tl


def _decompose_with_model(request: str, request_id: str) -> TaskList:
    """Call the LLM with think+answer tools to get a step list."""
    from birdclaw.agent.prompts import ANSWER_SCHEMA, THINK_SCHEMA
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.types import Message

    DECOMPOSE_PROMPT = (
        "List the steps needed to complete this task. "
        "Output a numbered list, one step per line, no prose. "
        "Each step must be a single concrete action. "
        "Maximum 4 steps — combine related actions into one step.\n\n"
        f"Task: {request}"
    )

    messages = [Message(role="user", content=DECOMPOSE_PROMPT)]
    tools = [ANSWER_SCHEMA, THINK_SCHEMA]

    result = llm_client.generate(messages, tools=tools, tool_choice="auto")

    raw_text = ""
    if result.tool_calls:
        tc = result.tool_calls[0]
        if tc.name == "answer":
            raw_text = tc.arguments.get("content", "")
        elif tc.name == "think":
            # Model thought but didn't answer — use thinking as steps
            raw_text = result.thinking
    if not raw_text:
        raw_text = result.content

    steps = _parse_step_list(raw_text, request)
    tl = TaskList(request_id=request_id, original_request=request)
    tl.steps = steps
    return tl


def _parse_step_list(text: str, original: str) -> list[TaskStep]:
    """Parse a numbered list into TaskStep objects."""
    import re
    lines = text.strip().splitlines()
    steps = []

    for line in lines:
        # Match "1. do something" or "- do something" or "* do something"
        m = re.match(r"^[\d]+[.)]\s*(.+)$", line.strip()) or \
            re.match(r"^[-*]\s*(.+)$", line.strip())
        if m:
            desc = m.group(1).strip()
            if desc:
                steps.append(_make_step(desc, desc))

    # Fallback: if parsing failed, treat whole request as one step
    if not steps:
        logger.warning("step parsing failed — falling back to single step")
        steps = [_make_step("complete task", original)]

    return steps
