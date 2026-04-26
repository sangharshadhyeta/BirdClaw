"""Per-stage step budget tracking.

Learns realistic budgets from empirical run history (P75 of past step counts).
Falls back to config defaults on first run or unknown stage types.

History file: ~/.birdclaw/memory/stage_history.jsonl
Each line: {"type": str, "steps": int, "goal_len": int, "ts": float}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from birdclaw.config import settings

logger = logging.getLogger(__name__)

# Schema offered when a stage is approaching its budget limit
REQUEST_BUDGET_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "request_budget",
        "description": (
            "Request additional steps for the current stage when significant work remains. "
            "Call this before the budget runs out — do NOT wait until forced."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "additional_steps": {
                    "type": "integer",
                    "description": "How many more steps you need (e.g. 20 for 10 more sections)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why: what work remains (e.g. '15 sections still to write')",
                },
            },
            "required": ["additional_steps", "reason"],
        },
    },
}


def history_path() -> Path:
    return settings.data_dir / "memory" / "stage_history.jsonl"


def historical_budget(stage_type: str) -> int:
    """Return P75 step count for stage_type from history, or config default.

    After enough runs the agent learns realistic budgets from empirical data.
    Falls back to settings.stage_budgets on first run or unknown types.
    """
    path = history_path()
    default = settings.stage_budgets.get(stage_type, 10)
    if not path.exists():
        return default
    samples: list[int] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == stage_type and isinstance(rec.get("steps"), int):
                samples.append(rec["steps"])
    except Exception as exc:
        logger.debug("stage_history read failed: %s", exc)
        return default
    if len(samples) < 3:
        return default
    samples.sort()
    p75_idx = int(len(samples) * 0.75)
    p75 = samples[min(p75_idx, len(samples) - 1)]
    return max(default, min(p75, 200))


def log_stage(stage_type: str, steps_taken: int, goal: str) -> None:
    """Append a completion record — feeds future historical_budget calls."""
    path = history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"type": stage_type, "steps": steps_taken, "goal_len": len(goal), "ts": time.time()}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:
        logger.debug("stage_history write failed: %s", exc)
