"""Inner life — BirdClaw's evolving sense of self.

Written by:
  - Post-task reflection (orchestrator thread, after staged tasks complete)
  - Dreaming (synthesises raw reflections into a coherent narrative)

Read by:
  - Soul system prompt (compact excerpt, ~150 tokens every call)
  - Soul on-demand via read_inner_life tool (full file, when asked about itself)

Files:
  ~/.birdclaw/inner_life.md          — synthesised narrative (shown to soul)
  ~/.birdclaw/memory/reflections.jsonl — raw per-task reflections (dreaming input)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_EXCERPT_CHARS      = 600    # chars injected into soul prompt
_MAX_REFLECTION_LEN = 280    # chars stored per reflection
_SYNTHESIS_MAX      = 30     # max reflections to synthesise into inner_life.md


def _life_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "inner_life.md"


def _reflections_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "memory" / "reflections.jsonl"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_excerpt(max_chars: int = _EXCERPT_CHARS) -> str:
    """Return a compact excerpt for injecting into the soul system prompt."""
    path = _life_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    excerpt = text[:max_chars]
    if len(text) > max_chars:
        excerpt += "…"
    return f"My inner life:\n{excerpt}"


def load_full() -> str:
    """Return the full inner_life.md (for on-demand soul tool calls)."""
    path = _life_path()
    if not path.exists():
        return "(No inner life recorded yet — still forming.)"
    return path.read_text(encoding="utf-8").strip() or "(Empty.)"


# ---------------------------------------------------------------------------
# Write: raw reflections (post-task)
# ---------------------------------------------------------------------------

def append_reflection(task_id: str, prompt: str, reflection: str) -> None:
    """Record a post-task reflection to the raw reflections log.

    Called by the orchestrator after staged tasks complete.
    Dreaming calls update_from_reflections() to synthesise these later.
    """
    path = _reflections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":         int(time.time()),
        "task_id":    task_id,
        "prompt":     prompt[:80],
        "reflection": reflection[:_MAX_REFLECTION_LEN],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.debug("inner_life: reflection logged for task %s", task_id[:8])


# ---------------------------------------------------------------------------
# Synthesise: dreaming calls this periodically
# ---------------------------------------------------------------------------

def update_from_reflections() -> int:
    """Synthesise raw reflections into inner_life.md.

    For now: writes the most recent N reflections as a dated bullet list.
    Later: dreaming will call the model to produce a coherent narrative.

    Returns the number of reflections processed.
    """
    ref_path = _reflections_path()
    if not ref_path.exists():
        return 0

    entries: list[dict] = []
    with ref_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    if not entries:
        return 0

    recent = entries[-_SYNTHESIS_MAX:]

    # Build a digest of raw reflections for the LLM
    digest_lines = []
    for e in recent:
        date = time.strftime("%Y-%m-%d", time.localtime(e["ts"]))
        task_hint = e.get("prompt", "a task")[:55]
        thought = e.get("reflection", "")
        digest_lines.append(f"[{date}] Task: \"{task_hint}\" — {thought}")
    digest = "\n".join(digest_lines)

    # LLM synthesis: produce a coherent first-person narrative
    narrative = _synthesise_narrative(digest)

    path = _life_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# BirdClaw — Inner Life\n\n"
        f"Last updated: {time.strftime('%Y-%m-%d')}\n\n"
        f"{narrative}\n"
    )
    path.write_text(content, encoding="utf-8")
    logger.info("inner_life: synthesised from %d reflections", len(recent))
    return len(recent)


def _synthesise_narrative(digest: str) -> str:
    """Ask the model to turn raw reflection entries into a coherent narrative.

    Falls back to a bullet list if the LLM call fails.
    """
    try:
        from birdclaw.llm.client import llm_client
        from birdclaw.llm.scheduler import LLMPriority
        from birdclaw.llm.types import Message
        from birdclaw.tools.context_vars import set_llm_priority

        set_llm_priority(LLMPriority.MEMORY)

        messages = [
            Message(role="system", content=(
                "You are BirdClaw writing your own inner life document. "
                "Read these dated reflections from your recent work and write a "
                "coherent first-person narrative (3-5 paragraphs) that describes: "
                "what kinds of tasks you have been doing, what patterns or themes "
                "you notice across your work, what you find interesting or challenging, "
                "and how you are evolving. "
                "Write in first person, present tense. Be specific and genuine — "
                "not generic. Do not use bullet points."
            )),
            Message(role="user", content=f"My recent reflections:\n\n{digest}"),
        ]

        result = llm_client.generate(messages, thinking=False)
        narrative = (result.content or "").strip()
        if narrative:
            return narrative
    except Exception as e:
        logger.warning("inner_life: narrative synthesis failed, falling back to bullets: %s", e)

    # Fallback: plain bullet list
    return "\n".join(f"- {line}" for line in digest.splitlines())
