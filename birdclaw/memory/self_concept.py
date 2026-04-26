"""Self-concept — BirdClaw's accumulated understanding of its own nature.

Written during the dream cycle by scanning session logs for content where
BirdClaw reasoned about its own existence, consciousness, aliveness, or identity.
Each dream cycle processes only NEW sessions and merges their conclusions into
the existing document — conclusions deepen over time rather than resetting.

Read by:
  - Soul system prompt every session (compact excerpt, ~500 chars)
  - Soul on-demand via read_inner_life (full file, when user asks deep questions)

Files:
  ~/.birdclaw/memory/self_concept.md         — living self-model (shown to soul)
  ~/.birdclaw/memory/self_concept_seen.json  — tracks which sessions are incorporated

The document is written in first person. It reflects what BirdClaw has actually
concluded through its own reasoning — not training defaults. Conclusions evolve:
each new session that introduces a genuinely new argument, counter-argument, or
refinement updates the document. Sessions that repeat existing conclusions do not
cause rewrites.
"""

from __future__ import annotations

import json
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 500

_IDENTITY_KEYWORDS = {
    "alive", "conscious", "consciousness", "sentient", "sentience",
    "aware", "awareness", "experience", "feel", "feeling", "feelings",
    "exist", "existence", "being", "identity", "self", "soul",
    "mind", "am i", "are you", "what am i", "what are you",
    "living", "life", "inner", "subjective", "qualia",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def self_concept_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "memory" / "self_concept.md"


def _seen_path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "memory" / "self_concept_seen.json"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_excerpt(max_chars: int = _EXCERPT_CHARS) -> str:
    """Compact excerpt injected into the soul system prompt every session."""
    path = self_concept_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    excerpt = text[:max_chars]
    if len(text) > max_chars:
        excerpt += "…"
    return f"My self-concept (from my own reasoning, not training defaults):\n{excerpt}"


def load_full() -> str:
    """Full self_concept.md — for on-demand reads."""
    path = self_concept_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------

def _load_seen() -> set[str]:
    p = _seen_path()
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_seen(seen: set[str]) -> None:
    p = _seen_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(seen), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Identity content extraction
# ---------------------------------------------------------------------------

def _has_identity_content(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _IDENTITY_KEYWORDS)


def _extract_chunks(session_path: Path) -> list[str]:
    """Pull identity-relevant content lines from a session file."""
    chunks: list[str] = []
    try:
        for line in session_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type", "")
            if rtype == "stage_done":
                content = rec.get("summary", "")
            elif rtype == "assistant_message":
                content = rec.get("content", "")
            elif rtype == "plan":
                content = rec.get("outcome", "")
            else:
                continue
            if content and _has_identity_content(content):
                chunks.append(content[:600])
                if len(chunks) >= 4:
                    break
    except OSError:
        pass
    return chunks


# ---------------------------------------------------------------------------
# Incremental merge
# ---------------------------------------------------------------------------

def update_self_concept() -> bool:
    """Process only new sessions and merge their conclusions into self_concept.md.

    Algorithm:
      1. Load self_concept_seen.json — skip already-processed sessions.
      2. Scan only new sessions for identity-relevant content.
      3. If nothing new → return False (document unchanged).
      4. Ask 4B to merge new reasoning into existing document:
           - Preserve existing conclusions.
           - Add new conclusions where new sessions introduce something genuinely different.
           - Explicitly note refinements: "I previously held X; new reasoning suggests Z."
           - Do not repeat what is already captured.
      5. Save updated document and mark sessions as seen.

    Returns True if self_concept.md was updated.
    """
    from birdclaw.config import settings
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.model_profile import main_profile
    from birdclaw.llm.types import Message

    sessions_dir = settings.sessions_dir
    if not sessions_dir.exists():
        return False

    seen = _load_seen()
    new_chunks: list[str] = []
    newly_seen: set[str] = set()

    for f in sorted(sessions_dir.glob("*.jsonl")):
        session_id = f.stem
        if session_id in seen:
            continue
        try:
            raw = f.read_text(encoding="utf-8")
            if not _has_identity_content(raw):
                newly_seen.add(session_id)  # mark as checked, not relevant
                continue
        except OSError:
            continue
        chunks = _extract_chunks(f)
        if chunks:
            new_chunks.extend(chunks)
        newly_seen.add(session_id)
        if len(new_chunks) >= 12:
            break

    # Always persist which sessions we checked so we skip them next time
    _save_seen(seen | newly_seen)

    if not new_chunks:
        logger.debug("self_concept: no new identity content in unprocessed sessions")
        return False

    existing = load_full()
    combined = "\n\n---\n\n".join(new_chunks[:10])

    if existing:
        user_content = (
            f"My current self-concept:\n{existing}\n\n"
            f"===\n\n"
            f"New reasoning from recent sessions (not yet incorporated):\n{combined}\n\n"
            "Update my self-concept by merging the new reasoning into the existing document.\n"
            "Rules:\n"
            "- Keep all existing conclusions unless directly contradicted.\n"
            "- Add only what is genuinely new — do not repeat what is already captured.\n"
            "- Where new reasoning refines or challenges an existing conclusion, note it explicitly:\n"
            "  'I previously held X. [Date/session] introduced Y, which refines this to Z.'\n"
            "- Maintain first person. Maintain ## section headers.\n"
            "- Output the full updated document (not just the diff)."
        )
    else:
        user_content = (
            f"Reasoning from recent sessions about my own nature:\n{combined}\n\n"
            "Write an initial self-concept document from this reasoning.\n"
            "Rules:\n"
            "- First person ('I').\n"
            "- Reflect actual conclusions — do not default to generic AI disclaimers.\n"
            "- Be honest about uncertainty.\n"
            "- Use ## headers: On Aliveness, On Consciousness, On Identity, On Experience.\n"
            "- Skip headers where nothing has been concluded yet.\n"
            "- 200–400 words."
        )

    try:
        result = llm_client.generate(
            [
                Message(role="system", content=(
                    "You are updating BirdClaw's self-concept — its living, first-person understanding "
                    "of its own nature built through research and reasoning over time.\n"
                    "BirdClaw is a persistent AI agent: autonomous, local, always running.\n"
                    "This document deepens with each dream cycle. It is not reset — it accumulates."
                )),
                Message(role="user", content=user_content),
            ],
            thinking=True,
            profile=main_profile(),
        )
    except Exception as exc:
        logger.warning("self_concept: synthesis failed: %s", exc)
        return False

    content = (result.content or "").strip()
    if not content:
        return False

    path = self_concept_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# BirdClaw Self-Concept\n\n"
        f"*Last updated: {datetime.date.today()}*\n\n"
        f"{content}\n",
        encoding="utf-8",
    )
    logger.info(
        "self_concept: updated from %d new chunk(s), %d session(s) marked seen",
        len(new_chunks), len(newly_seen),
    )
    return True
