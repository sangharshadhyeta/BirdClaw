"""User knowledge — facts and preferences about the user.

Persisted to ~/.birdclaw/user_knowledge.md.

Written by:
  - Soul loop via remember_user() tool (real-time, as the user reveals things)
  - Dreaming (retroactive extraction from old task outputs)

Read by:
  - Soul system prompt (compact excerpt, ≤12 bullets, ~200 tokens)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_CATEGORIES = ("facts", "preferences", "interests", "behaviors")
_MAX_PER_CATEGORY = 20   # bullets stored per category before pruning
_EXCERPT_ITEMS    = 12   # bullets injected into soul prompt


def _path() -> Path:
    from birdclaw.config import settings
    return settings.data_dir / "user_knowledge.md"


def _blank_doc() -> str:
    return (
        "# User Knowledge\n\n"
        "## facts\n\n"
        "## preferences\n\n"
        "## interests\n\n"
        "## behaviors\n"
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_excerpt(max_items: int = _EXCERPT_ITEMS) -> str:
    """Return a compact excerpt for injecting into the soul system prompt.

    behaviors bullets always appear first — they are interaction rules that
    must survive even when the file has many other entries.
    """
    path = _path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""

    # Split bullets by section so behaviors can be prioritised
    behaviors: list[str] = []
    others: list[str] = []
    current_section = ""
    for ln in text.splitlines():
        if ln.startswith("## "):
            current_section = ln[3:].strip()
        elif ln.startswith("- "):
            if current_section == "behaviors":
                behaviors.append(ln)
            else:
                others.append(ln)

    combined = behaviors + others
    if not combined:
        return ""

    lines = combined[:max_items]
    parts = []
    if behaviors:
        parts.append("Interaction rules (always follow):\n" + "\n".join(behaviors[:6]))
    remaining = [l for l in others if l in lines]
    if remaining:
        parts.append("About the user:\n" + "\n".join(remaining[:max_items - len(behaviors[:6])]))
    return "\n\n".join(parts)


def load_all() -> str:
    """Return the full user_knowledge.md content."""
    path = _path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def remember(fact: str, category: str = "facts") -> str:
    """Append a fact to the user knowledge file.

    Args:
        fact:     The fact or preference to record (one sentence).
        category: One of 'facts', 'preferences', 'interests'.

    Returns a short confirmation string.
    """
    fact = fact.strip()
    if not fact:
        return "Nothing to remember."

    if category not in _CATEGORIES:
        category = "facts"

    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)

    content = path.read_text(encoding="utf-8") if path.exists() else _blank_doc()

    # Deduplicate: skip if a very similar fact is already present (simple substring)
    fact_lower = re.sub(r"\s+", " ", fact.lower())
    existing_bullets = [ln.lstrip("- ").strip().lower() for ln in content.splitlines() if ln.startswith("- ")]
    for existing in existing_bullets:
        if fact_lower in existing or existing in fact_lower:
            return f"Already known: {fact!r}"

    # Ensure the section exists
    section_header = f"## {category}"
    if section_header not in content:
        content = content.rstrip() + f"\n\n{section_header}\n"

    # Insert the fact immediately after the section header line
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    inserted = False
    for i, line in enumerate(lines):
        result.append(line)
        if not inserted and line.strip() == section_header:
            result.append(f"- {fact}\n")
            inserted = True

    if not inserted:
        result.append(f"- {fact}\n")

    path.write_text("".join(result), encoding="utf-8")
    logger.info("user_knowledge: [%s] %r", category, fact[:70])
    return f"Remembered ({category}): {fact!r}"
