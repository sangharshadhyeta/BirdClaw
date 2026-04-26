"""Skill loader — reads SKILL.md files and selects one for a given query.

Skills are markdown runbooks that guide the agent through specific task types.
They override free model decisions with explicit step-by-step instructions.

Two skill locations (in priority order):
  1. ~/.birdclaw/skills/<name>/SKILL.md   — user-defined skills
  2. <package>/skills/<name>/SKILL.md     — built-in skills

Selection: token-overlap against each skill's `tags` frontmatter list,
falling back to `description`. Returns the best-matching skill text to
inject as extra_system, or None if no skill scores above threshold.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field  # noqa: F401 — field used in Skill dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Built-in skills live alongside this file
_BUILTIN_SKILLS_DIR = Path(__file__).parent

# Minimum token overlap to consider a skill relevant
_MATCH_THRESHOLD = 1


@dataclass
class StageInfo:
    """Parsed information for a single skill stage."""
    instruction: str         # the text to show the model
    next_tools: list[str]    # tools to show on the step immediately after this stage


@dataclass
class Skill:
    name: str
    description: str
    tags: list[str]
    body: str              # raw markdown body (below frontmatter)
    source: str            # file path, for debugging
    stages: dict[int, StageInfo] = field(default_factory=dict)
    # Standing-goal fields — optional, used by cron service
    schedule: str  = ""    # cron expr ("0 9 * * *") or "every:N" (seconds)
    enabled:  bool = True  # whether scheduled execution is active


def _parse_stages(body: str) -> dict[int, StageInfo]:
    """Parse embedded stage sections from skill body.

    Expects sections like:
        ## stage:1 plan
        Instruction text.
        next_tools: think

        ## stage:2 check
        ...

    Returns dict of {stage_num: StageInfo}.
    """
    result: dict[int, StageInfo] = {}
    parts = re.split(r"(?m)^## stage:(\d+)\s.*$", body)
    # parts alternates: [pre, num, text, num, text, ...]
    it = iter(parts)
    next(it)  # skip pre-header text
    for num_str, text in zip(it, it):
        try:
            num = int(num_str)
            text = text.strip()
            # Extract next_tools line (last line matching "next_tools: ...")
            next_tools: list[str] = []
            lines = []
            for line in text.splitlines():
                if line.startswith("next_tools:"):
                    raw = line.split(":", 1)[1].strip()
                    next_tools = [t.strip() for t in raw.split(",") if t.strip()]
                else:
                    lines.append(line)
            instruction = "\n".join(lines).strip()
            result[num] = StageInfo(instruction=instruction, next_tools=next_tools)
        except (ValueError, StopIteration):
            pass
    return result


# ---------------------------------------------------------------------------
# Frontmatter parser (tiny — no yaml dependency required)
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from body. Returns (meta_dict, body_text)."""
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 4:].strip()

    meta: dict[str, Any] = {}
    # Parse simple key: value lines (supports list values via [a,b,c] or YAML sequences)
    current_key: str | None = None
    list_items: list[str] = []

    for line in fm_text.splitlines():
        # List continuation (YAML sequence "-" style)
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            list_items.append(stripped[2:].strip().strip('"\''))
            continue

        # End of list
        if list_items and current_key and ":" in line and not line.startswith(" "):
            meta[current_key] = list_items
            list_items = []
            current_key = None

        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()

            # Inline list: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip('"\'') for x in val[1:-1].split(",") if x.strip()]
                meta[key] = items
            elif val:
                meta[key] = val.strip('"\'')
            else:
                # Start of a block list
                current_key = key
                list_items = []

    if list_items and current_key:
        meta[current_key] = list_items

    return meta, body


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_from_dir(skills_dir: Path) -> list[Skill]:
    """Load all SKILL.md files found inside skills_dir subdirectories."""
    skills: list[Skill] = []

    if not skills_dir.exists():
        return skills

    for skill_dir in sorted(skills_dir.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            text = skill_file.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            name = meta.get("name") or skill_dir.name
            description = meta.get("description") or ""
            tags_raw = meta.get("tags") or []
            tags = [str(t).lower() for t in tags_raw]

            schedule = meta.get("schedule") or ""
            enabled_raw = meta.get("enabled", "true")
            enabled = str(enabled_raw).lower() not in ("false", "0", "no")

            skills.append(Skill(
                name=name,
                description=description,
                tags=tags,
                body=body,
                source=str(skill_file),
                stages=_parse_stages(body),
                schedule=schedule,
                enabled=enabled,
            ))
            logger.debug("loaded skill %r from %s", name, skill_file)
        except Exception as e:
            logger.warning("failed to load skill from %s: %s", skill_file, e)

    return skills


def load_skills() -> list[Skill]:
    """Load all skills: user-defined first (higher priority), then built-in."""
    from birdclaw.config import settings

    user_skills_dir = settings.data_dir / "skills"
    builtin = _load_from_dir(_BUILTIN_SKILLS_DIR)
    user = _load_from_dir(user_skills_dir)

    # User skills take priority; deduplicate by name (user wins)
    by_name: dict[str, Skill] = {s.name: s for s in builtin}
    for s in user:
        by_name[s.name] = s

    return list(by_name.values())


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def select_skill(query: str, skills: list[Skill] | None = None) -> Skill | None:
    """Return the best-matching skill for the query, or None.

    Scoring: overlap between query tokens and skill tags + description tokens.
    Returns None if no skill scores >= _MATCH_THRESHOLD.
    """
    if skills is None:
        skills = load_skills()

    query_tokens = _tokenise(query)
    best: tuple[int, int, Skill | None] = (0, 0, None)

    for skill in skills:
        tag_tokens = set(skill.tags)
        desc_tokens = _tokenise(skill.description)
        score = len(query_tokens & (tag_tokens | desc_tokens))
        # Tie-break by name specificity (more tokens = more specific skill wins)
        name_len = len(_tokenise(skill.name))
        if (score, name_len) > (best[0], best[1]):
            best = (score, name_len, skill)

    score, _, skill = best
    if skill and score >= _MATCH_THRESHOLD:
        logger.info("skill selected: %r (score=%d)", skill.name, score)
        return skill

    logger.debug("no skill matched (best score=%d)", score)
    return None


def skill_context(query: str, skills: list[Skill] | None = None) -> str | None:
    """Return the skill body text to inject as extra_system, or None.

    This is the main entry point used by the agent loop.
    """
    skill = select_skill(query, skills)
    if skill is None:
        return None
    return f"## Active Skill: {skill.name}\n\n{skill.body}"


# ---------------------------------------------------------------------------
# Progressive disclosure helpers
# ---------------------------------------------------------------------------

def format_skill_metadata(skills: list[Skill] | None = None) -> str:
    """Compact one-liner per skill for soul injection (Tier 1 disclosure).

    Shows name + description only. The full runbook is loaded lazily via
    load_skill_body() when the soul decides to invoke a specific skill.
    """
    if skills is None:
        skills = load_skills()
    if not skills:
        return ""
    lines = ["Available skills (use read_skill(name) to load full instructions):"]
    for s in skills:
        lines.append(f"  • {s.name}: {s.description}")
    return "\n".join(lines)


def load_skill_body(name: str, skills: list[Skill] | None = None) -> str:
    """Return the full body of a named skill (Tier 2 disclosure).

    Case-insensitive name match. Returns an error string if not found.
    """
    if skills is None:
        skills = load_skills()
    needle = name.strip().lower()
    for s in skills:
        if s.name.lower() == needle:
            return f"## Skill: {s.name}\n\nSource: {s.source}\n\n{s.body}"
    available = ", ".join(s.name for s in skills) or "none"
    return f"No skill found with name '{name}'. Available: {available}"
