"""Skill tool — on-demand multi-stage skill loading (claw-code pattern).

Instead of injecting skill content into the system prompt (context overshoot),
skills are fetched as a tool call — one stage at a time.

The model calls use_skill("code-generation") to get the stage list, then
use_skill("code-generation", stage=1) to get stage 1 instructions, etc.
Each stage is short (~200 chars) so context stays minimal for small models.

Matches claw-code-parity pattern:
  tools/src/lib.rs → execute_skill() → reads SKILL.md → returns as tool result
"""

from __future__ import annotations

from birdclaw.skills.loader import load_skills, select_skill
from birdclaw.tools.registry import Tool, registry

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _use_skill(name: str | None = None, stage: int | str | None = None, skill_name: str | None = None, **_extra) -> str:
    """Accept both 'name' and 'skill_name' — gemma4 uses skill_name naturally.
    Also coerces stage to int since gemma4 sometimes passes it as a string.
    """
    name = name or skill_name
    if not name:
        return "Error: provide 'skill_name' parameter (e.g. 'code-generation')."
    # Coerce stage to int — gemma4 sometimes passes "3" instead of 3
    if stage is not None:
        try:
            stage = int(stage)
        except (TypeError, ValueError):
            stage = None
    # Strip $ prefix (claw-code convention)
    if name.startswith("$"):
        name = name[1:]

    skills = load_skills()
    skill_map = {s.name: s for s in skills}

    # Exact match first
    skill = skill_map.get(name)
    if skill is None:
        # Fuzzy: substring match
        lower = name.lower()
        for sname, s in skill_map.items():
            if lower in sname.lower() or sname.lower() in lower:
                skill = s
                break

    # Fall back to token-overlap selection
    if skill is None:
        skill = select_skill(name, skills)

    if skill is None:
        available = ", ".join(skill_map.keys()) or "(none installed)"
        return f"Skill '{name}' not found. Available: {available}"

    # No stage specified — auto-start at stage 1
    if stage is None:
        if not skill.stages:
            return f"## Skill: {skill.name}\n\n{skill.body}"
        stage = 1  # default to first stage

    # Specific stage requested
    if not skill.stages:
        return f"Skill '{skill.name}' has no stages defined. Full body:\n\n{skill.body}"

    stage_info = skill.stages.get(stage)
    if stage_info is None:
        available = sorted(skill.stages.keys())
        return f"Stage {stage} not found in '{skill.name}'. Available stages: {available}"

    return f"## {skill.name} — Stage {stage}\n\n{stage_info.instruction}"


def _list_skills() -> str:
    """List all available skills with their descriptions."""
    skills = load_skills()
    if not skills:
        return "No skills installed."
    lines = ["Available skills:"]
    for s in sorted(skills, key=lambda x: x.name):
        desc = s.description or "(no description)"
        n_stages = len(s.stages) if s.stages else 0
        stage_note = f" ({n_stages} stages)" if n_stages else ""
        lines.append(f"  {s.name}{stage_note}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="use_skill",
    description=(
        "Get step-by-step instructions for a task. "
        "Call use_skill('code-generation') to start writing code. "
        "Call use_skill('document-creation') to start writing a document. "
        "Add stage=N to get that stage's instruction."
    ),
    input_schema={
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Skill name: 'code-generation' or 'document-creation'.",
            },
            "stage": {
                "type": "integer",
                "description": "Stage number (1-based). Omit to get the stage list.",
            },
        },
        "required": ["skill_name"],
    },
    handler=_use_skill,
    tags=["skill", "plan", "runbook", "steps", "guide", "how", "code", "document"],
))

registry.register(Tool(
    name="list_skills",
    description="List all available skills with their descriptions.",
    input_schema={"properties": {}, "required": []},
    handler=lambda: _list_skills(),
    tags=["skill", "list", "available"],
))
