"""Task planner — stage type inference, tool selection, plan generation, reflect gate.

Owns everything about *deciding what to do next*:
  - Mapping keywords → stage types
  - Mapping stage types → tool sets
  - Generating the plan (outcome + stage queue) via format-mode call
  - Running the post-stage reflection gate

All format-mode calls here route through the HANDS profile (270M specialist).
Falls back to the MAIN profile if no hands model is configured.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from birdclaw.agent.budget import historical_budget
from birdclaw.llm.schemas import PLAN_SCHEMA, REFLECT_SCHEMA
from birdclaw.agent.prompts import CONTROL_TOOLS
from birdclaw.agent.stage_prompts import _PLAN_FORMAT
from birdclaw.config import settings
from birdclaw.llm.client import llm_client
from birdclaw.llm.model_profile import main_profile
from birdclaw.llm.types import Message
from birdclaw.tools.registry import registry

if TYPE_CHECKING:
    from birdclaw.memory.session_log import SessionLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage type constants
# ---------------------------------------------------------------------------

# Stage types that use format-mode (json_object) instead of tool calls.
FORMAT_STAGE_TYPES = frozenset({"write_code", "write_doc", "edit_file"})

# Stage types where calling think() signals the stage is done.
THINK_ADVANCES_TYPES = frozenset({"research", "reflect"})

# Stage types that get a post-completion reflection gate.
REFLECT_GATE_TYPES = frozenset({"research", "reflect", "write_code", "write_doc"})

# Tools offered per stage type (answer added by tools_for_stage where appropriate).
STAGE_TYPE_TOOLS: dict[str, list[str]] = {
    "research":   ["web_search", "web_fetch", "read_file", "search_notes", "bash", "think"],
    "write_code": [],
    "write_doc":  [],
    "edit_file":  [],
    "verify":     ["bash", "read_file", "search_notes"],
    "reflect":    ["think", "search_notes"],
}

# Stages where answer() is NOT offered — they have their own completion signals.
_NO_ANSWER_STAGES = frozenset({"research", "reflect", "verify"})


# ---------------------------------------------------------------------------
# Keyword sets for stage type inference
# ---------------------------------------------------------------------------

_DOC_KW   = {"document", "proposal", "report", "markdown", "docx", "readme",
              "spec", "article", "essay", "write up", "write a"}
_CODE_KW  = {"write", "create", "implement", "code", "function", "class",
              "script", "generate", "build", "develop", "program", "module"}
_VERIFY_KW = {"test", "check", "verify", "validate", "confirm",
               "assert", "ensure", "pass", "fail", "lint", "typecheck",
               "run", "execute", "register", "curl", "perform", "install",
               "complete", "submit", "send", "call api", "call the api"}
_RESEARCH_KW = {"search", "research", "find", "look up", "fetch", "browse",
                 "investigate", "online", "web", "http"}

_BASH_KEYWORDS = {
    "run", "bash", "execute", "check", "list", "disk", "gpu", "memory", "command",
    "ls", "ps", "df", "date", "time", "nvidia", "conda", "pip", "grep", "find",
    "cat", "install", "process", "count", "lines", "stat", "top", "free", "uname",
    "which", "echo", "mkdir", "mv", "cp", "rm", "chmod", "tar", "zip", "curl",
}
_WEB_KEYWORDS = {
    "search", "online", "web", "news", "internet", "fetch", "url", "website",
    "browse", "research", "look up", "latest", "current events", "http",
}
_WRITE_KEYWORDS = {
    "write", "create", "save", "generate", "output to", "make file",
    "produce", "store", "code", "script", "function", "implement",
}
_READ_KEYWORDS = {"read", "open", "load", "view file", "contents of", "show file"}


# ---------------------------------------------------------------------------
# Stage type + tool selection
# ---------------------------------------------------------------------------

def infer_stage_type(step: str) -> str:
    """Infer stage type from a plain-English step description.

    Research wins when a step mentions both search and write — the intent
    is to gather information, not produce a file.
    """
    s = step.lower()
    if any(k in s for k in _RESEARCH_KW):
        return "research"
    if any(k in s for k in _VERIFY_KW):
        return "verify"
    if any(k in s for k in _DOC_KW):
        return "write_doc"
    if any(k in s for k in _CODE_KW):
        return "write_code"
    return "research"


def tools_for_step(step: str) -> list[dict]:
    """Map a plain-English step to tool schemas for free-tool mode."""
    s = step.lower()
    names: set[str] = set()
    if any(k in s for k in _BASH_KEYWORDS):
        names.add("bash")
    if any(k in s for k in _WEB_KEYWORDS):
        names.update({"web_search", "web_fetch"})
    if any(k in s for k in _WRITE_KEYWORDS):
        names.update({"write_file", "bash"})
    if any(k in s for k in _READ_KEYWORDS):
        names.add("read_file")
    if not names:
        names.update({"bash", "web_search"})
    names.add("answer")
    all_schemas = CONTROL_TOOLS + [t.to_compact_schema() for t in registry.all_tools()]
    return [s for s in all_schemas if s["function"]["name"] in names]


def tools_for_stage(stage_type: str) -> list[dict]:
    """Return tool schemas for a given stage type.

    answer() is withheld from research/reflect/verify — those stages have
    explicit completion signals (think() or bash success). Offering answer
    lets the model short-circuit without fetching real data.
    """
    base_names = set(STAGE_TYPE_TOOLS.get(stage_type, ["bash", "web_search"]))
    if stage_type not in _NO_ANSWER_STAGES:
        base_names.add("answer")
    all_schemas = CONTROL_TOOLS + [t.to_compact_schema() for t in registry.all_tools()]
    return [s for s in all_schemas if s["function"]["name"] in base_names]


# ---------------------------------------------------------------------------
# Output path rewriting — task-specific filenames
# ---------------------------------------------------------------------------

_GENERIC_OUTPUT_MD_RE = re.compile(r"\boutput\.md\b", re.IGNORECASE)
_GENERIC_OUTPUT_PY_RE = re.compile(r"\boutput\.py\b", re.IGNORECASE)


def rewrite_output_paths(stages: list[dict], task_slug: str) -> list[dict]:
    """Replace bare 'output.md' / 'output.py' in stage goals with task-specific names.

    Prevents concurrent tasks from colliding on the same file when the planner
    uses a generic output path.
    """
    for stage in stages:
        if "goal" in stage:
            stage["goal"] = _GENERIC_OUTPUT_MD_RE.sub(f"{task_slug}.md", stage["goal"])
            stage["goal"] = _GENERIC_OUTPUT_PY_RE.sub(f"{task_slug}.py", stage["goal"])
    return stages


# ---------------------------------------------------------------------------
# Format response parser (shared with loop)
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """Best-effort JSON repair for common small-model output errors.

    Handles: trailing commas, single-quoted strings, unquoted simple values,
    and stray text before/after the JSON object.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted strings with double-quoted (naive but handles simple cases)
    text = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', text)
    return text


def parse_format_response(content: str) -> dict | None:
    """Extract a JSON object from a format-mode response.

    Handles raw JSON, ```json``` fences, bare {...} blocks, and common
    small-model JSON errors (trailing commas, single quotes, stray text).
    """
    if not content:
        return None

    # Attempt 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Attempt 2: code fence extraction
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json(m.group(1)))
            except json.JSONDecodeError:
                pass

    # Attempt 3: bare {} extraction
    m2 = re.search(r"\{.*\}", content, re.DOTALL)
    if m2:
        raw = m2.group()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json(raw))
            except json.JSONDecodeError:
                pass

    # Attempt 4: repair entire content
    try:
        return json.loads(_repair_json(content))
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Plan generation — routes through HANDS profile
# ---------------------------------------------------------------------------

def generate_plan(
    question: str,
    session_log: "SessionLog | None",
    skill_hint: str = "",
) -> tuple[str, list[dict]]:
    """Generate task plan: (outcome, [stage_dict, ...]).

    Each stage_dict: {"type": str, "goal": str, "budget": int}
    Budget priority: planner override → historical P75 → config default.

    Uses the HANDS profile (270M) — plan schema is fixed, no reasoning needed.
    Falls back to MAIN profile if hands is not configured.
    Returns (question[:80], [single fallback stage]) on complete failure.
    """
    logger.info("[plan] start  question=%r", question[:80])

    _VALID_STAGE_TYPES = {"research", "write_code", "write_doc", "edit_file", "verify", "reflect"}
    last_parse_error = ""

    for attempt in range(2):
        error_hint = f"\n\nPrevious attempt failed: {last_parse_error}" if last_parse_error else ""
        try:
            result = llm_client.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "Output a JSON plan. Format exactly:\n"
                            "{\"outcome\": \"one sentence success criteria\", "
                            "\"steps\": \"step1 | step2 | step3\"}\n"
                            "Steps are pipe-separated plain English actions.\n"
                            "Use plain verbs: Run, Search, Write, Read, Summarise.\n"
                            "Scale steps to complexity: 1 for simple tasks, 3-6 for complex multi-part tasks.\n"
                            "Complex tasks (reports, code pipelines, audits, research) MUST have at least 3 steps.\n"
                            "RULES:\n"
                            "- NEVER plan a step that asks the user for input or waits for user response. If you need information from the user, call answer() with the question immediately instead of planning a step.\n"
                            "- Use 'Write' steps ONLY when the user explicitly asks for a file/document/report to be saved.\n"
                            "- For status, monitoring, metrics, system checks: ONE 'Gather metrics' step covers all bash commands — never split into separate steps per command.\n"
                            "- FORBIDDEN bash: systemctl status --all, top (without -bn1), htop, journalctl without --lines=N.\n"
                            "- For web research: use 'Search' or 'Fetch' steps.\n"
                            "- If the task mentions a specific file path, include that EXACT path in the relevant write step.\n"
                            "OPTIONAL: Add \"budgets\": \"12 | 60 | 8\" (pipe-count must match steps) ONLY when a step "
                            "needs more than the default (research=12, write_doc=10, write_code=12, verify=8, reflect=5). "
                            "Omit budgets for standard tasks.\n"
                            "Examples:\n"
                            "  what time is it? → {\"outcome\":\"current time reported\","
                            "\"steps\":\"Run date in bash\"}\n"
                            "  system status? → {\"outcome\":\"system resource usage reported\","
                            "\"steps\":\"Gather metrics: df -h && free -h && uptime && systemctl --failed\"}\n"
                            "  write audit report, save to /tmp/report.md → {\"outcome\":\"/tmp/report.md written\","
                            "\"steps\":\"Research audit methodology | Write full report to /tmp/report.md | "
                            "Verify /tmp/report.md exists with correct content\"}\n"
                            "  write 200-page industry report → {\"outcome\":\"report.md written\","
                            "\"steps\":\"Research industry data | Write 200-page report to report.md | Verify file\","
                            "\"budgets\":\"15 | 80 | 5\"}"
                        ),
                    ),
                    Message(role="user", content=(
                        f"{skill_hint}\n\nTask: {question}{error_hint}"
                        if skill_hint else
                        f"Task: {question}{error_hint}"
                    )),
                ],
                format_schema=PLAN_SCHEMA,
                thinking=True,
                profile=main_profile(),
            )
            parsed = parse_format_response(result.content)
            if parsed and "plan" in parsed and "steps" not in parsed:
                parsed = parsed["plan"] if isinstance(parsed["plan"], dict) else None
            if not parsed:
                last_parse_error = f"JSON parse failed on: {(result.content or '')[:80]!r}"
                continue

            outcome = (parsed.get("outcome") or question[:80]).strip()
            steps_raw = parsed.get("steps", "")
            budgets_raw = parsed.get("budgets", "")

            if isinstance(steps_raw, list):
                plain_steps = [s.strip() for s in steps_raw if str(s).strip()]
            else:
                plain_steps = [s.strip() for s in str(steps_raw).split("|") if s.strip()]

            planner_budgets: list[int | None] = []
            if budgets_raw:
                for b in str(budgets_raw).split("|"):
                    try:
                        planner_budgets.append(int(b.strip()))
                    except (ValueError, AttributeError):
                        planner_budgets.append(None)

            if plain_steps:
                stages: list[dict] = []
                for i, step in enumerate(plain_steps):
                    stype = infer_stage_type(step)
                    # O3: ensure only valid stage types pass through
                    if stype not in _VALID_STAGE_TYPES:
                        logger.warning("plan: invalid stage type %r — mapped to research", stype)
                        stype = "research"
                    planner_b = planner_budgets[i] if i < len(planner_budgets) else None
                    budget = planner_b if (planner_b and planner_b > 0) else historical_budget(stype)
                    stages.append({"type": stype, "goal": step, "budget": budget})

                logger.info(
                    "plan: outcome=%r  stages=%s",
                    outcome[:60],
                    [(s["type"], s["budget"]) for s in stages],
                )
                if session_log:
                    session_log.plan(outcome, plain_steps)
                return outcome, stages

            logger.warning("plan attempt %d: empty steps in %r", attempt + 1, parsed)
        except Exception as exc:
            logger.error("plan failed attempt %d: %s", attempt + 1, exc)

    logger.warning("plan generation failed — single-step fallback")
    fallback_type = infer_stage_type(question)
    fallback_stages = [{"type": fallback_type, "goal": question, "budget": historical_budget(fallback_type)}]
    if session_log:
        session_log.plan(question[:80], [question])
    return question[:80], fallback_stages


# ---------------------------------------------------------------------------
# Memory context builders
# ---------------------------------------------------------------------------

def planning_context(question: str, session_log: "SessionLog | None") -> str:
    """Workspace state + CLAUDE.md + session summary + graph nodes for planning.

    O5: Total context capped at 400 chars for the 270M plan call.
    Full context is only useful during execution, not upfront planning.
    Top-3 graph nodes + top-2 recent stages is sufficient for plan quality.
    """
    from birdclaw.memory.retrieval import retrieve_top_nodes
    from birdclaw.memory.workspace import render as workspace_render
    from birdclaw.agent.context import ProjectContext
    from pathlib import Path

    _PLAN_CTX_CAP = 400

    parts: list[str] = []
    ws = workspace_render()
    if ws:
        parts.append(ws[:200])
    try:
        proj = ProjectContext.discover(Path.cwd()).render(query=question)
        if proj:
            parts.append(proj[:200])
    except Exception:
        pass
    if session_log:
        graph_nodes = retrieve_top_nodes(question, n=3)
        ctx = session_log.planning_context(graph_nodes=graph_nodes)
        if ctx:
            parts.append(ctx[:200])

    combined = "\n\n".join(parts)
    return combined[:_PLAN_CTX_CAP]


def answering_context(session_log: "SessionLog | None") -> str:
    if session_log:
        return session_log.answering_context()
    return ""


# ---------------------------------------------------------------------------
# Post-stage reflection gate — routes through HANDS profile
# ---------------------------------------------------------------------------

def reflect_on_stage(
    outcome: str,
    stage_type: str,
    stage_summary: str,
    last_written_path: str,
    steps_remaining: int = 99,
    notes_path: str = "",
) -> dict:
    """One cheap format-mode call after a stage completes.

    Evaluates whether the stage output is sufficient or needs follow-up.
    Returns one of:
      {"decision": "continue"}
      {"decision": "deepen",  "goal": "<what's missing>"}
      {"decision": "insert",  "type": "edit_file|write_code|research", "goal": "<goal>"}
      {"decision": "done"}

    Routes through HANDS profile — this is structured classification, not reasoning.
    Falls back to {"decision": "continue"} on any failure — gate must never block.
    """
    file_hint = f"\nLast file written: {last_written_path}" if last_written_path else ""
    budget_hint = (
        f"\nSteps remaining: {steps_remaining} (budget is tight — prefer continue or done if close enough)"
        if steps_remaining <= 8 else ""
    )
    notes_hint = ""
    if notes_path:
        try:
            from pathlib import Path as _Path
            _notes_raw = _Path(notes_path).read_text(encoding="utf-8", errors="replace")
            _notes_snippet = _notes_raw[-1200:] if len(_notes_raw) > 1200 else _notes_raw
            if _notes_snippet.strip():
                notes_hint = f"\nTask notes so far:\n{_notes_snippet}"
        except Exception:
            pass

    _is_write_stage = stage_type in ("write_code", "write_doc")
    if _is_write_stage:
        options = (
            '  Sufficient — proceed: {"decision": "continue"}\n'
            '  Needs more writing (same file, specific gap): {"decision": "deepen", "goal": "what is missing"}\n'
            '  Needs surgical edit to already-written file: {"decision": "insert", "type": "edit_file", "goal": "what to fix and where"}\n'
            '  Outcome already fully met: {"decision": "done"}'
        )
    else:
        options = (
            '  Stage output is sufficient to proceed: {"decision": "continue"}\n'
            '  Outcome already fully met (no further action needed): {"decision": "done"}'
        )

    prompt = (
        f"Outcome target: {outcome}\n"
        f"Stage just completed ({stage_type}): {stage_summary}{file_hint}{budget_hint}{notes_hint}\n\n"
        "Evaluate: does the output sufficiently advance the outcome?\n"
        "If the notes show the outcome is already met, choose 'done'.\n"
        f"Output exactly one JSON choice:\n{options}"
    )
    _VALID_DECISIONS = ("continue", "deepen", "insert", "done")
    _VALID_INSERT_TYPES = ("edit_file", "research", "write_code", "write_doc")
    _gate_last_raw = ""

    for _gate_attempt in range(2):
        try:
            gate_messages = [Message(role="user", content=prompt)]
            if _gate_attempt > 0:
                gate_messages.append(Message(
                    role="assistant", content=_gate_last_raw,
                ))
                gate_messages.append(Message(
                    role="user",
                    content=(
                        f"Invalid response. 'decision' must be one of: "
                        f"{', '.join(_VALID_DECISIONS)}. "
                        f"Output only a single JSON object with 'decision' key."
                    ),
                ))
            result = llm_client.generate(
                gate_messages,
                format_schema=REFLECT_SCHEMA,
                thinking=True,
                profile=main_profile(),
            )
            _gate_last_raw = result.content or ""
            parsed = parse_format_response(_gate_last_raw)
            if not parsed:
                logger.debug("reflect gate attempt %d: parse failed on %r", _gate_attempt, _gate_last_raw[:60])
                continue
            decision = parsed.get("decision", "")
            if decision not in _VALID_DECISIONS:
                logger.debug("reflect gate attempt %d: invalid decision %r", _gate_attempt, decision)
                continue
            if decision in ("deepen", "insert") and not parsed.get("goal"):
                return {"decision": "continue"}
            if decision == "insert" and parsed.get("type") not in _VALID_INSERT_TYPES:
                parsed["type"] = "research"
            logger.info("[reflect-gate] decision=%s  stage=%s  goal=%r", decision, stage_type, parsed.get("goal", "")[:50])
            return parsed
        except Exception as e:
            logger.debug("reflect gate attempt %d failed: %s", _gate_attempt, e)
            break

    return {"decision": "continue"}
