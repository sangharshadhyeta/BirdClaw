from __future__ import annotations
import json
import logging
import re
from typing import Any, Literal, Optional

from birdclaw.llm.schemas import SUBTASK_PLAN_SCHEMA

from .subtask_manifest import SubtaskItem, SubtaskManifest

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#{1,3} .+", re.MULTILINE)
_DEF_RE     = re.compile(r"^(?:def |class )\w+", re.MULTILINE)


def _summarise_existing(content: str) -> str:
    """O1: Extract headings/def lines + last 200 chars instead of 1200 chars of raw content."""
    lines: list[str] = []
    for m in _HEADING_RE.finditer(content):
        lines.append(m.group().strip())
    for m in _DEF_RE.finditer(content):
        lines.append(m.group().strip())
    tail = content.strip()[-200:] if len(content) > 200 else ""
    summary = "\n".join(lines)
    if tail and tail not in summary:
        summary = (summary + "\n...\n" + tail).strip()
    return summary[:600] or content[:300]


_PLAN_SYSTEM = (
    "You are a task planner. Given a writing goal and optional existing file content, "
    "output a JSON subtask list.\n\n"
    "For documents: each subtask is one section with a heading.\n"
    "For code: each subtask is one function or class.\n\n"
    "Rules:\n"
    "- Output ONLY valid JSON, no explanation.\n"
    "- For docs: 'anchor' is the heading text (without ##).\n"
    "- For code: 'anchor' is just the function or class name (no def/class prefix).\n"
    "- 'min_chars' is the minimum expected body length in characters.\n"
    "- 'kind' is one of: section, function, class, test.\n"
    "- Order matters: list items in logical writing order.\n"
    "- Do NOT include items already marked complete in existing content.\n\n"
    'Output format:\n{"subtasks": [\n'
    '  {"title": "Executive Summary", "anchor": "Executive Summary", "kind": "section", "min_chars": 300},\n'
    "  ...\n]}"
)

_REPLAN_SYSTEM = (
    "You are a task replanner. The current manifest is incomplete. "
    "Add the missing subtasks only.\n"
    "Output ONLY valid JSON in the same format. Do NOT include already-complete items.\n\n"
    '{"subtasks": [...new items only...]}'
)


def _parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        return data.get("subtasks", [])
    except json.JSONDecodeError as exc:
        logger.warning("SubtaskPlanner JSON error: %s — raw: %.200s", exc, raw)
        return []


def _items_from_raw(raw_items: list[dict], start_index: int = 0) -> list[SubtaskItem]:
    items = []
    for i, r in enumerate(raw_items):
        title = str(r.get("title", f"Item {start_index + i}"))
        anchor = str(r.get("anchor", title))
        kind = r.get("kind", "section")
        if kind not in ("section", "function", "class", "test"):
            kind = "section"
        min_chars = int(r.get("min_chars", 200))
        items.append(SubtaskItem(
            index=start_index + i,
            title=title,
            anchor=anchor,
            kind=kind,
            expected_min_chars=max(80, min_chars),
        ))
    return items


def plan(
    llm_client: Any,
    stage_goal: str,
    file_path: str,
    file_type: Literal["doc", "code"],
    existing_content: str = "",
) -> SubtaskManifest:
    existing_note = ""
    if existing_content.strip():
        # O1: send only headings + tail instead of full content — ~70% token cut
        existing_note = (
            "\n\nExisting content (headings only — do not re-plan completed sections):\n"
            + _summarise_existing(existing_content)
        )

    prompt = (
        f"Stage goal: {stage_goal}\n"
        f"File: {file_path}\n"
        f"File type: {file_type}{existing_note}"
    )

    from birdclaw.llm.types import Message
    messages = [
        Message(role="system", content=_PLAN_SYSTEM),
        Message(role="user", content=prompt),
    ]

    result = llm_client.generate(messages, format_schema=SUBTASK_PLAN_SCHEMA, thinking=True)
    raw_items = _parse_response(result.content or "")

    if not raw_items:
        logger.warning("SubtaskPlanner returned no items — using single-item fallback")
        default_kind: Literal["function", "section"] = "function" if file_type == "code" else "section"
        raw_items = [{"title": stage_goal[:60], "anchor": stage_goal[:60], "kind": default_kind, "min_chars": 400}]

    return SubtaskManifest(
        stage_goal=stage_goal,
        file_path=file_path,
        file_type=file_type,
        items=_items_from_raw(raw_items),
    )


def replan(
    llm_client: Any,
    manifest: SubtaskManifest,
    gap_description: str,
) -> list[SubtaskItem]:
    """Append new items to manifest when reflect gate says 'deepen'."""
    completed = "\n".join(f"  ✓ {it.title}" for it in manifest.items if it.status == "complete")
    prompt = (
        f"Stage goal: {manifest.stage_goal}\n"
        f"File: {manifest.file_path} (type: {manifest.file_type})\n"
        f"Already complete:\n{completed}\n\n"
        f"Gap identified: {gap_description}"
    )

    from birdclaw.llm.types import Message
    messages = [
        Message(role="system", content=_REPLAN_SYSTEM),
        Message(role="user", content=prompt),
    ]

    result = llm_client.generate(messages, format_schema=SUBTASK_PLAN_SCHEMA, thinking=True)
    raw_items = _parse_response(result.content or "")
    new_items = _items_from_raw(raw_items, start_index=len(manifest.items))
    manifest.items.extend(new_items)
    return new_items
