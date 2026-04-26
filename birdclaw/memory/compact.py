"""Context compaction — keeps the agent loop from growing without bound.

Full port of claw-code-parity/rust/crates/runtime/src/compact.rs.

When the conversation history exceeds a token threshold, older messages are
summarised into a structured system block and removed. The last N messages
are always preserved verbatim. Successive compactions stack cleanly.

Summary structure (mirrors Rust):
    - Scope: N messages compacted (user=X, assistant=Y, tool=Z)
    - Tools mentioned: bash, read_file, ...
    - Recent user requests: last 3 user messages
    - Pending work: messages containing todo/next/pending/follow up/remaining
    - Key files referenced: file paths extracted from messages
    - Current work: last non-empty text from any role
    - Key timeline: all messages as one-liner summaries

Special preamble on resume:
    "This session is being continued from a previous conversation that ran
    out of context. The summary below covers the earlier portion..."
    + "Continue the conversation from where it left off without asking the
    user any further questions. Resume directly — do not acknowledge the
    summary, do not recap what was happening..."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from birdclaw.llm.types import Message

# ---------------------------------------------------------------------------
# Preamble strings (match Rust constants exactly)
# ---------------------------------------------------------------------------

_CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation that ran "
    "out of context. The summary below covers the earlier portion of the conversation.\n\n"
)
_RECENT_MESSAGES_NOTE = "Recent messages are preserved verbatim."
_DIRECT_RESUME_INSTRUCTION = (
    "Continue the conversation from where it left off without asking the user "
    "any further questions. Resume directly — do not acknowledge the summary, "
    "do not recap what was happening, and do not preface with continuation text."
)

# File extensions worth tracking in summaries
_INTERESTING_EXTENSIONS = frozenset({
    "py", "rs", "ts", "tsx", "js", "json", "md", "toml", "yaml", "yml",
    "sh", "txt", "cfg", "ini",
})

# Keywords that imply pending work
_PENDING_KEYWORDS = frozenset({"todo", "next", "pending", "follow up", "remaining"})

# ---------------------------------------------------------------------------
# Config + result
# ---------------------------------------------------------------------------

@dataclass
class CompactionConfig:
    preserve_recent_messages: int = 4
    max_estimated_tokens: int = 2_000
    # format stages (write_doc/write_code) need more recent messages preserved
    # so the model sees the last section it wrote; caller may override this
    preserve_recent_messages_format: int = 8


@dataclass
class CompactionResult:
    summary: str
    formatted_summary: str
    compacted_messages: "list[Message]"
    removed_message_count: int


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: chars / 4 + 1. No tokenizer needed."""
    return len(text) // 4 + 1


def estimate_messages_tokens(messages: "list[Message]") -> int:
    return sum(estimate_tokens(m.content or "") for m in messages)


# ---------------------------------------------------------------------------
# Compaction decision
# ---------------------------------------------------------------------------

def _compacted_prefix_len(messages: "list[Message]") -> int:
    """Return 1 if the first message is an existing compacted summary, else 0."""
    if not messages:
        return 0
    first = messages[0]
    if first.role == "system" and first.content.startswith(_CONTINUATION_PREAMBLE):
        return 1
    return 0


def should_compact(
    messages: "list[Message]",
    config: CompactionConfig | None = None,
    in_format_stage: bool = False,
) -> bool:
    """Return True if the compactable portion exceeds the token threshold."""
    cfg = config or CompactionConfig()
    preserve = cfg.preserve_recent_messages_format if in_format_stage else cfg.preserve_recent_messages
    start = _compacted_prefix_len(messages)
    compactable = messages[start:]
    return (
        len(compactable) > preserve
        and estimate_messages_tokens(compactable) >= cfg.max_estimated_tokens
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _extract_file_candidates(content: str) -> list[str]:
    candidates = []
    for token in content.split():
        token = token.strip(",.;:)(\"\\'`")
        if "/" in token:
            ext = token.rsplit(".", 1)[-1].lower() if "." in token else ""
            if ext in _INTERESTING_EXTENSIONS:
                candidates.append(token)
    return candidates


def _collapse_blank_lines(text: str) -> str:
    result, last_blank = [], False
    for line in text.splitlines():
        is_blank = not line.strip()
        if is_blank and last_blank:
            continue
        result.append(line)
        last_blank = is_blank
    return "\n".join(result)


def _strip_tag_block(content: str, tag: str) -> str:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    s = content.find(start_tag)
    e = content.find(end_tag)
    if s != -1 and e != -1:
        return content[:s] + content[e + len(end_tag):]
    return content


def _extract_tag_block(content: str, tag: str) -> str | None:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    s = content.find(start_tag)
    e = content.find(end_tag)
    if s != -1 and e != -1:
        return content[s + len(start_tag):e]
    return None


# ---------------------------------------------------------------------------
# Summary generation (no LLM — pure heuristic, matches Rust)
# ---------------------------------------------------------------------------

def _summarise_messages(messages: "list[Message]") -> str:
    user_count = sum(1 for m in messages if m.role == "user")
    assistant_count = sum(1 for m in messages if m.role == "assistant")
    tool_count = sum(1 for m in messages if m.role == "tool")

    tool_names = sorted({
        m.name for m in messages if m.role == "tool" and m.name
    })

    lines = [
        "<summary>",
        "Conversation summary:",
        f"- Scope: {len(messages)} earlier messages compacted "
        f"(user={user_count}, assistant={assistant_count}, tool={tool_count}).",
    ]

    if tool_names:
        lines.append(f"- Tools mentioned: {', '.join(tool_names)}.")

    # Recent user requests (last 3, oldest first)
    user_msgs = [m.content for m in messages if m.role == "user" and m.content.strip()]
    if user_msgs:
        lines.append("- Recent user requests:")
        for req in user_msgs[-3:]:
            lines.append(f"  - {_truncate(req)}")

    # Pending work — messages mentioning todo/next/pending/remaining
    pending = []
    for m in reversed(messages):
        text = m.content or ""
        if any(kw in text.lower() for kw in _PENDING_KEYWORDS):
            pending.append(_truncate(text))
        if len(pending) >= 3:
            break
    if pending:
        lines.append("- Pending work:")
        for item in reversed(pending):
            lines.append(f"  - {item}")

    # Key files
    all_files: list[str] = []
    for m in messages:
        all_files.extend(_extract_file_candidates(m.content or ""))
    key_files = sorted(set(all_files))[:8]
    if key_files:
        lines.append(f"- Key files referenced: {', '.join(key_files)}.")

    # Current work — last non-empty content from any message
    current = next(
        (m.content.strip() for m in reversed(messages) if m.content.strip()),
        None
    )
    if current:
        lines.append(f"- Current work: {_truncate(current, 200)}")

    # Key timeline
    lines.append("- Key timeline:")
    for m in messages:
        content_summary = _truncate(m.content or "", 160)
        lines.append(f"  - {m.role}: {content_summary}")

    lines.append("</summary>")
    return "\n".join(lines)


def format_compact_summary(summary: str) -> str:
    """Strip <analysis> tag, reformat <summary> tag into readable text."""
    without_analysis = _strip_tag_block(summary, "analysis")
    inner = _extract_tag_block(without_analysis, "summary")
    if inner:
        formatted = without_analysis.replace(
            f"<summary>{inner}</summary>",
            f"Summary:\n{inner.strip()}",
        )
    else:
        formatted = without_analysis
    return _collapse_blank_lines(formatted).strip()


def _extract_summary_highlights(summary: str) -> list[str]:
    """Extract bullet lines, excluding the key timeline section."""
    lines, in_timeline = [], False
    for line in format_compact_summary(summary).splitlines():
        t = line.rstrip()
        if not t or t in ("Summary:", "Conversation summary:"):
            continue
        if t == "- Key timeline:":
            in_timeline = True
            continue
        if not in_timeline:
            lines.append(t)
    return lines


def _extract_summary_timeline(summary: str) -> list[str]:
    lines, in_timeline = [], False
    for line in format_compact_summary(summary).splitlines():
        t = line.rstrip()
        if t == "- Key timeline:":
            in_timeline = True
            continue
        if not in_timeline:
            continue
        if not t:
            break
        lines.append(t)
    return lines


def _merge_summaries(existing: str | None, new_summary: str) -> str:
    """Stack existing and new summaries when compacting a second time."""
    if not existing:
        return new_summary

    prev_highlights = _extract_summary_highlights(existing)
    new_fmt = format_compact_summary(new_summary)
    new_highlights = _extract_summary_highlights(new_fmt)
    new_timeline = _extract_summary_timeline(new_fmt)

    lines = ["<summary>", "Conversation summary:"]

    if prev_highlights:
        lines.append("- Previously compacted context:")
        lines.extend(f"  {l}" for l in prev_highlights)

    if new_highlights:
        lines.append("- Newly compacted context:")
        lines.extend(f"  {l}" for l in new_highlights)

    if new_timeline:
        lines.append("- Key timeline:")
        lines.extend(f"  {l}" for l in new_timeline)

    lines.append("</summary>")
    return "\n".join(lines)


def get_continuation_message(
    summary: str,
    suppress_follow_up: bool = True,
    recent_preserved: bool = True,
) -> str:
    base = _CONTINUATION_PREAMBLE + format_compact_summary(summary)
    if recent_preserved:
        base += f"\n\n{_RECENT_MESSAGES_NOTE}"
    if suppress_follow_up:
        base += f"\n{_DIRECT_RESUME_INSTRUCTION}"
    return base


def _extract_existing_summary(messages: "list[Message]") -> str | None:
    """Extract prior compacted summary text from the first system message."""
    if not messages or messages[0].role != "system":
        return None
    text = messages[0].content or ""
    if not text.startswith(_CONTINUATION_PREAMBLE):
        return None
    summary = text[len(_CONTINUATION_PREAMBLE):]
    # Strip trailing notes
    for suffix in (f"\n\n{_RECENT_MESSAGES_NOTE}", f"\n{_DIRECT_RESUME_INSTRUCTION}"):
        if suffix in summary:
            summary = summary[:summary.index(suffix)]
    return summary.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _goal_score(message: "Message", goal_tokens: set[str]) -> int:
    """Score a message by keyword overlap with the current stage goal.

    ACON insight: messages that share vocabulary with the upcoming goal
    are more likely to contain facts the model will need. Preserving them
    instead of recency-only keeps the most *relevant* context alive.
    """
    if not goal_tokens:
        return 0
    text = (message.content or "").lower()
    msg_tokens = {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}
    return len(goal_tokens & msg_tokens)


def compact(
    messages: "list[Message]",
    config: CompactionConfig | None = None,
    in_format_stage: bool = False,
    current_goal: str = "",
) -> CompactionResult:
    """Compact the message list if it exceeds the token threshold.

    Returns a CompactionResult. If compaction was not needed, returns the
    original messages unchanged with removed_message_count=0.
    """
    from birdclaw.llm.types import Message as Msg

    cfg = config or CompactionConfig()

    if not should_compact(messages, cfg, in_format_stage=in_format_stage):
        return CompactionResult(
            summary="",
            formatted_summary="",
            compacted_messages=messages,
            removed_message_count=0,
        )

    preserve = cfg.preserve_recent_messages_format if in_format_stage else cfg.preserve_recent_messages
    existing_summary = _extract_existing_summary(messages)
    prefix_len = 1 if existing_summary else 0
    keep_from = max(prefix_len, len(messages) - preserve)
    candidates = messages[prefix_len:keep_from]
    tail = messages[keep_from:]

    # ACON-style: if we know the next goal, score candidates by relevance and
    # rescue the top-scoring ones into the preserved tail (up to preserve // 2).
    # This keeps goal-relevant facts alive instead of purely recency-based keeps.
    rescued: list[Message] = []
    if current_goal and candidates:
        goal_tokens = {w for w in re.findall(r"[a-z0-9]+", current_goal.lower()) if len(w) > 2}
        scored = sorted(
            enumerate(candidates),
            key=lambda t: _goal_score(t[1], goal_tokens),
            reverse=True,
        )
        rescue_budget = max(1, preserve // 2)
        rescued_indices: set[int] = set()
        for idx, msg in scored[:rescue_budget]:
            if _goal_score(msg, goal_tokens) > 0:
                rescued_indices.add(idx)
        # Maintain original order for rescued messages
        rescued = [m for i, m in enumerate(candidates) if i in rescued_indices]
        candidates = [m for i, m in enumerate(candidates) if i not in rescued_indices]

    removed = candidates
    preserved = rescued + list(tail)

    new_raw_summary = _summarise_messages(removed)
    merged_summary = _merge_summaries(existing_summary, new_raw_summary)
    formatted = format_compact_summary(merged_summary)
    continuation = get_continuation_message(
        merged_summary,
        suppress_follow_up=True,
        recent_preserved=bool(preserved),
    )

    compacted = [Msg(role="system", content=continuation)] + list(preserved)

    logger.info(
        "[compact] removed=%d  rescued=%d  preserved=%d  total_after=%d  goal=%r",
        len(removed), len(rescued), len(preserved), len(compacted), current_goal[:40],
    )
    return CompactionResult(
        summary=merged_summary,
        formatted_summary=formatted,
        compacted_messages=compacted,
        removed_message_count=len(removed),
    )
