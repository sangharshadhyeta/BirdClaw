"""Progressive memory via line search.

Instead of injecting whole files or blind tails, each LLM call searches for
the exact lines relevant to its current goal. The file is the memory; search
is the retrieval mechanism.

Two entry points:

  search_lines(pattern, paths)
      Exact pattern search — regex or plain string.
      Use when you know what you're looking for: a function name, a section
      heading, a specific variable. Returns matching lines with ±context lines.

  search_relevant(goal, paths)
      Goal-driven search — extracts key terms from the goal and finds lines
      that overlap with them. Use when building context for a write/verify call:
      "which notes do I need to check to do this task?"

Both return a formatted string ready for LLM injection, capped to keep
context small. Empty string means nothing relevant found — caller falls back
to last-N-lines or no injection.

Tool registration (search_notes) is at the bottom — imported by tools/__init__.py.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_MAX_INJECT_CHARS = 1200   # hard cap on total injected text
_MIN_TERM_LEN     = 3      # ignore short words when extracting terms
_STOP_WORDS       = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into",
    "write", "create", "make", "add", "get", "set", "use", "run",
    "build", "file", "code", "function", "class", "method", "data",
    "next", "new", "all", "each", "any", "not", "its", "are", "was",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _Match:
    path:     Path
    line_no:  int          # 1-based
    line:     str
    context:  list[str] = field(default_factory=list)   # surrounding lines
    score:    int = 0      # relevance score for ranking


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _extract_terms(goal: str) -> list[str]:
    """Extract meaningful search terms from a goal string."""
    # Pull out snake_case identifiers, camelCase, file paths, and plain words
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[a-z]+", goal.lower())
    terms = []
    seen: set[str] = set()
    for t in tokens:
        if (
            len(t) >= _MIN_TERM_LEN
            and t not in _STOP_WORDS
            and t not in seen
        ):
            seen.add(t)
            terms.append(t)
    # Also split camelCase / snake_case fragments
    extras = []
    for t in terms:
        parts = re.findall(r"[a-z]+", re.sub(r"([A-Z])", r"_\1", t).lower())
        for p in parts:
            if len(p) >= _MIN_TERM_LEN and p not in _STOP_WORDS and p not in seen:
                seen.add(p)
                extras.append(p)
    return terms + extras


def _gather_matches(
    paths: Sequence[str | Path],
    predicate,          # (line: str) -> bool
    context_lines: int,
    max_results: int,
) -> list[_Match]:
    matches: list[_Match] = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.is_file():
            continue
        lines = _read_lines(p)
        for i, line in enumerate(lines):
            if predicate(line):
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines + 1)
                ctx = lines[lo:i] + lines[i + 1:hi]
                matches.append(_Match(path=p, line_no=i + 1, line=line, context=ctx))
                if len(matches) >= max_results:
                    return matches
    return matches


def _format_matches(matches: list[_Match], cap: int = _MAX_INJECT_CHARS) -> str:
    if not matches:
        return ""
    parts: list[str] = []
    total = 0
    for m in matches:
        block = f"[{m.path.name}:{m.line_no}] {m.line}"
        for c in m.context:
            block += f"\n  {c}"
        total += len(block)
        if total > cap:
            break
        parts.append(block)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_lines(
    pattern:       str,
    paths:         Sequence[str | Path],
    context_lines: int = 3,
    max_results:   int = 8,
    use_regex:     bool = False,
) -> str:
    """Search files for an exact pattern. Returns formatted lines for LLM injection.

    Use for known lookups: function names, section headings, variable names.
    Returns empty string if nothing found.
    """
    if not pattern or not paths:
        return ""

    if use_regex:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            predicate = lambda line: bool(compiled.search(line))
        except re.error:
            predicate = lambda line: pattern.lower() in line.lower()
    else:
        needle = pattern.lower()
        predicate = lambda line: needle in line.lower()

    matches = _gather_matches(paths, predicate, context_lines, max_results)
    logger.debug("[search] exact pattern=%r  paths=%d  hits=%d", pattern[:40], len(list(paths)), len(matches))
    return _format_matches(matches)


def search_relevant(
    goal:          str,
    paths:         Sequence[str | Path],
    context_lines: int = 2,
    max_results:   int = 6,
) -> str:
    """Goal-driven search: which lines in these files are relevant to this goal?

    Extracts key terms from the goal, scores each line by term overlap, and
    returns the highest-scoring matches. Answers "what do I already know about
    this task?" before the model writes its next piece.

    Returns empty string if nothing relevant found.
    """
    if not goal or not paths:
        return ""

    terms = _extract_terms(goal)
    if not terms:
        return ""

    term_set = set(terms)

    def _score(line: str) -> int:
        words = set(re.findall(r"[a-z0-9_]+", line.lower()))
        return len(term_set & words)

    # Gather all lines with any term overlap, scored
    candidates: list[_Match] = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.is_file():
            continue
        lines = _read_lines(p)
        for i, line in enumerate(lines):
            s = _score(line)
            if s > 0:
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines + 1)
                ctx = lines[lo:i] + lines[i + 1:hi]
                candidates.append(_Match(path=p, line_no=i + 1, line=line, context=ctx, score=s))

    if not candidates:
        logger.debug("[search] relevant  goal=%r  terms=%d  hits=0", goal[:40], len(terms))
        return ""

    # Return top matches by score, deduplicating adjacent lines
    candidates.sort(key=lambda m: (-m.score, m.path.name, m.line_no))
    seen_lines: set[tuple[Path, int]] = set()
    top: list[_Match] = []
    for m in candidates:
        key = (m.path, m.line_no)
        if key not in seen_lines:
            seen_lines.add(key)
            top.append(m)
        if len(top) >= max_results:
            break

    # Re-sort by file + line number for readable output
    top.sort(key=lambda m: (m.path.name, m.line_no))
    logger.debug(
        "[search] relevant  goal=%r  terms=%d  candidates=%d  top=%d",
        goal[:40], len(terms), len(candidates), len(top),
    )
    return _format_matches(top)


def find_section(path: str | Path, title: str, file_type: str) -> str:
    """Find the section or function most relevant to `title` and return just that block.

    For docs:  finds the ## heading whose words best overlap with title words.
               Returns from that heading to the line before the next heading.
    For code:  finds the def/class whose name best overlaps with title terms.
               Returns from that line to the line before the next top-level def/class.

    Returns empty string if nothing relevant found (score == 0 or file empty).
    Caps the returned block at 60 lines to avoid huge sections.
    """
    p = Path(path)
    lines = _read_lines(p)
    if not lines:
        return ""

    if file_type == "doc" or str(path).endswith(".md"):
        header_re = re.compile(r"^(#{1,3}) (.+)")
        name_group = 2
    elif file_type == "code" or str(path).endswith(".py"):
        header_re = re.compile(r"^(?:async def |def |class )(\w+)")
        name_group = 1
    else:
        return ""

    terms = set(_extract_terms(title))
    if not terms:
        return ""

    # Collect all headers: (line_idx, score, line)
    headers: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = header_re.match(line)
        if m:
            # Split snake_case and strip punctuation so "run_stage" → {"run","stage"}
            name_words = set(re.findall(r"[a-z0-9]+", m.group(name_group).lower()))
            score = len(terms & name_words)
            headers.append((i, score, line))

    if not headers:
        return ""

    best_idx, best_score, best_line = max(headers, key=lambda h: h[1])
    if best_score == 0:
        logger.debug("[search] find_section  title=%r  path=%s  result=no-match", title[:40], Path(path).name)
        return ""

    # Section ends at the next header (or EOF)
    end_idx = len(lines)
    for (i, _, _) in headers:
        if i > best_idx:
            end_idx = i
            break

    start = max(0, best_idx - 1)
    block = lines[start:end_idx]
    if len(block) > 60:
        block = block[:60]
    logger.debug(
        "[search] find_section  title=%r  path=%s  matched=%r  lines=%d  score=%d",
        title[:40], Path(path).name, best_line.strip()[:50], len(block), best_score,
    )
    return "\n".join(block)


def find_continuation_point(path: str | Path, file_type: str) -> str:
    """Return from the last section/function header to EOF — the natural re-entry point.

    Used as a fallback when find_section finds nothing relevant.
    For docs (md):  last ## or ### heading → end
    For code (py):  last def / class / async def → end
    Fallback:       last 30 lines

    Empty string if file doesn't exist or is empty.
    """
    p = Path(path)
    lines = _read_lines(p)
    if not lines:
        return ""

    if file_type == "doc" or str(path).endswith(".md"):
        header_re = re.compile(r"^#{1,3} ")
    elif file_type == "code" or str(path).endswith(".py"):
        header_re = re.compile(r"^(async def |def |class )")
    else:
        header_re = None

    if header_re:
        last_idx = -1
        for i, line in enumerate(lines):
            if header_re.match(line):
                last_idx = i
        if last_idx >= 0:
            start = max(0, last_idx - 2)
            logger.debug("[search] continuation  path=%s  last_header=line%d", Path(path).name, last_idx + 1)
            return "\n".join(lines[start:])

    logger.debug("[search] continuation  path=%s  fallback=last30", Path(path).name)
    return "\n".join(lines[-30:])


# ---------------------------------------------------------------------------
# Default paths helper — notes + BIRDCLAW.md in cwd
# ---------------------------------------------------------------------------

def _default_search_paths() -> list[Path]:
    """Return candidate files to search when no paths specified.

    Priority order (most recent first):
      1. */notes.md      — task research notes (search results, think reasoning)
      2. */BIRDCLAW.md   — task history logs
      3. *.md            — cwd root markdown (issues.md, README.md, etc.)
    Capped at 20 files sorted by mtime descending.
    """
    import os
    cwd = Path(os.getcwd())
    candidates: list[Path] = []
    for p in cwd.glob("*/notes.md"):      # task research notes
        if p.is_file():
            candidates.append(p)
    for p in cwd.glob("*/BIRDCLAW.md"):   # task history logs
        if p.is_file():
            candidates.append(p)
    for p in cwd.glob("*.md"):            # cwd root markdown
        if p.is_file():
            candidates.append(p)
    candidates = list({p.resolve(): p for p in candidates}.values())  # deduplicate
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[:20]


# ---------------------------------------------------------------------------
# search_notes tool handler
# ---------------------------------------------------------------------------

def search_notes_handler(query: str, paths: list[str] | None = None, use_regex: bool = False) -> str:
    """Search notes/files for lines relevant to a query.

    Goal-driven (default) or pattern-based (use_regex=True).
    Returns formatted matching lines with context, capped at 1200 chars.
    """
    if not query:
        return json.dumps({"error": "query is required"})

    target_paths: list[Path]
    if paths:
        target_paths = [Path(p) for p in paths]
    else:
        target_paths = _default_search_paths()

    if not target_paths:
        return json.dumps({"results": "", "note": "no files found to search"})

    if use_regex:
        result = search_lines(query, target_paths, context_lines=2, max_results=8, use_regex=True)
    else:
        # Try goal-driven first; fall back to exact if empty
        result = search_relevant(query, target_paths, context_lines=2, max_results=6)
        if not result:
            result = search_lines(query, target_paths, context_lines=2, max_results=6)

    if result:
        logger.info("[search] search_notes  query=%r  paths=%d  found=%d chars", query[:40], len(target_paths), len(result))
        return json.dumps({"results": result})
    logger.debug("[search] search_notes  query=%r  paths=%d  found=nothing", query[:40], len(target_paths))
    return json.dumps({"results": "", "note": "nothing relevant found"})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

from birdclaw.tools.registry import Tool, registry  # noqa: E402

registry.register(Tool(
    name="search_notes",
    description=(
        "Search notes and files for lines relevant to a goal or pattern. "
        "Use to answer 'what do I already know about X?' before writing or verifying. "
        "Defaults to searching *.md notes files in the workspace."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Goal phrase or pattern to search for (e.g. 'authentication flow').",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File paths to search. Omit to search workspace notes files.",
            },
            "use_regex": {
                "type": "boolean",
                "description": "Treat query as a regex pattern (default: false = semantic term matching).",
            },
        },
        "required": ["query"],
    },
    handler=search_notes_handler,
    tags=["search", "notes", "find", "lines", "relevant", "context", "memory", "lookup", "recall"],
))
