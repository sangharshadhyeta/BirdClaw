"""Project context discovery — Port 10.

Python port of claw-code-parity/rust/crates/runtime/src/prompt.rs
(ProjectContext, instruction file discovery, git snapshot helpers).

Walks ancestor directories collecting CLAUDE.md / CLAUDE.local.md /
.claw/CLAUDE.md / .claw/instructions.md, deduplicates by content hash,
and optionally captures a git status + diff snapshot.

Typical usage in the agent loop:

    from birdclaw.agent.context import ProjectContext
    ctx = ProjectContext.discover_with_git(Path.cwd())
    system_prompt = build_system_prompt(project_context=ctx.render())
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (mirrors prompt.rs)
# ---------------------------------------------------------------------------

MAX_INSTRUCTION_FILE_CHARS = 4_000
MAX_TOTAL_INSTRUCTION_CHARS = 12_000

# BIRDCLAW.md section filtering — only kick in above this size
_BIRDCLAW_FILTER_THRESHOLD = 600   # chars; below this inject in full
_BIRDCLAW_MAX_INJECTED = 800       # chars injected after filtering

_INSTRUCTION_CANDIDATES = [
    "BIRDCLAW.md",
    "BIRDCLAW.local.md",
    ".birdclaw/instructions.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    ".claw/CLAUDE.md",
    ".claw/instructions.md",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ContextFile:
    path: Path
    content: str


@dataclass
class ProjectContext:
    cwd:               Path
    current_date:      str
    git_status:        Optional[str]       = None
    git_diff:          Optional[str]       = None
    instruction_files: list[ContextFile]   = field(default_factory=list)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, cwd: Path, current_date: str = "") -> "ProjectContext":
        if not current_date:
            current_date = date.today().isoformat()
        files = _discover_instruction_files(cwd)
        return cls(cwd=cwd, current_date=current_date, instruction_files=files)

    @classmethod
    def discover_with_git(cls, cwd: Path, current_date: str = "") -> "ProjectContext":
        ctx = cls.discover(cwd, current_date)
        ctx.git_status = _read_git_status(cwd)
        ctx.git_diff   = _read_git_diff(cwd)
        return ctx

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, query: str = "") -> str:
        """Return a markdown block suitable for injection into a system prompt.

        query: when non-empty, BIRDCLAW.md sections are filtered by keyword
        relevance so only the most relevant portion is injected (saves tokens).
        """
        lines = ["# Project context"]
        bullets = [
            f"Today's date is {self.current_date}.",
            f"Working directory: {self.cwd}",
        ]
        if self.instruction_files:
            bullets.append(
                f"Instruction files discovered: {len(self.instruction_files)}."
            )
        lines.extend(f" - {b}" for b in bullets)

        if self.git_status:
            lines += ["", "Git status snapshot:", self.git_status]
        if self.git_diff:
            lines += ["", "Git diff snapshot:", self.git_diff]

        parts = ["\n".join(lines)]
        if self.instruction_files:
            parts.append(_render_instruction_files(self.instruction_files, query=query))
        return "\n\n".join(parts)

    def render_soul(self) -> str:
        """Compact context for the soul layer (~150 tokens max).

        The soul only needs: date, cwd, and git branch.
        It does NOT need CLAUDE.md or git diffs — those are for the agent loop.
        """
        lines = [f"Date: {self.current_date}", f"Dir: {self.cwd}"]
        if self.git_status:
            # Keep only the first line (## branch-name) — the rest is file status noise
            branch_line = self.git_status.splitlines()[0] if self.git_status else ""
            if branch_line:
                lines.append(f"Branch: {branch_line.lstrip('#').strip()}")
        return "  ".join(lines)


# ---------------------------------------------------------------------------
# Instruction file discovery
# ---------------------------------------------------------------------------

def _discover_instruction_files(cwd: Path) -> list[ContextFile]:
    # Build ancestor chain from root → cwd (inclusive)
    ancestors: list[Path] = []
    cursor: Optional[Path] = cwd
    while cursor is not None:
        ancestors.append(cursor)
        parent = cursor.parent
        cursor = parent if parent != cursor else None
    ancestors.reverse()

    files: list[ContextFile] = []
    for directory in ancestors:
        for candidate_rel in _INSTRUCTION_CANDIDATES:
            _push_context_file(files, directory / candidate_rel)

    return _dedupe_instruction_files(files)


def _push_context_file(files: list[ContextFile], path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return
    except OSError:
        return
    if content.strip():
        files.append(ContextFile(path=path, content=content))


def _dedupe_instruction_files(files: list[ContextFile]) -> list[ContextFile]:
    seen: set[str] = set()
    result: list[ContextFile] = []
    for f in files:
        h = _content_hash(_normalize_instruction_content(f.content))
        if h not in seen:
            seen.add(h)
            result.append(f)
    return result


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Instruction file rendering
# ---------------------------------------------------------------------------

def _is_birdclaw_notes(path: Path) -> bool:
    return path.name == "BIRDCLAW.md"


def _tokenise(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _select_birdclaw_sections(content: str, query: str) -> str:
    """Return the most query-relevant sections of BIRDCLAW.md.

    Splits on '## ' headings, scores each section by keyword overlap with the
    query, and returns the top-scoring sections up to _BIRDCLAW_MAX_INJECTED
    chars. Falls back to a simple head-truncation when query is empty.
    """
    if len(content) <= _BIRDCLAW_FILTER_THRESHOLD:
        return content.strip()

    if not query:
        return content.strip()[:_BIRDCLAW_MAX_INJECTED]

    # Split into (heading, body) pairs; keep the file header as section 0
    parts = re.split(r"(?m)^(##\s[^\n]*)", content)
    # parts: [pre-heading-text, heading1, body1, heading2, body2, ...]
    raw_sections: list[tuple[str, str]] = []
    it = iter(parts)
    pre = next(it, "")
    if pre.strip():
        raw_sections.append(("", pre))
    for heading, body in zip(it, it):
        raw_sections.append((heading, body))

    query_tokens = _tokenise(query)
    scored: list[tuple[int, str]] = []
    for heading, body in raw_sections:
        block = heading + body
        score = len(query_tokens & _tokenise(block))
        scored.append((score, block.strip()))

    # Sort highest-score first, then reconstruct up to budget
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: list[str] = []
    budget = _BIRDCLAW_MAX_INJECTED
    for _, block in scored:
        if budget <= 0:
            break
        chosen.append(block[:budget])
        budget -= len(block)

    return "\n\n".join(chosen).strip() or content.strip()[:_BIRDCLAW_MAX_INJECTED]


def _render_instruction_files(files: list[ContextFile], query: str = "") -> str:
    sections = ["# Claude instructions"]
    remaining = MAX_TOTAL_INSTRUCTION_CHARS

    for f in files:
        if remaining == 0:
            sections.append(
                "_Additional instruction content omitted after reaching the prompt budget._"
            )
            break

        if _is_birdclaw_notes(f.path) and query:
            raw = _select_birdclaw_sections(f.content, query)
        else:
            raw = _truncate_instruction_content(f.content, remaining)

        consumed = min(len(raw), remaining)
        remaining -= consumed

        label = f"{_display_context_path(f.path)} (scope: {f.path.parent})"
        sections.append(f"## {label}")
        sections.append(raw)

    return "\n\n".join(sections)


def _truncate_instruction_content(content: str, remaining: int) -> str:
    hard_limit = min(MAX_INSTRUCTION_FILE_CHARS, remaining)
    trimmed = content.strip()
    if len(trimmed) <= hard_limit:
        return trimmed
    return trimmed[:hard_limit] + "\n\n[truncated]"


def _display_context_path(path: Path) -> str:
    return path.name if path.name else str(path)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize_instruction_content(content: str) -> str:
    return collapse_blank_lines(content).strip()


def collapse_blank_lines(content: str) -> str:
    """Collapse runs of blank lines to a single blank line."""
    result: list[str] = []
    prev_blank = False
    for line in content.splitlines():
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result.append(line.rstrip())
        prev_blank = is_blank
    return "\n".join(result) + ("\n" if result else "")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks"] + args,
            cwd=cwd,
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        text = result.stdout.decode("utf-8", errors="replace")
        return text.strip() or None
    except Exception:
        return None


def _read_git_status(cwd: Path) -> Optional[str]:
    return _run_git(["status", "--short", "--branch"], cwd)


def _read_git_diff(cwd: Path) -> Optional[str]:
    sections: list[str] = []

    staged = _run_git(["diff", "--cached"], cwd)
    if staged:
        sections.append(f"Staged changes:\n{staged.rstrip()}")

    unstaged = _run_git(["diff"], cwd)
    if unstaged:
        sections.append(f"Unstaged changes:\n{unstaged.rstrip()}")

    return "\n\n".join(sections) if sections else None
