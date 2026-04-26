"""File operation tools — Python port of claw-code-parity/rust/crates/runtime/src/file_ops.rs.

Tools registered:
  read_file   — read with offset/limit
  write_file  — create or overwrite (workspace boundary enforced)
  edit_file   — exact string replacement with optional replace_all
  glob_search — recursive glob sorted by mtime, capped at 100
  grep_search — ripgrep-style search with output_mode, context lines, head_limit, offset
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from birdclaw.config import settings
from birdclaw.tools.permission import enforcer
from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match Rust)
# ---------------------------------------------------------------------------

_MAX_SIZE = 10 * 1024 * 1024       # 10 MB
_BINARY_CHECK_BYTES = 8 * 1024     # first 8 KB checked for NUL
_GLOB_RESULT_CAP = 100


# ---------------------------------------------------------------------------
# Path normalisation (two variants matching Rust)
# ---------------------------------------------------------------------------

def _normalize_path(path: Path) -> Path:
    """Resolve path; requires it to exist (raises FileNotFoundError otherwise)."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    return resolved


def _normalize_path_allow_missing(path: Path) -> Path:
    """Resolve path even if it does not yet exist (for new files)."""
    # Walk up to find the longest existing prefix, then resolve + reattach suffix
    parts = path.parts
    for i in range(len(parts), 0, -1):
        candidate = Path(*parts[:i])
        if candidate.exists():
            suffix = Path(*parts[i:]) if i < len(parts) else Path()
            return (candidate.resolve() / suffix)
    return path.absolute()


# ---------------------------------------------------------------------------
# Workspace boundary check
# ---------------------------------------------------------------------------

def _validate_workspace(resolved: Path, write: bool = False) -> None:
    """Raise PermissionError if resolved path is outside all readable roots.

    7a (parity port): explicitly follows symlinks before checking the boundary.
    A symlink inside the workspace that points outside (e.g. workspace/link →
    /etc/passwd) would pass a string-prefix check but fails here because we
    resolve the actual target first.

    Read roots = workspace_roots + src_dir (birdclaw source — always readable).
    Write roots = workspace_roots + data_dir + src_dir only when self_modify=True.
    """
    # Follow symlinks — the target may escape the workspace even if the link path doesn't
    try:
        if resolved.is_symlink():
            resolved = resolved.resolve()
    except OSError:
        pass

    # src_dir is always readable so the agent can introspect its own capabilities
    try:
        src_resolved = settings.src_dir.resolve()
        if resolved == src_resolved or resolved.is_relative_to(src_resolved):
            if write and not settings.self_modify:
                raise PermissionError(
                    f"writes to birdclaw source ({resolved}) require BC_SELF_MODIFY=1"
                )
            return
    except OSError:
        pass

    for root in settings.workspace_roots:
        try:
            root_resolved = root.resolve()
        except OSError:
            root_resolved = root.absolute()
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return

    roots_str = ", ".join(str(r) for r in settings.workspace_roots)
    raise PermissionError(
        f"path {resolved} is outside workspace roots ({roots_str})"
    )


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------

def _is_binary(path: Path) -> bool:
    """Return True if the file looks binary (NUL byte in first 8 KB)."""
    try:
        with path.open("rb") as f:
            chunk = f.read(_BINARY_CHECK_BYTES)
        return b"\x00" in chunk
    except OSError:
        return False


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a file, returning its text content.

    Args:
        path:   Absolute or relative file path.
        offset: Line offset to start reading from (0-based).
        limit:  Maximum number of lines to return.

    Returns:
        JSON string with {"content": "..."} or {"error": "..."}.
    """
    p = Path(path)
    try:
        resolved = _normalize_path(p)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    if _is_binary(resolved):
        return json.dumps({"error": f"binary file: {path}"})

    size = resolved.stat().st_size
    if size > _MAX_SIZE:
        return json.dumps({"error": f"file too large: {size} bytes (max {_MAX_SIZE})"})

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return json.dumps({"error": str(e)})

    lines = text.splitlines(keepends=True)
    if offset:
        lines = lines[offset:]
    if limit is not None:
        lines = lines[:limit]

    return json.dumps({"content": "".join(lines)})


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def _strip_leaked_tool_call(content: str) -> str:
    """Strip trailing tool-call JSON that the model leaked into file content.

    Only strips a trailing object whose keys are exclusively write-schema
    fields (path/content/old/new). Avoids false-positive truncation of
    legitimate content that happens to contain those key names.
    """
    import re as _re, json as _json
    # Match the last top-level {...} block at the very end of the string
    match = _re.search(r'\n\s*(\{[^{}]+\})\s*$', content)
    if not match:
        return content
    try:
        obj = _json.loads(match.group(1))
    except Exception:
        return content
    if isinstance(obj, dict) and set(obj.keys()) <= {"path", "content", "old", "new"}:
        return content[:match.start()].rstrip()
    return content


def write_file(path: str, content: str) -> str:
    """Write (create or overwrite) a file.

    Args:
        path:    Absolute or relative file path.
        content: Text content to write.

    Returns:
        JSON string with {"written": N} or {"error": "..."}.
    """
    p = Path(path)
    resolved = _normalize_path_allow_missing(p)

    result = enforcer.check_file_write(resolved)
    if not result.allowed:
        return json.dumps({"error": str(result)})

    content = _strip_leaked_tool_call(content)

    if len(content.encode("utf-8")) > _MAX_SIZE:
        return json.dumps({"error": f"content too large (max {_MAX_SIZE} bytes)"})

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as e:
        logger.error("[files] write_file error  path=%s  error=%s", resolved, e)
        return json.dumps({"error": str(e)})

    logger.info("[files] write_file  path=%s  chars=%d", resolved, len(content))
    return json.dumps({"written": len(content), "path": str(resolved)})


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file.

    Args:
        path:        File path.
        old_string:  The exact text to find.
        new_string:  The replacement text.
        replace_all: If True, replace every occurrence; otherwise require exactly one match.

    Returns:
        JSON string with {"replaced": N, "patch": "..."} or {"error": "..."}.
    """
    p = Path(path)
    try:
        resolved = _normalize_path(p)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    result = enforcer.check_file_write(resolved)
    if not result.allowed:
        return json.dumps({"error": str(result)})

    try:
        original = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return json.dumps({"error": str(e)})

    count = original.count(old_string)
    if count == 0:
        return json.dumps({"error": "old_string not found in file"})
    if not replace_all and count > 1:
        return json.dumps({
            "error": f"old_string found {count} times; set replace_all=true or use a more specific string"
        })

    if replace_all:
        updated = original.replace(old_string, new_string)
    else:
        updated = original.replace(old_string, new_string, 1)

    try:
        resolved.write_text(updated, encoding="utf-8")
    except OSError as e:
        return json.dumps({"error": str(e)})

    # Simple patch: all-minus then all-plus (matches Rust make_patch)
    patch_lines = (
        [f"--- {path}", f"+++ {path}"]
        + [f"-{line}" for line in old_string.splitlines()]
        + [f"+{line}" for line in new_string.splitlines()]
    )
    patch = "\n".join(patch_lines)

    replaced = count if replace_all else 1
    logger.info("[files] edit_file  path=%s  replaced=%d  old_len=%d→new_len=%d",
                resolved, replaced, len(old_string), len(new_string))
    return json.dumps({"replaced": replaced, "patch": patch})


# ---------------------------------------------------------------------------
# glob_search
# ---------------------------------------------------------------------------

def glob_search(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. "**/*.py").
        path:    Root directory to search in (default: first workspace root).

    Returns:
        JSON string with {"matches": [...]} sorted by mtime descending, max 100 results.
    """
    root = Path(path) if path else settings.workspace_roots[0]
    search_pattern = str(root / pattern)

    try:
        matches = _glob.glob(search_pattern, recursive=True)
    except Exception as e:
        return json.dumps({"error": str(e)})

    # Sort by mtime descending, cap at 100
    try:
        matches.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    except OSError:
        pass

    matches = matches[:_GLOB_RESULT_CAP]
    return json.dumps({"matches": matches})


# ---------------------------------------------------------------------------
# grep_search
# ---------------------------------------------------------------------------

def grep_search(
    pattern: str,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "files_with_matches",
    context_lines: int = 0,
    lines_before: int = 0,
    lines_after: int = 0,
    head_limit: int = 250,
    offset: int = 0,
    case_insensitive: bool = False,
    multiline: bool = False,
) -> str:
    """Search file contents using grep.

    Args:
        pattern:          Regex or literal pattern.
        path:             Directory or file to search (default: first workspace root).
        file_glob:        Glob filter for files (e.g. "*.py").
        output_mode:      "files_with_matches" | "content" | "count".
        context_lines:    Lines of context before and after each match (-C).
        lines_before:     Lines before each match (-B, overrides context_lines).
        lines_after:      Lines after each match (-A, overrides context_lines).
        head_limit:       Maximum output lines/entries (default 250, 0 = unlimited).
        offset:           Skip first N entries.
        case_insensitive: Case-insensitive search.
        multiline:        Allow . to match newlines.

    Returns:
        JSON string with {"results": [...]} or {"error": "..."}.
    """
    search_path = path or str(settings.workspace_roots[0])

    cmd = ["grep", "-rn"]

    if case_insensitive:
        cmd.append("-i")
    if multiline:
        cmd.append("-P")  # PCRE for (?s) multiline
        pattern = "(?s)" + pattern

    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    # content mode: default (shows matching lines with line numbers)

    # Context lines
    before = lines_before or context_lines
    after = lines_after or context_lines
    if before:
        cmd.extend([f"-B{before}"])
    if after:
        cmd.extend([f"-A{after}"])

    if file_glob:
        cmd.extend(["--include", file_glob])

    cmd.extend(["--", pattern, search_path])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "grep timed out after 30s"})
    except FileNotFoundError:
        return json.dumps({"error": "grep not found on PATH"})

    lines = proc.stdout.splitlines()

    # Apply offset and head_limit
    if offset:
        lines = lines[offset:]
    if head_limit:
        lines = lines[:head_limit]

    return json.dumps({"results": lines})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="read_file",
    description="Read the contents of a file. Supports offset and limit for large files.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "offset": {"type": "integer", "description": "Line offset to start from."},
            "limit": {"type": "integer", "description": "Max lines to return."},
        },
        "required": ["path"],
    },
    handler=read_file,
    tags=["read", "file", "open", "view", "cat", "show", "content", "source", "code"],
))

registry.register(Tool(
    name="write_file",
    description="Write (create or overwrite) a file with the given content.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
    tags=["write", "create", "save", "file", "output", "generate", "new"],
))

registry.register(Tool(
    name="edit_file",
    description=(
        "Replace an exact string in a file. Requires a unique match unless replace_all=true. "
        "Returns a patch summary."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to find."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    handler=edit_file,
    tags=["edit", "modify", "change", "update", "replace", "patch", "fix", "refactor"],
))

registry.register(Tool(
    name="glob_search",
    description="Find files matching a glob pattern (e.g. '**/*.py'). Results sorted by modification time.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Root directory (default: workspace root)."},
        },
        "required": ["pattern"],
    },
    handler=glob_search,
    tags=["find", "search", "files", "glob", "pattern", "list", "locate"],
))

registry.register(Tool(
    name="grep_search",
    description=(
        "Search file contents by regex. output_mode: files_with_matches | content | count. "
        "Supports context lines, file glob filter, case-insensitive, head_limit."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex or literal pattern."},
            "path": {"type": "string", "description": "File or directory to search."},
            "file_glob": {"type": "string", "description": "Filter files by glob (e.g. '*.py')."},
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
            },
            "context_lines": {"type": "integer"},
            "lines_before": {"type": "integer"},
            "lines_after": {"type": "integer"},
            "head_limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "case_insensitive": {"type": "boolean"},
            "multiline": {"type": "boolean"},
        },
        "required": ["pattern"],
    },
    handler=grep_search,
    tags=["grep", "search", "find", "text", "pattern", "regex", "content", "code", "string"],
))
