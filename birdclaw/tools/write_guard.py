"""Write guard — programmatic pre/post-write checks for file output stages.

Catches the most common small-model output failures before they reach disk:
  1. Leaked JSON envelope  — model outputs {"tool_name":..,"params":..} instead of content
  2. Empty content         — nothing to write
  3. Syntax error (.py)    — py_compile before accepting code
  4. Kind tracking         — returns "create" vs "update" for every write

None of these checks require an LLM call.
"""

from __future__ import annotations

import json
import logging
import py_compile
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# JSON patterns that indicate a model leaked its tool-call envelope into content
_TOOL_ENVELOPE_KEYS = {"tool_name", "tool_call", "function", "action", "params", "arguments"}
_JSON_START = re.compile(r"^\s*[{\[]")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PreWriteResult:
    ok: bool
    content: str          # possibly fixed/normalized
    error: str = ""       # human-readable reason if ok=False


@dataclass
class PostWriteResult:
    kind: Literal["create", "update"]
    bytes_written: int
    checks: list[str] = field(default_factory=list)   # passed checks
    errors: list[str] = field(default_factory=list)   # failed checks


# ---------------------------------------------------------------------------
# Leaked JSON detection + normalization
# ---------------------------------------------------------------------------

def _looks_like_json_envelope(content: str) -> bool:
    """True if content appears to be a raw tool-call JSON object, not actual content."""
    if not _JSON_START.match(content):
        return False
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        # Try to parse just the first {...} block
        m = re.search(r"\{.*\}", content[:2000], re.DOTALL)
        if not m:
            return False
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return False
    if not isinstance(obj, dict):
        return False
    return bool(_TOOL_ENVELOPE_KEYS & set(obj.keys()))


def _extract_from_envelope(content: str, fallback_path: str = "") -> tuple[str, str]:
    """Try to extract (actual_content, path) from a tool-call envelope.

    Handles schemas:
      {"tool_name": "write_doc", "params": {"file_path": "...", "content": "..."}}
      {"path": "...", "content": "..."}
      {"content": "..."}
    Returns ("", "") if extraction fails.
    """
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return "", fallback_path
        try:
            obj = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return "", fallback_path

    if not isinstance(obj, dict):
        return "", fallback_path

    # Schema: {"tool_name": ..., "params": {"file_path": ..., "content": ...}}
    params = obj.get("params") or obj.get("arguments") or obj.get("args") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, ValueError):
            params = {}

    actual = (
        params.get("content")
        or obj.get("content")
        or params.get("text")
        or obj.get("text")
        or ""
    )
    path = (
        params.get("file_path")
        or params.get("path")
        or obj.get("file_path")
        or obj.get("path")
        or fallback_path
    )
    return str(actual), str(path)


# ---------------------------------------------------------------------------
# Syntax check
# ---------------------------------------------------------------------------

def _check_python_syntax(content: str) -> str | None:
    """Return error message if content has a Python syntax error, else None."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8",
                                    delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        py_compile.compile(tmp_path, doraise=True)
        return None
    except py_compile.PyCompileError as exc:
        return str(exc)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pre_write_check(
    path: str,
    content: str,
    file_type: str = "",          # "code" | "doc" | "" (inferred from extension)
    existed_before: bool = False,
) -> PreWriteResult:
    """Run all programmatic checks before writing content to disk.

    Returns PreWriteResult.ok=False with an error message if the write should
    be rejected so the caller can retry with a corrected prompt.
    Returns PreWriteResult with (possibly fixed) content if all checks pass.
    """
    # 1. Empty content
    if not content or not content.strip():
        return PreWriteResult(ok=False, content=content, error="empty content — nothing to write")

    # 2. Leaked JSON envelope — try to normalize before rejecting
    if _looks_like_json_envelope(content):
        extracted, extracted_path = _extract_from_envelope(content, path)
        if extracted and extracted.strip():
            logger.warning(
                "write_guard: leaked JSON envelope detected — extracted %d chars of actual content",
                len(extracted),
            )
            content = extracted
            if extracted_path and not path:
                path = extracted_path
        else:
            return PreWriteResult(
                ok=False,
                content=content,
                error=(
                    "content appears to be a raw JSON tool call, not actual file content. "
                    "Output the file content directly — not wrapped in JSON."
                ),
            )

    # 3. Empty after normalization
    if not content.strip():
        return PreWriteResult(ok=False, content=content, error="content is empty after JSON extraction")

    # 4. Python syntax check — file_type="code" takes priority; falls back to ext.
    ext = Path(path).suffix.lower() if path else ""
    if file_type:
        is_code = file_type == "code"
    else:
        is_code = ext == ".py"
    if is_code:
        err = _check_python_syntax(content)
        if err:
            return PreWriteResult(
                ok=False,
                content=content,
                error=f"Python syntax error — fix before writing:\n{err}",
            )

    return PreWriteResult(ok=True, content=content)


def post_write_check(path: str, existed_before: bool) -> PostWriteResult:
    """Verify the file was successfully written and return kind + metadata."""
    p = Path(path)
    checks: list[str] = []
    errors: list[str] = []
    kind: Literal["create", "update"] = "update" if existed_before else "create"

    if not p.exists():
        errors.append(f"file not found after write: {path}")
        return PostWriteResult(kind=kind, bytes_written=0, errors=errors)

    checks.append("exists")
    size = p.stat().st_size
    if size == 0:
        errors.append("file is empty after write")
    else:
        checks.append(f"non-empty ({size} bytes)")

    return PostWriteResult(kind=kind, bytes_written=size, checks=checks, errors=errors)
