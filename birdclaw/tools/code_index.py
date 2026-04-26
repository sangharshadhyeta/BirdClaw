"""Code symbol index — AST-based lookup for Python files.

Builds a lightweight index mapping symbol names to file locations.
Paths are stored relative to each scan root, so the index is portable
regardless of where the project is installed.

Usage:
    rebuild_index()                     # scan project root (cwd)
    find_symbol("check_file_write")     # → JSON with relative path + line range
    find_symbol("PermissionEnforcer")   # → class location
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

from birdclaw.tools.registry import Tool, registry

# Scan root anchored to this file's location — never depends on cwd.
# birdclaw/tools/code_index.py → birdclaw/tools/ → birdclaw/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Index structure:
#   {symbol_name: [{file, start_line, end_line, kind, parent}]}
# Paths are relative to the root they were found in.
# A name may appear in multiple files, hence a list per name.
# ---------------------------------------------------------------------------

_index: dict[str, list[dict]] = {}

_SKIP_DIRS = frozenset({
    ".venv", "venv", ".env",
    "__pycache__", ".git",
    "node_modules", "target",
    "dist", "build", ".mypy_cache", ".ruff_cache",
    "repo",  # reference projects — skip, we only index live code
})


def _should_skip(path: Path) -> bool:
    return bool(set(path.parts) & _SKIP_DIRS)


def _index_file(path: Path, root: Path) -> None:
    """Parse one Python file and add its symbols to _index.

    Paths stored as relative to `root`.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError):
        return

    rel = str(path.relative_to(root))

    # Build a parent-class lookup: function node → class name (if method)
    class_children: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                    class_children[id(child)] = node.name

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)):
            continue

        end_line = getattr(node, "end_lineno", node.lineno)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry = {
                "file": rel,
                "start_line": node.lineno,
                "end_line": end_line,
                "kind": "function",
                "parent": class_children.get(id(node)),
            }
            _index.setdefault(node.name, []).append(entry)

        elif isinstance(node, ast.ClassDef):
            entry = {
                "file": rel,
                "start_line": node.lineno,
                "end_line": end_line,
                "kind": "class",
                "parent": None,
            }
            _index.setdefault(node.name, []).append(entry)

        elif isinstance(node, ast.Assign) and node.col_offset == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    entry = {
                        "file": rel,
                        "start_line": node.lineno,
                        "end_line": node.lineno,
                        "kind": "assignment",
                        "parent": None,
                    }
                    _index.setdefault(target.id, []).append(entry)


def rebuild_index(roots: list[Path] | None = None) -> int:
    """Scan directories and rebuild the symbol index.

    Default: the project root derived from this file's location — never
    depends on cwd, so the scan is always bounded regardless of where
    birdclaw is launched from.

    Args:
        roots: Directories to scan. Defaults to [_PROJECT_ROOT].

    Returns:
        Number of unique symbol names indexed.
    """
    global _index
    _index = {}

    scan_roots = roots if roots is not None else [_PROJECT_ROOT]

    for root in scan_roots:
        if not root.exists():
            continue
        try:
            root_dev = os.stat(root).st_dev
        except OSError:
            continue
        for py_file in root.rglob("*.py"):
            if _should_skip(py_file):
                continue
            # Skip files on a different filesystem (container overlays, mounts)
            try:
                if os.stat(py_file).st_dev != root_dev:
                    continue
            except OSError:
                continue
            _index_file(py_file, root)

    return len(_index)


def find_symbol(name: str, kind: str | None = None) -> str:
    """Find a Python symbol by name and return its file location.

    Args:
        name: Exact symbol name (function, class, or module-level constant).
        kind: Optional — "function", "class", or "assignment".

    Returns:
        JSON {"matches": [{file, start_line, end_line, kind, parent}, ...]}
        or   {"error": "...", "did_you_mean": [...]}
    """
    if not _index:
        rebuild_index()

    matches = _index.get(name, [])
    if kind:
        matches = [m for m in matches if m["kind"] == kind]

    if not matches:
        lower = name.lower()
        hints = [k for k in _index if lower in k.lower()][:5]
        msg: dict = {"error": f"symbol {name!r} not found"}
        if hints:
            msg["did_you_mean"] = hints
        return json.dumps(msg)

    return json.dumps({"matches": matches})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="find_symbol",
    description=(
        "Find a Python function, class, or constant by name. "
        "Returns the relative file path and exact line range. "
        "Use before read_file when you know what to look for but not where it lives."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": ["function", "class", "assignment"],
            },
        },
        "required": ["name"],
    },
    handler=find_symbol,
    tags=["find", "symbol", "function", "class", "definition", "locate", "where", "code", "search"],
))

# Build at import time — scans cwd, fast for a single project
rebuild_index()
