"""Stage format schemas.

These JSON schemas are passed as response_format to the llama.cpp server for
stages where tool_calls API is unreliable (large content output on small models).

Two generate modes:
  "tool_call" — standard tool_calls API. Short outputs: think(), bash(), answer().
  "format"    — response_format=json_object constrained generation. Large outputs:
                code functions, document sections, task plans.
                Use thinking=False for format stages to avoid empty content.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plan schema — generated once before the stage loop
# ---------------------------------------------------------------------------
# Flat structure (no nested arrays) to keep Ollama's GBNF grammar simple.
# `stages` is a comma-separated string; Python reconstructs the stage list.
# Valid stage types: research, write_code, write_doc, verify, reflect
# Examples: "write_code,verify"  |  "research,write_doc,reflect"  |  ""

_PLAN_FORMAT: dict = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "description": "One sentence: what does success look like?",
        },
        "steps": {
            "type": "string",
            "description": (
                "Pipe-separated plain English steps. Each step is one action. "
                "Example: 'Run df -h in bash | Run du -sh /* in bash | Summarise results'. "
                "Use plain verbs: Run, Search, Write, Read, Summarise. "
                "One step for simple tasks. Three or fewer for most tasks."
            ),
        },
        "budgets": {
            "type": "string",
            "description": (
                "OPTIONAL. Pipe-separated step budgets matching each step — only set when "
                "a step needs more than the type default (research=12, write_doc=10, "
                "write_code=12, verify=8, reflect=5). "
                "Example for a 200-page report: '12 | 60 | 8'. Omit for standard tasks."
            ),
        },
    },
    "required": ["outcome", "steps"],
}

# ---------------------------------------------------------------------------
# Write-loop schemas — used for format-mode content generation stages
# ---------------------------------------------------------------------------

_CODE_FORMAT: dict = {
    "type": "object",
    "properties": {
        "path":    {"type": "string", "description": "actual filename to write, e.g. fetch_url.py or scraper.py"},
        "content": {"type": "string", "description": "Python source code for this function, with imports on the first call"},
        "done":    {"type": "boolean", "description": "set to true only when ALL planned functions are written; false otherwise"},
    },
    "required": ["path", "content", "done"],
}

_DOC_FORMAT: dict = {
    "type": "object",
    "properties": {
        "path":    {"type": "string", "description": "actual output filename, e.g. proposal.md or report.md"},
        "section": {"type": "string", "description": "section heading text"},
        "content": {"type": "string", "description": "full section body — write multiple paragraphs with detail"},
        "done":    {"type": "boolean", "description": "set to true only when ALL planned sections are written; false otherwise"},
    },
    "required": ["path", "content", "done"],
}
