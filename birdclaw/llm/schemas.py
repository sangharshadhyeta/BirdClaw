"""Named JSON schemas for grammar-constrained decoding via llama.cpp.

Each schema is used in `response_format: {type: json_schema, json_schema: {strict: true, schema: ...}}`
so the server converts it to GBNF at generation time — invalid tokens are masked before
they are produced, not caught after. This eliminates JSON parse failures entirely for
these call sites.

Usage:
    from birdclaw.llm.schemas import PLAN_SCHEMA, EDIT_PATCH_SCHEMA, ...
    result = llm_client.generate(..., format_schema=PLAN_SCHEMA)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plan generation — generate_plan() in planner.py
# ---------------------------------------------------------------------------

PLAN_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "outcome":  {"type": "string"},
                "steps":    {"type": "string"},
                "budgets":  {"type": "string"},
            },
            "required": ["outcome", "steps"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Edit file patch — edit_file stage in loop.py
# ---------------------------------------------------------------------------

EDIT_PATCH_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "edit_patch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["old", "new"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Post-stage reflection gate — reflect_on_stage() in planner.py
# ---------------------------------------------------------------------------

REFLECT_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "reflect",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["continue", "deepen", "insert", "done"],
                },
                "goal": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["edit_file", "research", "write_code", "write_doc"],
                },
            },
            "required": ["decision"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Subtask manifest planning — subtask_planner.plan()
# ---------------------------------------------------------------------------

SUBTASK_PLAN_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtask_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title":     {"type": "string"},
                            "anchor":    {"type": "string"},
                            "kind":      {"type": "string", "enum": ["section", "function", "class", "test"]},
                            "min_chars": {"type": "integer"},
                        },
                        "required": ["title", "anchor", "kind", "min_chars"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["subtasks"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Subtask write item — subtask_executor._write_item()
# ---------------------------------------------------------------------------

WRITE_ITEM_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "write_item",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
                "section": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Soul routing — soul_loop.py 270M flat routing decision
# ---------------------------------------------------------------------------
# Flat schema: no nested objects, no arrays — works reliably with 270M models
# that cannot use OpenAI tool_calls format. Grammar constraint forces valid JSON.

SOUL_ROUTING_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "soul_routing",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["answer", "run_command", "create_task", "stop_task", "escalate", "remember_self"],
                },
                "text": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["action", "text", "note"],
            "additionalProperties": False,
        },
    },
}
