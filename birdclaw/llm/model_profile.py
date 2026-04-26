"""Model profiles — capability declarations per LLM endpoint.

Instead of scattering model-specific workarounds across loop.py and client.py,
each profile declares what a given model/endpoint can and cannot do reliably.
The agent picks the right profile per call type:

  MAIN  — the 4B reasoning model (llama.cpp, Gemma-4-E4B)
          Always runs with thinking enabled in tool stages.
          Never runs format-mode (hands profile handles that).

  HANDS — a small specialist model (e.g. 270M function-call fine-tune)
          Handles all format-mode calls: planning schema, reflection gate,
          edit_file patches, soul routing.
          Fast, schema-focused, no extended thinking needed.

If no hands model is configured, HANDS falls back to MAIN so existing
single-model deployments continue to work without any config change.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from dataclasses import dataclass

from birdclaw.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_url: str
    model: str
    thinking_in_tool_stages: bool
    thinking_in_format_stages: bool  # always False due to llama.cpp constraint
    tool_choice_required_works: bool  # gemma4: required+thinking = reasoning-only output
    max_tools_per_turn: int


def main_profile() -> ModelProfile:
    """4B reasoning model — think freely, never run format-mode."""
    return ModelProfile(
        name="main",
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        thinking_in_tool_stages=True,
        thinking_in_format_stages=False,
        tool_choice_required_works=False,
        max_tools_per_turn=6,
    )


def hands_profile() -> ModelProfile:
    """Small specialist model — schema-constrained output only.

    Falls back to main profile if no hands model is configured.
    This keeps single-model deployments working with zero config change.
    """
    hands_url   = settings.llm_hands_base_url
    hands_model = settings.llm_hands_model
    if not hands_url or not hands_model:
        logger.debug("[profile] hands falls back to main (no hands model configured)")
        return main_profile()
    return ModelProfile(
        name="hands",
        base_url=hands_url,
        model=hands_model,
        thinking_in_tool_stages=False,
        thinking_in_format_stages=False,
        tool_choice_required_works=True,
        max_tools_per_turn=3,
    )


# ---------------------------------------------------------------------------
# Display name helpers — query llama.cpp for the real loaded model filename
# ---------------------------------------------------------------------------

def _fetch_loaded_model_name(base_url: str, fallback: str) -> str:
    """Query the llama.cpp /v1/models endpoint for the real loaded model filename.

    Returns the first model id (e.g. "gemma-4-E4B-it-Q8_0.gguf") or `fallback`
    if the endpoint is unreachable. Cached for 60 s to avoid per-call HTTP overhead.
    """
    import time as _time

    cache_key = base_url
    cached = _fetch_loaded_model_name._cache.get(cache_key)
    if cached and _time.time() - cached[1] < 60:
        return cached[0]

    try:
        import httpx as _httpx
        url  = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        resp = _httpx.get(url + "/models", timeout=2.0)
        if resp.status_code == 200:
            data   = resp.json()
            models = data.get("data", [])
            if models:
                name = models[0].get("id", fallback)
                _fetch_loaded_model_name._cache[cache_key] = (name, _time.time())
                return name
    except Exception:
        pass

    _fetch_loaded_model_name._cache[cache_key] = (fallback, _time.time())
    return fallback


_fetch_loaded_model_name._cache: dict = {}  # type: ignore[attr-defined]


def main_model_display_name() -> str:
    """Real filename of the main (4B) model as reported by llama.cpp."""
    return _fetch_loaded_model_name(settings.llm_base_url, settings.llm_model)


def hands_model_display_name() -> str:
    """Real filename of the hands (270M) model as reported by llama.cpp."""
    url   = settings.llm_hands_base_url or settings.llm_base_url
    model = settings.llm_hands_model    or settings.llm_model
    return _fetch_loaded_model_name(url, model)


def combined_display_name() -> str:
    """'main_name / hands_name' for CLI/TUI header. Single name if models are the same."""
    main  = main_model_display_name()
    hands = hands_model_display_name()
    if hands == main:
        return main
    return f"{main} / {hands}"


# ---------------------------------------------------------------------------
# functiongemma inline tool-call parser
# ---------------------------------------------------------------------------

# functiongemma-270M writes tool calls inline in message content rather than
# using the OpenAI tool_calls array. Patterns we handle:
#
#   <tool_call>{"name": "answer", "arguments": {...}}</tool_call>
#   <tool_call>\n{"name": "answer", "arguments": {...}}\n</tool_call>
#   {"name": "answer", "arguments": {...}}          ← bare JSON (no wrapper)

_TOOL_CALL_TAG_RE = _re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    _re.DOTALL,
)
_TOOL_CALL_BARE_RE = _re.compile(
    r'(\{\s*["\']name["\']\s*:.*?["\']arguments["\']\s*:.*?\})',
    _re.DOTALL,
)


def parse_functiongemma_tool_calls(content: str) -> list[dict] | None:
    """Extract tool calls from functiongemma's inline content format.

    Returns a list of dicts with keys: id, name, arguments (dict).
    Returns None when no tool calls are found (caller should treat content
    as a plain text reply).

    Handles:
      • <tool_call>{...}</tool_call>  — primary format
      • Bare JSON object with "name" + "arguments" keys  — fallback
      • Single-quoted JSON (ast.literal_eval fallback)
    """
    if not content:
        return None

    results: list[dict] = []

    def _try_parse(raw: str) -> dict | None:
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass
        # Single-quote fallback (some checkpoints use Python dict syntax)
        try:
            import ast as _ast
            return _ast.literal_eval(raw)
        except Exception:
            return None

    # 1. Tagged form: <tool_call>…</tool_call>
    for m in _TOOL_CALL_TAG_RE.finditer(content):
        obj = _try_parse(m.group(1))
        if obj is None:
            continue
        name = obj.get("name") or obj.get("function") or obj.get("tool")
        args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {"content": args}
        if name:
            results.append({"id": f"fg_{len(results)}", "name": name, "arguments": args})

    # 2. Bare JSON fallback (only if tagged form found nothing)
    if not results:
        for m in _TOOL_CALL_BARE_RE.finditer(content):
            obj = _try_parse(m.group(1))
            if obj is None:
                continue
            name = obj.get("name") or obj.get("function") or obj.get("tool")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {"content": args}
            if name:
                results.append({"id": f"fg_{len(results)}", "name": name, "arguments": args})

    return results if results else None


def is_functiongemma(profile: ModelProfile) -> bool:
    """Return True when this profile points at a functiongemma checkpoint."""
    m = profile.model.lower()
    return "functiongemma" in m or "function-gemma" in m or "function_gemma" in m