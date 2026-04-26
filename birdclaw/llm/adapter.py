"""Response format adapter.

Normalises tool calls from any model format into our internal GenerationResult.
The agent loop never touches raw API responses — everything comes through here.

Formats handled (in priority order):
  1. OpenAI native  — finish_reason="tool_calls", msg["tool_calls"] present
  2. Gemma native   — same as above but msg["reasoning"] carries thinking
  3. XML block      — <tool_call>{"name":...,"arguments":...}</tool_call> in content
  4. JSON code block — ```json\\n{"name":...,"arguments":...}\\n``` in content
  5. Plain text     — treat entire content as an answer

Also handles L12 (tool name hallucination) via fuzzy name matching against
the registered tool set.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import get_close_matches

from birdclaw.llm.types import GenerationResult, ToolCall
from birdclaw.llm.usage import TokenUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# <think>...</think> blocks (may be multi-line)
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

# <tool_call> ... </tool_call>  (Qwen / Mistral XML style)
_XML_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# ```json ... ``` or ``` ... ``` code fences containing a tool-call-shaped object
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Bare JSON object at start/end of message (last-resort)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# FunctionGemma format: <start_function_call>call:TOOL{k:<escape>v<escape>}<end_function_call>
# The model often emits multiple calls; we take only the first valid one.
_FUNCGEMMA_CALL_RE = re.compile(
    r"<start_function_call>\s*call:(\w+)\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
# Key-value pairs inside the call block: key:<escape>value<escape>
# Fallback (missing closing <escape>): key:<escape>value   until } or next key
_FUNCGEMMA_ARG_RE = re.compile(r"(\w+):<escape>(.*?)(?:<escape>|(?=\s*\w+:|$))", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> tuple[str, str]:
    """Remove <think>…</think> blocks. Returns (clean_text, joined_thinking)."""
    captured: list[str] = []

    def _grab(m: re.Match) -> str:
        captured.append(m.group(1).strip())
        return ""

    clean = _THINK_RE.sub(_grab, text).strip()
    return clean, "\n---\n".join(captured)


def _fuzzy_tool_name(name: str, known: set[str]) -> str:
    """Return the best matching known tool name, or the original if no match."""
    if name in known:
        return name
    matches = get_close_matches(name, known, n=1, cutoff=0.7)
    if matches:
        logger.warning("Tool name %r corrected to %r (fuzzy match)", name, matches[0])
        return matches[0]
    return name


def _repair_json_args(text: str) -> str:
    """Best-effort fix for common small-model JSON errors in tool arguments."""
    import re as _re
    text = text.strip()
    # Remove trailing commas before } or ]
    text = _re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted strings with double-quoted
    text = _re.sub(r"(?<![\\])'([^']*)'", r'"\1"', text)
    return text


def _parse_arguments(raw: str | dict) -> dict:
    """Parse tool arguments from string or dict, returning {} on failure.

    Handles: dict pass-through, valid JSON, JSON with common errors
    (trailing commas, single quotes), and bare {...} extraction.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try JSON repair
    try:
        return json.loads(_repair_json_args(raw))
    except json.JSONDecodeError:
        pass
    # Try extracting the first {...} block
    m = _BARE_JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json_args(m.group()))
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse tool arguments: %r", raw[:200])
    return {}


def _make_tool_call(obj: dict, known_tools: set[str], index: int) -> ToolCall | None:
    """Build a ToolCall from a parsed dict, applying fuzzy name correction."""
    # Support both {name, arguments} and {function: {name, arguments}}
    if "function" in obj:
        obj = obj["function"]

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name", "")
    if not name:
        return None

    name = _fuzzy_tool_name(str(name), known_tools)
    arguments = _parse_arguments(obj.get("arguments") or obj.get("parameters") or obj.get("args") or {})
    call_id = obj.get("id") or f"call_{index}"

    return ToolCall(name=name, arguments=arguments, id=call_id)


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _parse_openai_tool_calls(msg: dict, known_tools: set[str]) -> list[ToolCall]:
    """Parse standard OpenAI tool_calls array from the message."""
    result = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        name = tc.get("function", {}).get("name", "")
        name = _fuzzy_tool_name(name, known_tools)
        args = _parse_arguments(tc.get("function", {}).get("arguments") or {})
        result.append(ToolCall(name=name, arguments=args, id=tc.get("id") or f"call_{i}"))
    return result


def _parse_xml_tool_calls(content: str, known_tools: set[str]) -> list[ToolCall]:
    """Parse <tool_call>...</tool_call> blocks from content."""
    result = []
    for i, m in enumerate(_XML_TOOL_RE.finditer(content)):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        tc = _make_tool_call(obj, known_tools, i)
        if tc:
            result.append(tc)
    return result


def _parse_funcgemma_tool_calls(content: str, known_tools: set[str]) -> list[ToolCall]:
    """Parse FunctionGemma's native format: <start_function_call>call:NAME{k:<escape>v<escape>}..."""
    result = []
    for i, m in enumerate(_FUNCGEMMA_CALL_RE.finditer(content)):
        raw_name = m.group(1).strip()
        args_block = m.group(2)
        name = _fuzzy_tool_name(raw_name, known_tools)
        args: dict = {}
        for am in _FUNCGEMMA_ARG_RE.finditer(args_block):
            args[am.group(1)] = am.group(2).strip()
        result.append(ToolCall(name=name, arguments=args, id=f"call_{i}"))
    # Only return the first valid call — model tends to hallucinate duplicates
    return result[:1]


def _parse_json_fence_tool_calls(content: str, known_tools: set[str]) -> list[ToolCall]:
    """Parse ```json {...} ``` code fences that look like tool calls."""
    result = []
    for i, m in enumerate(_JSON_FENCE_RE.finditer(content)):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # Only treat as a tool call if it has a recognisable name field
        if not any(k in obj for k in ("name", "tool", "tool_name", "function")):
            continue
        tc = _make_tool_call(obj, known_tools, i)
        if tc:
            result.append(tc)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_response(raw: dict, known_tools: set[str] | None = None) -> GenerationResult:
    """Convert a raw Ollama/OpenAI API response dict into a GenerationResult.

    Args:
        raw:         The full JSON response from the completions endpoint.
        known_tools: Set of registered tool names for fuzzy matching.
                     If None, no fuzzy correction is applied.

    Returns:
        GenerationResult with normalised tool_calls (may be empty).
    """
    tools = known_tools or set()
    choice = raw["choices"][0]
    finish_reason: str = choice.get("finish_reason", "stop")
    msg = choice["message"]

    # Parse token usage if present
    usage: TokenUsage | None = None
    if "usage" in raw:
        usage = TokenUsage.from_api_response(raw["usage"])

    # --- Thinking extraction ---
    # Gemma 4 puts reasoning in msg["reasoning"]; others use <think> tags in content
    native_reasoning: str = msg.get("reasoning") or ""
    content_raw: str = msg.get("content") or ""
    content, think_from_tags = _strip_thinking(content_raw)
    thinking = "\n---\n".join(filter(None, [native_reasoning.strip(), think_from_tags]))

    # --- Path 1: Standard OpenAI / Gemma native tool_calls ---
    if msg.get("tool_calls"):
        tool_calls = _parse_openai_tool_calls(msg, tools)
        if tool_calls:
            logger.debug("adapter: openai format, %d tool call(s)", len(tool_calls))
            return GenerationResult(
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage=usage,
            )

    # --- Path 2: XML <tool_call> blocks in content ---
    if "<tool_call>" in content:
        tool_calls = _parse_xml_tool_calls(content, tools)
        if tool_calls:
            logger.debug("adapter: xml format, %d tool call(s)", len(tool_calls))
            # Strip the XML from the visible content
            clean_content = _XML_TOOL_RE.sub("", content).strip()
            return GenerationResult(
                content=clean_content,
                thinking=thinking,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage=usage,
            )

    # --- Path 3: JSON code fence tool calls ---
    if "```" in content:
        tool_calls = _parse_json_fence_tool_calls(content, tools)
        if tool_calls:
            logger.debug("adapter: json-fence format, %d tool call(s)", len(tool_calls))
            clean_content = _JSON_FENCE_RE.sub("", content).strip()
            return GenerationResult(
                content=clean_content,
                thinking=thinking,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage=usage,
            )

    # --- Path 4: FunctionGemma native format ---
    if "<start_function_call>" in content:
        tool_calls = _parse_funcgemma_tool_calls(content, tools)
        if tool_calls:
            logger.debug("adapter: funcgemma format, %d tool call(s)", len(tool_calls))
            clean_content = _FUNCGEMMA_CALL_RE.sub("", content).strip()
            return GenerationResult(
                content=clean_content,
                thinking=thinking,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage=usage,
            )

    # --- Path 5: Plain text — treat as answer ---
    logger.debug("adapter: plain text (finish_reason=%s, %d chars)", finish_reason, len(content))
    return GenerationResult(
        content=content,
        thinking=thinking,
        tool_calls=[],
        finish_reason=finish_reason,
        usage=usage,
    )
