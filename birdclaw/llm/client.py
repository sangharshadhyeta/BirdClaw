"""LLM client — OpenAI-compatible API.

Talks to any OpenAI-compatible endpoint (llama.cpp default).

Configure via ~/.birdclaw/.env:
    BC_LLM_BASE_URL=http://localhost:8081/v1
    BC_LLM_MODEL=gemma-4-e4b-it-Q8_0.gguf
    BC_LLM_HANDS_BASE_URL=http://localhost:8082/v1   (optional 270M hands model)
    BC_LLM_HANDS_MODEL=functiongemma-270m.gguf        (optional)

For parallel inference, start llama-server with --parallel N.
Multiple threads can call generate() concurrently — httpx.Client is thread-safe.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from birdclaw.config import settings
from birdclaw.llm.adapter import parse_response
from birdclaw.llm.types import GenerationResult, Message, ToolCall
from birdclaw.llm.usage import UsageTracker

if TYPE_CHECKING:
    from birdclaw.llm.model_profile import ModelProfile

logger = logging.getLogger(__name__)


def _build_response_format(format_schema: dict | None) -> dict | None:
    if not format_schema:
        return None
    # Pre-built strict schemas (from llm/schemas.py) pass through unchanged —
    # llama.cpp converts them to GBNF grammar for token-level constraint.
    if format_schema.get("type") == "json_schema":
        return format_schema
    # Legacy json_object mode — loose JSON, no grammar constraint.
    if format_schema.get("type") == "json_object":
        return format_schema
    # Raw dict schema — wrap with strict=True for grammar-constrained decoding.
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": format_schema},
    }


class LLMClient:
    """Thread-safe httpx client for any OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=120.0)
        self.usage = UsageTracker()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        return headers

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        model: str | None = None,
        thinking: bool = True,
        format_schema: dict[str, Any] | None = None,
        _bypass_scheduler: bool = False,
        profile: "ModelProfile | None" = None,
    ) -> GenerationResult:
        """Send a chat completion request and return a structured result.

        Args:
            messages:      Conversation history.
            tools:         OpenAI-format tool schemas. Mutually exclusive with format_schema.
            tool_choice:   "auto" (default) or "required". Ignored if no tools.
            model:         Override the model name for this call.
            thinking:      Whether the caller wants thinking enabled. Controls whether
                           the think tool is offered — not sent to the API directly.
            format_schema: Request structured JSON output. Cannot be combined with tools.
            profile:       ModelProfile to use for endpoint + model selection.
                           If provided, overrides model and base_url for this call.
        """
        if not _bypass_scheduler and settings.llm_scheduler_enabled:
            from birdclaw.llm.scheduler import get_scheduler
            from birdclaw.tools.context_vars import get_llm_priority
            scheduler = get_scheduler()
            priority  = get_llm_priority()
            fut = scheduler.submit(
                lambda: self.generate(
                    messages, tools=tools, tool_choice=tool_choice,
                    model=model, thinking=thinking, format_schema=format_schema,
                    _bypass_scheduler=True, profile=profile,
                ),
                priority=priority,
            )
            return fut.result()

        # Profile overrides take precedence over explicit model arg and settings
        active_model = (profile.model if profile else None) or model or settings.llm_model
        base_url     = (profile.base_url if profile else None) or settings.llm_base_url

        payload: dict[str, Any] = {
            "model": active_model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "stream": False,
        }

        response_format = _build_response_format(format_schema)
        if response_format:
            payload["response_format"] = response_format
        elif tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        url = f"{base_url.rstrip('/')}/chat/completions"
        logger.debug("POST %s  model=%s  profile=%s", url, active_model, profile.name if profile else "default")
        self.usage.record_request()

        known_tools: set[str] = set()
        if tools:
            known_tools = {t["function"]["name"] for t in tools if "function" in t}

        # Retry on transient errors or empty content — small models occasionally
        # time out or return an empty message on the first attempt.
        import time as _time
        last_exc: Exception | None = None
        result: GenerationResult | None = None
        _t0 = _time.perf_counter()
        for _attempt in range(4):
            if _attempt > 0:
                _time.sleep(1.0 * _attempt)
                logger.warning("[llm] retry %d/4 after: %s", _attempt + 1, last_exc)
            try:
                resp = self._http.post(url, json=payload, headers=self._headers())
                if not resp.is_success:
                    logger.error("[llm] HTTP %d: %s", resp.status_code, resp.text[:300])
                    # Forgiving: auto-trim on context overflow and retry
                    if resp.status_code == 400:
                        try:
                            err = resp.json().get("error", {})
                            if err.get("type") == "exceed_context_size_error":
                                msgs = payload["messages"]
                                if len(msgs) > 4:
                                    # keep system + first user msg + last 3 exchanges
                                    trimmed = msgs[:2] + msgs[-3:]
                                    payload["messages"] = trimmed
                                    logger.warning(
                                        "context overflow (%d tokens) — trimmed %d→%d messages, retrying",
                                        err.get("n_prompt_tokens", 0),
                                        len(msgs), len(trimmed),
                                    )
                                    last_exc = ValueError(
                                        f"context overflow trimmed: {len(msgs)}→{len(trimmed)} msgs"
                                    )
                                    continue
                        except Exception:
                            pass
                resp.raise_for_status()
                result = parse_response(resp.json(), known_tools=known_tools)

                # ── functiongemma inline tool-call extraction ──────────────
                # functiongemma-270M writes tool calls as <tool_call>{...}</tool_call>
                # inside the message content instead of the OpenAI tool_calls array.
                # Parse and promote them so the rest of the agent loop is unaffected.
                if not result.tool_calls and tools and profile is not None:
                    from birdclaw.llm.model_profile import (
                        is_functiongemma,
                        parse_functiongemma_tool_calls,
                    )
                    if is_functiongemma(profile) and result.content:
                        parsed_calls = parse_functiongemma_tool_calls(result.content)
                        if parsed_calls:
                            logger.debug(
                                "functiongemma: extracted %d inline tool call(s): %s",
                                len(parsed_calls),
                                [c["name"] for c in parsed_calls],
                            )
                            result = GenerationResult(
                                content=result.content,
                                tool_calls=[
                                    ToolCall(
                                        id=c["id"],
                                        name=c["name"],
                                        arguments=c["arguments"],
                                    )
                                    for c in parsed_calls
                                ],
                                usage=result.usage,
                                thinking=result.thinking,
                            )

                # Retry if model returned neither content nor tool calls (empty response)
                if not result.content and not result.tool_calls and not result.thinking:
                    last_exc = ValueError("empty LLM response (no content, no tool calls)")
                    logger.warning("[llm] empty response attempt %d — retrying", _attempt + 1)
                    continue
                if result.usage:
                    self.usage.record(result.usage)

                # 7c: log timing + tokens on every successful call
                _elapsed = _time.perf_counter() - _t0
                _in  = result.usage.input_tokens if result.usage else 0
                _out = result.usage.output_tokens if result.usage else 0
                _tc  = result.tool_calls[0].name if result.tool_calls else None
                logger.info(
                    "[llm] model=%s  msgs=%d  →%s  time=%.1fs  in=%d out=%d",
                    active_model.split("/")[-1][:30],
                    len(payload["messages"]),
                    f"tool:{_tc}" if _tc else f"content:{len(result.content or '')}ch",
                    _elapsed, _in, _out,
                )
                return result
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                continue
            except httpx.HTTPStatusError:
                raise

        # All retries exhausted — re-raise or return last result
        if last_exc and result is None:
            raise last_exc  # type: ignore[misc]
        return result  # type: ignore[return-value]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# Module-level singleton — import this everywhere
llm_client = LLMClient()