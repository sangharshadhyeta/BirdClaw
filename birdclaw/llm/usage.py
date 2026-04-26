"""Token usage tracking and cost estimation.

Port of claw-code-parity/rust/crates/runtime/src/usage.rs.

For local models (Ollama/Gemma) cost = $0.00, but token counts still matter:
    - Feed the compaction threshold in memory/compact.py
    - Show the user how much context is being consumed
    - Multi-model pipelines may mix local + remote models

Pricing table covers known cloud models by name fragment.
Unknown/local models report $0.00 cost with no warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def summary_lines(self, label: str) -> list[str]:
        return self.summary_lines_for_model(label, model=None)

    def summary_lines_for_model(self, label: str, model: str | None) -> list[str]:
        cost = estimate_cost(self, model) if model else UsageCostEstimate()
        model_suffix   = f" model={model}" if model else ""
        pricing_suffix = ""
        return [
            (
                f"{label}: total_tokens={self.total_tokens()}"
                f" input={self.input_tokens}"
                f" output={self.output_tokens}"
                f" cache_write={self.cache_creation_input_tokens}"
                f" cache_read={self.cache_read_input_tokens}"
                f" estimated_cost={format_usd(cost.total_cost_usd())}"
                f"{model_suffix}{pricing_suffix}"
            ),
            (
                f"  cost breakdown:"
                f" input={format_usd(cost.input_cost_usd)}"
                f" output={format_usd(cost.output_cost_usd)}"
                f" cache_write={format_usd(cost.cache_creation_cost_usd)}"
                f" cache_read={format_usd(cost.cache_read_cost_usd)}"
            ),
        ]

    @classmethod
    def from_api_response(cls, usage_dict: dict) -> "TokenUsage":
        """Parse from OpenAI-compat API response usage object."""
        return cls(
            input_tokens=usage_dict.get("prompt_tokens", 0),
            output_tokens=usage_dict.get("completion_tokens", 0),
            cache_creation_input_tokens=usage_dict.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            cache_read_input_tokens=0,
        )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_million: float
    output_cost_per_million: float
    cache_creation_cost_per_million: float = 0.0
    cache_read_cost_per_million: float = 0.0


_PRICING_TABLE: dict[str, ModelPricing] = {
    "haiku": ModelPricing(
        input_cost_per_million=1.0,
        output_cost_per_million=5.0,
        cache_creation_cost_per_million=1.25,
        cache_read_cost_per_million=0.1,
    ),
    "sonnet": ModelPricing(
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        cache_creation_cost_per_million=18.75,
        cache_read_cost_per_million=1.5,
    ),
    "opus": ModelPricing(
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        cache_creation_cost_per_million=18.75,
        cache_read_cost_per_million=1.5,
    ),
    "gpt-4o": ModelPricing(
        input_cost_per_million=5.0,
        output_cost_per_million=15.0,
    ),
    "gpt-4o-mini": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.6,
    ),
    "gpt-4-turbo": ModelPricing(
        input_cost_per_million=10.0,
        output_cost_per_million=30.0,
    ),
    "gpt-3.5": ModelPricing(
        input_cost_per_million=0.5,
        output_cost_per_million=1.5,
    ),
    "llama": ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
    "gemma": ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
    "mistral": ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
    "qwen": ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
    "phi": ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
}

_LOCAL_PRICING = ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0)


def pricing_for_model(model: str) -> ModelPricing:
    """Return pricing for a model by name fragment match. Local/unknown → $0."""
    lower = model.lower()
    for fragment, pricing in _PRICING_TABLE.items():
        if fragment in lower:
            return pricing
    return _LOCAL_PRICING


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

@dataclass
class UsageCostEstimate:
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cache_creation_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0

    def total_cost_usd(self) -> float:
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_creation_cost_usd
            + self.cache_read_cost_usd
        )


def estimate_cost(usage: TokenUsage, model: str) -> UsageCostEstimate:
    pricing = pricing_for_model(model)

    def _cost(tokens: int, per_million: float) -> float:
        return tokens / 1_000_000 * per_million

    return UsageCostEstimate(
        input_cost_usd=_cost(usage.input_tokens, pricing.input_cost_per_million),
        output_cost_usd=_cost(usage.output_tokens, pricing.output_cost_per_million),
        cache_creation_cost_usd=_cost(usage.cache_creation_input_tokens, pricing.cache_creation_cost_per_million),
        cache_read_cost_usd=_cost(usage.cache_read_input_tokens, pricing.cache_read_cost_per_million),
    )


def format_usd(amount: float) -> str:
    return f"${amount:.4f}"


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates token usage, request/response counts across turns.

    request_count  — number of times we sent a completion request to the LLM
    response_count — number of responses received (may differ on errors)
    turns          — alias for response_count (successful LLM turns)
    """

    _latest: TokenUsage = field(default_factory=TokenUsage)
    _cumulative: TokenUsage = field(default_factory=TokenUsage)
    _request_count: int = 0
    _response_count: int = 0

    def record_request(self) -> None:
        """Call before each LLM API call."""
        self._request_count += 1

    def record(self, usage: TokenUsage) -> None:
        """Call after each successful LLM response."""
        self._latest = usage
        self._cumulative = self._cumulative + usage
        self._response_count += 1
        logger.debug(
            "[usage] turn=%d  in=%d  out=%d  total_cum=%d",
            self._response_count, usage.input_tokens, usage.output_tokens,
            self._cumulative.total_tokens(),
        )

    @property
    def latest(self) -> TokenUsage:
        return self._latest

    def current_turn_usage(self) -> TokenUsage:
        return self._latest

    @property
    def cumulative(self) -> TokenUsage:
        return self._cumulative

    @property
    def turns(self) -> int:
        return self._response_count

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def response_count(self) -> int:
        return self._response_count

    @property
    def error_count(self) -> int:
        """Requests that did not get a successful response."""
        return max(0, self._request_count - self._response_count)

    def summary(self, model: str = "") -> str:
        u = self._cumulative
        cost = estimate_cost(u, model) if model else UsageCostEstimate()
        return (
            f"requests={self._request_count} responses={self._response_count} "
            f"tokens_total={u.total_tokens()} "
            f"input={u.input_tokens} output={u.output_tokens} "
            f"cache_write={u.cache_creation_input_tokens} "
            f"cache_read={u.cache_read_input_tokens} "
            f"cost={format_usd(cost.total_cost_usd())}"
            + (f" model={model}" if model else "")
        )
