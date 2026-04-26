"""Content pruner — remove low-information content before LLM injection.

Two tiers:

  keyword_prune(text, goal, max_chars)
      Zero LLM cost. Score every sentence/line by keyword overlap with the
      goal, keep the highest-scoring ones up to max_chars. Used everywhere:
      web_search snippets, GraphRAG context, planning context, file reads.

  semantic_prune(text, goal, max_chars)
      One cheap 270M call. Asks the model to extract only the sentences
      relevant to the goal. Used only for large unstructured content like
      raw web_fetch HTML-stripped text where keyword overlap is noisy.

Both functions are safe to call with empty input — they return "" or the
original text unchanged when content is already short.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Minimum content size to bother pruning — below this just return as-is
_MIN_PRUNE_CHARS = 200

# Stop-words excluded from keyword scoring
_STOP = frozenset({
    "the", "and", "for", "are", "was", "this", "that", "with", "have",
    "from", "they", "will", "been", "had", "has", "its", "not", "but",
    "can", "all", "one", "you", "your", "our", "their", "also", "more",
})


def _tokenise(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOP
    }


def _split_chunks(text: str) -> list[str]:
    """Split text into scoreable units: prefer sentences, fall back to lines."""
    # Try sentence splitting first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 3:
        return sentences
    # Fall back to non-empty lines
    return [l for l in text.splitlines() if l.strip()]


def keyword_prune(text: str, goal: str, max_chars: int = 800) -> str:
    """Score chunks by keyword overlap with goal; return top chunks up to max_chars.

    Preserves reading order within the budget: highest-scoring chunks go in
    first, then fill remaining budget with lower-scoring ones if space allows.
    """
    if not text or len(text) <= _MIN_PRUNE_CHARS:
        return text
    if not goal:
        return text[:max_chars]

    goal_tokens = _tokenise(goal)
    if not goal_tokens:
        return text[:max_chars]

    chunks = _split_chunks(text)
    scored: list[tuple[int, int, str]] = []  # (score, original_index, chunk)
    for i, chunk in enumerate(chunks):
        score = len(goal_tokens & _tokenise(chunk))
        scored.append((score, i, chunk))

    # Sort by score desc, then by original position for tie-breaking
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Collect chunks in score order until budget exhausted
    selected_indices: set[int] = set()
    budget = max_chars
    for score, idx, chunk in scored:
        if budget <= 0:
            break
        selected_indices.add(idx)
        budget -= len(chunk) + 1

    # Reconstruct in original reading order
    result = "\n".join(
        chunk for i, chunk in enumerate(chunks)
        if i in selected_indices
    )
    return result.strip() or text[:max_chars]


def semantic_prune(text: str, goal: str, max_chars: int = 800) -> str:
    """Use 270M to extract sentences relevant to goal from large unstructured text.

    Only called for content that keyword pruning handles poorly (e.g., web pages
    where the goal vocabulary doesn't appear verbatim in the relevant section).
    Adds ~0.3s latency — only use for async/background paths.

    Falls back to keyword_prune on any error.
    """
    if not text or len(text) <= _MIN_PRUNE_CHARS:
        return text
    if not goal:
        return text[:max_chars]

    # If keyword pruning already brings us under budget, use it — no LLM needed
    kw_result = keyword_prune(text, goal, max_chars)
    if len(kw_result) <= max_chars * 1.1:
        return kw_result

    try:
        from birdclaw.llm.client import llm_client
        from birdclaw.llm.model_profile import main_profile
        from birdclaw.llm.types import Message

        # Trim input to 2000 chars to keep 270M context manageable
        trimmed = text[:2000]
        result = llm_client.generate(
            [
                Message(
                    role="system",
                    content=(
                        "Extract only the sentences from the text that are relevant to the goal. "
                        "Return them as plain text. Omit navigation, ads, headers, footers, "
                        "unrelated paragraphs. Keep all relevant technical details verbatim."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Goal: {goal[:200]}\n\nText:\n{trimmed}",
                ),
            ],
            thinking=False,
            profile=main_profile(),
        )
        extracted = (result.content or "").strip()
        if extracted and len(extracted) > 50:
            logger.debug("semantic_prune: %d→%d chars", len(text), len(extracted))
            return extracted[:max_chars]
    except Exception as exc:
        logger.debug("semantic_prune: LLM failed (%s) — falling back to keyword_prune", exc)

    return kw_result
