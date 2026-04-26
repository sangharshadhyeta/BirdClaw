"""Document ingest pipeline — text → propositions → entities → graph.

Design philosophy (local model = unlimited calls):
    Every chunk gets two LLM passes — no cost pressure to skip steps.
    Pass 1: proposition extraction  (what are the atomic facts?)
    Pass 2: entity + relation extraction  (from propositions, not raw chunk)
    Running extraction on propositions rather than raw text gives the model
    cleaner, denser signal — propositions are already pronoun-resolved and
    standalone.

    Multiple model support: ingest_* functions accept an optional model_tag
    so a fast/small model can be used for extraction while the main loop
    uses a larger one.

Entity types (extended for a coding agent):
    PERSON, ORG, PLACE, CONCEPT   — general knowledge
    FILE, FUNCTION, CLASS, MODULE  — code entities
    TOOL, TASK, FACT               — agent-specific

Ingest targets:
    ingest_text(text, source)          — raw string
    ingest_file(path)                  — file from disk
    ingest_url(url)                    — fetch + ingest web content
    ingest_search_results(results)     — bulk ingest from web_search output

All functions return an IngestResult summary.
Both session_graph and knowledge_graph are updated; callers can choose target.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity types (enum values for extraction tool)
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "PERSON", "ORG", "PLACE", "CONCEPT",
    "FILE", "FUNCTION", "CLASS", "MODULE",
    "TOOL", "TASK", "FACT",
]

# ---------------------------------------------------------------------------
# Tool schemas — forced tool_choice so parser never sees free text
# ---------------------------------------------------------------------------

_PROP_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "emit_propositions",
        "description": "Decompose the paragraph into atomic, self-contained facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "propositions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Each item is one complete, standalone fact. "
                        "Replace all pronouns with their referents."
                    ),
                },
            },
            "required": ["propositions"],
        },
    },
}

_ENTITY_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "emit_entities",
        "description": "Output all named entities and relations found in the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":        {"type": "string"},
                            "type":        {"type": "string", "enum": ENTITY_TYPES},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "type", "description"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject":   {"type": "string"},
                            "predicate": {"type": "string"},
                            "object":    {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
            },
            "required": ["entities", "relations"],
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    source: str
    chunks: int = 0
    propositions: int = 0
    entities: int = 0
    relations: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"IngestResult({self.source!r}: "
            f"{self.chunks} chunks, {self.propositions} props, "
            f"{self.entities} entities, {self.relations} relations"
            + (f", {len(self.errors)} errors" if self.errors else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_id(source: str, text: str) -> str:
    payload = source + text[:200]
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _split_paragraphs(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Split on blank lines; merge small blocks; fall back to sentence chunking."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    if len(blocks) <= 1:
        # No blank lines — chunk on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        blocks, current = [], ""
        for s in sentences:
            if len(current) + len(s) > max_chunk_chars and current:
                blocks.append(current.strip())
                current = s
            else:
                current = (current + " " + s).strip()
        if current:
            blocks.append(current)

    # Merge consecutive small blocks up to max_chunk_chars
    merged: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) < max_chunk_chars:
            current = (current + "\n\n" + block).strip()
        else:
            if current:
                merged.append(current)
            current = block
    if current:
        merged.append(current)

    return [b for b in merged if len(b) > 20]


# ---------------------------------------------------------------------------
# LLM extraction passes
# ---------------------------------------------------------------------------

def _extract_propositions(chunk: str, model_tag: str | None = None) -> list[str]:
    """Pass 1: decompose chunk into atomic propositions."""
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.types import Message

    try:
        result = llm_client.generate(
            messages=[Message(role="user", content=chunk)],
            tools=[_PROP_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_propositions"}},
            model=model_tag,
        )
        if result.tool_calls:
            props = result.tool_calls[0].arguments.get("propositions", [])
            return [p.strip() for p in props if p.strip()]
    except Exception as e:
        logger.warning("proposition extraction failed: %s", e)

    return [chunk]  # fallback: treat chunk as one proposition


def _extract_entities(
    propositions: list[str],
    source: str,
    model_tag: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Pass 2: extract entities + relations from propositions.

    Input is the proposition list joined as bullet points — denser signal
    than the raw paragraph.
    """
    from birdclaw.llm.client import llm_client
    from birdclaw.llm.types import Message

    text = "\n".join(f"- {p}" for p in propositions)

    try:
        result = llm_client.generate(
            messages=[
                Message(
                    role="system",
                    content=(
                        "Extract all named entities and relationships from these facts. "
                        f"Source: {source}"
                    ),
                ),
                Message(role="user", content=text),
            ],
            tools=[_ENTITY_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_entities"}},
            model=model_tag,
        )
        if result.tool_calls:
            args = result.tool_calls[0].arguments
            return args.get("entities", []), args.get("relations", [])
    except Exception as e:
        logger.warning("entity extraction failed: %s", e)

    return [], []


# ---------------------------------------------------------------------------
# Graph population
# ---------------------------------------------------------------------------

def _populate_graph(
    propositions: list[str],
    entities: list[dict],
    relations: list[dict],
    source: str,
    graph,
) -> tuple[int, int, int]:
    """Upsert propositions, entities, and relations into the target graph.

    Returns (props_added, entities_added, relations_added).
    """
    props_added = 0
    for prop in propositions:
        graph.upsert_node(
            name=prop[:80],          # node name = truncated proposition
            node_type="fact",
            summary=prop,
            sources=[source],
        )
        props_added += 1

    entities_added = 0
    for e in entities:
        name = e.get("name", "").strip()
        if not name:
            continue
        graph.upsert_node(
            name=name,
            node_type=e.get("type", "CONCEPT").lower(),
            summary=e.get("description", ""),
            sources=[source],
        )
        entities_added += 1

    relations_added = 0
    for r in relations:
        subject = r.get("subject", "").strip()
        predicate = r.get("predicate", "").strip()
        obj = r.get("object", "").strip()
        if subject and predicate and obj:
            graph.upsert_edge(subject, predicate, obj)
            relations_added += 1

    return props_added, entities_added, relations_added


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest_text(
    text: str,
    source: str,
    graph=None,
    model_tag: str | None = None,
) -> IngestResult:
    """Ingest raw text into the graph.

    Args:
        text:       Document content.
        source:     Identifier (file path, URL, session ID, etc.).
        graph:      Target GraphStore (default: session_graph).
        model_tag:  Optional model override for extraction LLM calls.
    """
    from birdclaw.memory.graph import session_graph
    if graph is None:
        graph = session_graph

    result = IngestResult(source=source)
    chunks = _split_paragraphs(text)
    result.chunks = len(chunks)

    for chunk in chunks:
        # Pass 1 — propositions
        props = _extract_propositions(chunk, model_tag)
        result.propositions += len(props)

        # Pass 2 — entities + relations from propositions
        entities, relations = _extract_entities(props, source, model_tag)

        p, e, r = _populate_graph(props, entities, relations, source, graph)
        result.entities += e
        result.relations += r

    logger.info("%s", result)
    return result


def ingest_file(
    path: str,
    graph=None,
    model_tag: str | None = None,
) -> IngestResult:
    """Read a file from disk and ingest its content."""
    from pathlib import Path

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("ingest_file: cannot read %s: %s", path, e)
        return IngestResult(source=path, errors=[str(e)])

    return ingest_text(text, source=str(p), graph=graph, model_tag=model_tag)


def ingest_url(
    url: str,
    graph=None,
    model_tag: str | None = None,
) -> IngestResult:
    """Fetch a URL and ingest its text content.

    Uses the existing web tool's fetch logic so rate limiting and
    character caps are consistent.
    """
    from birdclaw.tools.web import fetch_url

    try:
        raw = fetch_url(url)
    except Exception as e:
        logger.error("ingest_url: fetch failed for %s: %s", url, e)
        return IngestResult(source=url, errors=[str(e)])

    # fetch_url returns a JSON string — extract the text field
    import json
    try:
        data = json.loads(raw)
        text = data.get("text") or data.get("content") or raw
    except (json.JSONDecodeError, AttributeError):
        text = raw

    return ingest_text(text, source=url, graph=graph, model_tag=model_tag)


def ingest_search_results(
    results: list[dict],
    graph=None,
    model_tag: str | None = None,
) -> IngestResult:
    """Ingest a list of web search result dicts (title + snippet + url).

    Each result is treated as a short document. For deep ingest, follow
    with ingest_url on the individual URLs.

    Expected dict format: {"title": "...", "snippet": "...", "url": "..."}
    """
    combined = IngestResult(source="search_results")

    for item in results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        text = f"{title}\n\n{snippet}".strip()
        if not text:
            continue
        r = ingest_text(text, source=url or "search", graph=graph, model_tag=model_tag)
        combined.chunks += r.chunks
        combined.propositions += r.propositions
        combined.entities += r.entities
        combined.relations += r.relations
        combined.errors += r.errors

    logger.info("%s", combined)
    return combined
