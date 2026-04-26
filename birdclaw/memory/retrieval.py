"""Graph retrieval — query → subgraph → compact text for context injection.

Flow:
    1. Tokenise query into keyword set
    2. Search session_graph then knowledge_graph for matching nodes (score by overlap)
    3. Deduplicate (session takes priority over knowledge for same key)
    4. BFS depth-2 from top seed nodes
    5. Render subgraph as compact bullet text, hard-capped at TOKEN_CAP tokens

The output is injected into the agent's system prompt — it must be short.
Token estimate: 1 token ≈ 4 chars (conservative for English/code mix).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from birdclaw.memory.graph import knowledge_graph, session_graph

logger = logging.getLogger(__name__)

# Hard cap on rendered output — ~300 tokens at 4 chars/token
TOKEN_CAP = 300
CHAR_CAP = TOKEN_CAP * 4

# Max seed nodes passed to BFS — keeps subgraph focused
MAX_SEEDS = 3

# BFS depth
BFS_DEPTH = 2


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens split on spaces and underscores, excluding very short words."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


# ---------------------------------------------------------------------------
# Scored node
# ---------------------------------------------------------------------------

@dataclass
class ScoredNode:
    key: str
    name: str
    node_type: str
    summary: str
    score: int
    source: str  # "session" or "knowledge"


# ---------------------------------------------------------------------------
# Search both graphs, deduplicate
# ---------------------------------------------------------------------------

def _search_merged(query: str, limit: int = 10) -> list[ScoredNode]:
    """Search session_graph and knowledge_graph, session taking priority."""
    tokens = _tokenise(query)
    if not tokens:
        return []

    seen: dict[str, ScoredNode] = {}

    for graph, source in ((session_graph, "session"), (knowledge_graph, "knowledge")):
        for node in graph.search(query, limit=limit):
            key = node["key"]
            name = node.get("name", key)
            name_tokens = _tokenise(name)
            summary_tokens = _tokenise(node.get("summary", ""))
            score = len(tokens & (name_tokens | summary_tokens))
            sn = ScoredNode(
                key=key,
                name=name,
                node_type=node.get("type", "entity"),
                summary=node.get("summary", ""),
                score=score,
                source=source,
            )
            # Session graph wins on key collision
            if key not in seen or source == "session":
                seen[key] = sn

    ranked = sorted(seen.values(), key=lambda n: n.score, reverse=True)
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Render subgraph as compact text
# ---------------------------------------------------------------------------

def _render_node(node: dict, neighbors: list[dict]) -> str:
    """Render one node + its neighbours as a compact bullet block."""
    name = node.get("name", node.get("key", "?"))
    ntype = node.get("type", "")
    summary = node.get("summary", "")

    lines = [f"[{ntype}] {name}" + (f" — {summary}" if summary else "")]

    for nb in neighbors[:3]:  # cap neighbours per node
        rel = nb.get("relation", "→")
        direction = nb.get("direction", "out")
        nb_name = nb.get("name", "?")
        arrow = f"→ {rel} →" if direction == "out" else f"← {rel} ←"
        lines.append(f"  {arrow} {nb_name}")

    return "\n".join(lines)


def _render_subgraph(bfs_nodes: list[dict], char_cap: int = CHAR_CAP) -> str:
    """Render a BFS result list into a capped text block."""
    parts: list[str] = []
    total = 0

    for node in bfs_nodes:
        neighbors = node.pop("neighbors", [])
        block = _render_node(node, neighbors)
        if total + len(block) > char_cap:
            break
        parts.append(block)
        total += len(block) + 1  # +1 for newline

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_n: int = MAX_SEEDS) -> str:
    """Main retrieval entry point.

    Returns a compact text block (≤ ~300 tokens) of relevant graph context
    for the given query. Empty string if nothing relevant is found.

    Args:
        query:  The current agent query or step instruction.
        top_n:  Number of seed nodes for BFS (default 3).
    """
    candidates = _search_merged(query, limit=top_n * 2)
    if not candidates:
        return ""

    seeds = [c.name for c in candidates[:top_n]]
    logger.debug("retrieval seeds: %s", seeds)

    # BFS from session first, fall back to knowledge for nodes not in session
    # Merge: session nodes override knowledge nodes with the same key
    session_nodes = {n["key"]: n for n in session_graph.bfs(seeds, depth=BFS_DEPTH)}
    knowledge_nodes = {n["key"]: n for n in knowledge_graph.bfs(seeds, depth=BFS_DEPTH)}

    merged: dict[str, dict] = {**knowledge_nodes, **session_nodes}  # session wins
    bfs_result = list(merged.values())

    if not bfs_result:
        # Seeds found in search but BFS returned nothing — render seeds directly
        bfs_result = [
            {"key": c.key, "name": c.name, "type": c.node_type,
             "summary": c.summary, "neighbors": []}
            for c in candidates[:top_n]
        ]

    rendered = _render_subgraph(bfs_result)
    # Final keyword-prune pass — removes nodes whose content doesn't overlap
    # with the query at all, freeing tokens for the actual reasoning.
    if len(rendered) > 400 and query:
        from birdclaw.llm.pruner import keyword_prune
        rendered = keyword_prune(rendered, goal=query, max_chars=CHAR_CAP)
    logger.debug("retrieval output: %d chars", len(rendered))
    return rendered


def retrieve_top_nodes(query: str, n: int = 3) -> list[str]:
    """Return just the top-N node name strings — used for planning_context().

    Lighter than full BFS; suitable when we only want a name list.
    """
    candidates = _search_merged(query, limit=n)
    return [c.name for c in candidates[:n]]


# ---------------------------------------------------------------------------
# Simple NER — extract entities from text and index into knowledge_graph
# ---------------------------------------------------------------------------

# Patterns for common entities found in tool results
_NER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("file_path",    re.compile(r"(?:^|[\s'\"(])(/(?:[\w.\-]+/)*[\w.\-]+\.\w{1,6})", re.MULTILINE)),
    ("function",     re.compile(r"\bdef\s+(\w{3,})\s*\(", re.MULTILINE)),
    ("class_name",   re.compile(r"\bclass\s+(\w{3,})\s*[:(]", re.MULTILINE)),
    ("error_type",   re.compile(r"\b(\w+(?:Error|Exception|Warning))\b")),
    ("import_path",  re.compile(r"(?:from|import)\s+([\w.]{4,})", re.MULTILINE)),
    ("url",          re.compile(r"https?://[\w.\-/?=&%#@:+]{10,80}")),
    ("package",      re.compile(r"\b([a-z][a-z0-9_\-]{2,})==[\d.]+")),  # pip freeze style
]

# Hard cap: don't extract more than this many entities per text block
_MAX_ENTITIES = 20


def extract_and_index(text: str, context: str = "") -> int:
    """Extract named entities from text and upsert them into knowledge_graph.

    Returns the number of new/updated nodes added.

    Args:
        text:    Raw text to mine (web_search result, web_fetch page, etc.)
        context: Short description of the source (e.g. tool name + query snippet).
    """
    if not text or len(text) < 20:
        return 0

    entities: dict[str, dict] = {}   # key → node attrs

    for entity_type, pattern in _NER_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1) if pattern.groups else m.group(0)
            name = name.strip("'\"() ")
            if len(name) < 3 or len(name) > 120:
                continue
            key = f"{entity_type}:{name}"
            if key not in entities:
                entities[key] = {
                    "key":     key,
                    "name":    name,
                    "type":    entity_type,
                    "summary": context[:80] if context else "",
                }
            if len(entities) >= _MAX_ENTITIES:
                break
        if len(entities) >= _MAX_ENTITIES:
            break

    for node in entities.values():
        try:
            knowledge_graph.upsert_node(
                name=node["name"],
                node_type=node["type"],
                summary=node["summary"],
            )
        except Exception:
            pass

    logger.debug("NER extracted %d entities from %d chars", len(entities), len(text))
    return len(entities)
