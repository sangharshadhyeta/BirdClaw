"""NetworkX-based knowledge graph store.

Two graph instances:
    session_graph    — ephemeral, this session only (in-memory, not persisted)
    knowledge_graph  — persistent across sessions (~/.birdclaw/memory/graph.json)

The dreaming phase (Phase 7) merges session_graph → knowledge_graph.

Node types:
    entity      — extracted concept, person, project, file, function, class
    fact        — a specific proposition extracted from a document
    task        — a user request or goal
    tool_result — cached result of a deterministic tool call (used by tool_cache)

Node attrs (all types):
    name        str   — display name
    type        str   — node type (see above)
    summary     str   — short description / proposition text
    sources     list  — file paths, URLs, or session IDs that contributed this node
    last_seen   str   — ISO timestamp of last update

Edge attrs:
    relation    str   — imports / calls / mentions / depends_on / related_to / cached_by
    weight      float — confidence / frequency (default 1.0)

Persistence:
    JSON node-link format (networkx.node_link_data) — human-readable, not pickle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

from birdclaw.config import settings

logger = logging.getLogger(__name__)

_NODE_TYPES = frozenset({"entity", "fact", "task", "tool_result"})
_RELATIONS = frozenset({
    "imports", "calls", "mentions", "depends_on", "related_to", "cached_by",
    "part_of", "produced_by",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_key(name: str) -> str:
    """Normalised node key — lower-case, stripped."""
    return name.lower().strip()


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------

class GraphStore:
    """Wrapper around a NetworkX DiGraph with typed nodes and JSON persistence."""

    def __init__(self, persist_path: Path | None = None) -> None:
        """
        Args:
            persist_path: If given, load/save from this JSON file.
                          If None, graph is in-memory only (session graph).
        """
        self._path = persist_path
        self._graph: nx.DiGraph = nx.DiGraph()
        if persist_path and persist_path.exists():
            self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        assert self._path is not None
        bak = self._path.with_suffix(self._path.suffix + ".bak")
        for candidate in (self._path, bak):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                self._graph = nx.node_link_graph(data, directed=True, multigraph=False, edges="edges")
                logger.info(
                    "loaded knowledge graph from %s (%d nodes, %d edges)",
                    candidate, self.node_count(), self.edge_count(),
                )
                return
            except Exception as e:
                logger.warning("could not load graph from %s: %s — trying backup", candidate, e)
        logger.warning("no valid graph file found — starting fresh")
        self._graph = nx.DiGraph()

    def save(self) -> None:
        """Persist the knowledge graph to disk atomically (no-op for session graph)."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph, edges="edges")
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        bak = self._path.with_suffix(self._path.suffix + ".bak")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        if self._path.exists():
            try:
                self._path.replace(bak)
            except OSError:
                pass
        tmp.replace(self._path)
        logger.info(
            "saved knowledge graph to %s (%d nodes, %d edges)",
            self._path, self.node_count(), self.edge_count(),
        )

    # ── Node upsert ───────────────────────────────────────────────────────────

    def upsert_node(
        self,
        name: str,
        node_type: str,
        summary: str = "",
        sources: list[str] | None = None,
        **extra: Any,
    ) -> str:
        """Insert or update a node. Returns the node key."""
        key = _node_key(name)
        ts = _now()

        if self._graph.has_node(key):
            node = self._graph.nodes[key]
            # Merge sources
            existing = set(node.get("sources", []))
            existing.update(sources or [])
            node["sources"] = list(existing)
            node["updated_at"] = ts
            node["last_seen"] = ts  # keep for backwards compat
            if summary:
                node["summary"] = summary
            node.update(extra)
        else:
            self._graph.add_node(
                key,
                name=name,
                type=node_type,
                summary=summary,
                sources=list(sources or []),
                created_at=ts,
                updated_at=ts,
                last_seen=ts,  # keep for backwards compat
                **extra,
            )
        return key

    def upsert_edge(
        self,
        subject: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
    ) -> None:
        """Insert or strengthen a directed edge between two nodes."""
        s_key = _node_key(subject)
        o_key = _node_key(obj)

        # Ensure nodes exist as minimal stubs
        for key, name in ((s_key, subject), (o_key, obj)):
            if not self._graph.has_node(key):
                self._graph.add_node(
                    key,
                    name=name,
                    type="entity",
                    summary="",
                    sources=[],
                    last_seen=_now(),
                )

        if self._graph.has_edge(s_key, o_key):
            self._graph.edges[s_key, o_key]["weight"] = (
                self._graph.edges[s_key, o_key].get("weight", 1.0) + weight
            )
        else:
            self._graph.add_edge(s_key, o_key, relation=relation, weight=weight)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_node(self, name: str) -> dict | None:
        """Return node attrs dict or None."""
        key = _node_key(name)
        if self._graph.has_node(key):
            return {"key": key, **dict(self._graph.nodes[key])}
        return None

    def search(self, query: str, limit: int = 10, node_type: str | None = None) -> list[dict]:
        """Token-overlap search over node names and summaries."""
        import re
        def _tok(text: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

        tokens = _tok(query)
        scored: list[tuple[int, dict]] = []

        for key, data in self._graph.nodes(data=True):
            if node_type and data.get("type") != node_type:
                continue
            name_tokens = _tok(data.get("name", ""))
            summary_tokens = _tok(data.get("summary", ""))
            score = len(tokens & (name_tokens | summary_tokens))
            if score > 0:
                scored.append((score, {"key": key, **data}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def bfs(self, seeds: list[str], depth: int = 2) -> list[dict]:
        """BFS from seed node names. Returns all reachable node dicts within depth.

        Used by retrieval to build a subgraph context snippet.
        """
        visited: set[str] = set()
        frontier: set[str] = set()

        for name in seeds:
            key = _node_key(name)
            if self._graph.has_node(key):
                frontier.add(key)

        result: list[dict] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for key in frontier:
                if key in visited:
                    continue
                visited.add(key)
                data = dict(self._graph.nodes[key])
                neighbors = []
                for n in self._graph.successors(key):
                    edge = self._graph.edges[key, n]
                    neighbors.append({
                        "name": self._graph.nodes[n].get("name", n),
                        "relation": edge.get("relation", ""),
                        "direction": "out",
                    })
                for n in self._graph.predecessors(key):
                    edge = self._graph.edges[n, key]
                    neighbors.append({
                        "name": self._graph.nodes[n].get("name", n),
                        "relation": edge.get("relation", ""),
                        "direction": "in",
                    })
                result.append({"key": key, "neighbors": neighbors, **data})
                next_frontier.update(self._graph.successors(key))
                next_frontier.update(self._graph.predecessors(key))
            frontier = next_frontier - visited

        return result

    def nodes_by_type(self, node_type: str) -> Iterator[dict]:
        for key, data in self._graph.nodes(data=True):
            if data.get("type") == node_type:
                yield {"key": key, **data}

    def remove_node(self, name: str) -> bool:
        key = _node_key(name)
        if self._graph.has_node(key):
            self._graph.remove_node(key)
            return True
        return False

    def merge_from(self, other: "GraphStore") -> None:
        """Merge all nodes and edges from another graph into this one.

        Used by dreaming: session_graph → knowledge_graph.
        """
        for key, data in other._graph.nodes(data=True):
            self.upsert_node(
                name=data.get("name", key),
                node_type=data.get("type", "entity"),
                summary=data.get("summary", ""),
                sources=data.get("sources", []),
            )
        for u, v, data in other._graph.edges(data=True):
            self.upsert_edge(
                subject=other._graph.nodes[u].get("name", u),
                relation=data.get("relation", "related_to"),
                obj=other._graph.nodes[v].get("name", v),
                weight=data.get("weight", 1.0),
            )

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

def _knowledge_graph_path() -> Path:
    return settings.data_dir / "memory" / "graph.json"


# Persistent knowledge graph — loaded once, saved explicitly
knowledge_graph = GraphStore(persist_path=_knowledge_graph_path())

# Session graph — ephemeral, in-memory only; merged into knowledge_graph during dreaming
session_graph = GraphStore(persist_path=None)
