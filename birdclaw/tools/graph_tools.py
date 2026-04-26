"""Graph tools — knowledge graph operations exposed as agent tools.

Registered in the tool registry so they are:
  - Available to the model during the memorise phase
  - Exposed via McpToolBridge to external agents
  - Accessible to any sub-agent that connects to the graph MCP server

Tools:
    graph_search   — find nodes by keyword
    graph_get      — get one node + its neighbours
    graph_add      — upsert a node (fact, entity, concept, etc.)
    graph_relate   — add a directed edge between two nodes
"""

from __future__ import annotations

import json
import logging

from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _graph_search(query: str, limit: int = 10) -> str:
    from birdclaw.memory.graph import knowledge_graph, session_graph

    # Search both; session nodes shadow knowledge nodes on key collision
    k_results = {n["key"]: n for n in knowledge_graph.search(query, limit=limit)}
    s_results = {n["key"]: n for n in session_graph.search(query, limit=limit)}
    merged = {**k_results, **s_results}

    ranked = sorted(merged.values(), key=lambda n: len(n.get("summary", "")), reverse=True)
    nodes = []
    for n in ranked[:limit]:
        nodes.append({
            "name":    n.get("name", n["key"]),
            "type":    n.get("type", ""),
            "summary": n.get("summary", ""),
        })
    logger.debug("[graph] search  query=%r  results=%d", query[:40], len(nodes))
    return json.dumps({"nodes": nodes, "count": len(nodes)})


def _graph_get(name: str) -> str:
    from birdclaw.memory.graph import knowledge_graph, session_graph

    # Session graph takes priority
    node = session_graph.get_node(name) or knowledge_graph.get_node(name)
    if not node:
        return json.dumps({"error": f"node not found: {name!r}"})

    # Fetch neighbours via BFS depth 1
    graph = session_graph if session_graph.get_node(name) else knowledge_graph
    bfs = graph.bfs([name], depth=1)
    neighbours = []
    if bfs:
        for nb in bfs[0].get("neighbors", []):
            neighbours.append({
                "name":      nb.get("name", ""),
                "relation":  nb.get("relation", ""),
                "direction": nb.get("direction", "out"),
            })

    return json.dumps({
        "name":       node.get("name", name),
        "type":       node.get("type", ""),
        "summary":    node.get("summary", ""),
        "sources":    node.get("sources", []),
        "neighbours": neighbours[:10],
    })


def _graph_add(
    name: str,
    type: str = "fact",
    summary: str = "",
    source: str = "",
) -> str:
    from birdclaw.memory.graph import knowledge_graph

    valid_types = {
        "fact", "entity", "concept", "task", "tool_result",
        "person", "org", "place", "file", "function", "class",
        "module", "tool",
    }
    node_type = type.lower() if type.lower() in valid_types else "fact"
    sources = [source] if source else []
    key = knowledge_graph.upsert_node(
        name=name,
        node_type=node_type,
        summary=summary,
        sources=sources,
    )
    knowledge_graph.save()
    logger.info("[graph] add  name=%r  type=%s  key=%s", name[:40], node_type, key)
    return json.dumps({"ok": True, "key": key, "name": name, "type": node_type})


def _graph_relate(
    subject: str,
    predicate: str,
    target: str,
) -> str:
    from birdclaw.memory.graph import knowledge_graph

    knowledge_graph.upsert_edge(subject, predicate, target)
    knowledge_graph.save()
    return json.dumps({"ok": True, "subject": subject, "predicate": predicate, "object": target})


# ---------------------------------------------------------------------------
# Tool schemas + registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="graph_search",
    description=(
        "Search the knowledge graph for nodes matching a query. "
        "Returns up to `limit` matching nodes with name, type, and summary. "
        "Use this before graph_add to check if something is already known."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword search query."},
            "limit": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["query"],
    },
    handler=lambda **kw: _graph_search(**kw),
    tags=["memory", "graph", "search", "knowledge"],
))

registry.register(Tool(
    name="graph_get",
    description=(
        "Retrieve a specific node from the knowledge graph by name, "
        "including its neighbours (up to 10)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact node name to retrieve."},
        },
        "required": ["name"],
    },
    handler=lambda **kw: _graph_get(**kw),
    tags=["memory", "graph", "knowledge"],
))

registry.register(Tool(
    name="graph_add",
    description=(
        "Add or update a node in the knowledge graph. "
        "Use for facts, entities, concepts, code symbols, decisions, etc. "
        "Check graph_search first to avoid duplicates."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name":    {"type": "string",  "description": "Node name (unique key)."},
            "type":    {"type": "string",  "description": "Node type: fact | entity | concept | file | function | class | module | tool | person | org | place | task."},
            "summary": {"type": "string",  "description": "One-sentence description of what this node represents."},
            "source":  {"type": "string",  "description": "Where this knowledge came from (URL, file path, session ID)."},
        },
        "required": ["name", "summary"],
    },
    handler=lambda **kw: _graph_add(**kw),
    tags=["memory", "graph", "write", "knowledge"],
))

registry.register(Tool(
    name="graph_relate",
    description=(
        "Add a directed relationship between two nodes in the knowledge graph. "
        "Both nodes are created as stubs if they don't exist."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "subject":   {"type": "string", "description": "Source node name."},
            "predicate": {"type": "string", "description": "Relationship label (e.g. imports, calls, mentions, depends_on, part_of, related_to)."},
            "object":    {"type": "string", "description": "Target node name."},
        },
        "required": ["subject", "predicate", "object"],
    },
    handler=lambda **kw: _graph_relate(kw["subject"], kw["predicate"], kw["object"]),
    tags=["memory", "graph", "write", "knowledge"],
))
