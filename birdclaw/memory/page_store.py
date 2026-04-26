"""Page store — LLM-cleaned web content persisted for future retrieval.

The graph stores entities and relations (navigation layer).
The page store stores the actual cleaned content those nodes point to
(reference layer). When the agent needs to recall what a URL contained,
it follows the graph node's page_store_key to the full cleaned text here.

Storage:
    ~/.birdclaw/pages/<url_hash>.json     one file per URL
    Each file: { url, fetched_at, cleaned, source_tool, graph_node_key }

The condenser writes here after LLM-cleaning a web fetch result.
The memorise worker reads here to extract graph nodes.
The retrieval layer reads here to surface full content on demand.

File size is bounded by the condenser — only LLM-condensed content is
stored, never raw HTML. Typical entry: 500–3000 chars.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from birdclaw.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@dataclass
class PageEntry:
    url:            str
    fetched_at:     float
    cleaned:        str          # LLM-condensed markdown
    source_tool:    str          # "web_fetch" | "web_search"
    graph_node_key: str = ""     # backlink to graph node (set after memorise)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PageEntry":
        return cls(
            url=d["url"],
            fetched_at=d.get("fetched_at", 0.0),
            cleaned=d.get("cleaned", ""),
            source_tool=d.get("source_tool", ""),
            graph_node_key=d.get("graph_node_key", ""),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pages_dir() -> Path:
    d = settings.data_dir / "pages"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _entry_path(url: str) -> Path:
    return _pages_dir() / f"{_url_hash(url)}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def put(url: str, cleaned: str, source_tool: str = "web_fetch") -> PageEntry:
    """Store a cleaned page entry. Overwrites if URL already exists."""
    entry = PageEntry(
        url=url,
        fetched_at=time.time(),
        cleaned=cleaned,
        source_tool=source_tool,
    )
    try:
        _entry_path(url).write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("page_store.put failed for %s: %s", url, e)
    return entry


def get(url: str) -> PageEntry | None:
    """Retrieve a stored page entry by URL, or None if not stored."""
    p = _entry_path(url)
    if not p.exists():
        return None
    try:
        return PageEntry.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("page_store.get corrupt entry %s: %s", url, e)
        return None


def set_graph_node_key(url: str, key: str) -> None:
    """Update the graph_node_key backlink after memorise creates the graph node."""
    entry = get(url)
    if entry is None:
        return
    entry.graph_node_key = key
    try:
        _entry_path(url).write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("page_store.set_graph_node_key failed: %s", e)


def list_recent(n: int = 20) -> list[PageEntry]:
    """Return the n most recently fetched entries."""
    paths = sorted(
        _pages_dir().glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    entries = []
    for p in paths[:n]:
        try:
            entries.append(PageEntry.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return entries


def total_size_bytes() -> int:
    """Return total size of the page store on disk."""
    return sum(p.stat().st_size for p in _pages_dir().glob("*.json") if p.exists())
