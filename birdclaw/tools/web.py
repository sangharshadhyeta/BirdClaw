"""Web tools — fetch URLs and search via SearXNG."""

from __future__ import annotations

import json
import logging

import httpx
from bs4 import BeautifulSoup

from birdclaw.config import settings
from birdclaw.tools.registry import Tool, registry

logger = logging.getLogger(__name__)


def web_fetch(url: str, max_chars: int | None = None) -> str:
    """Fetch a URL and return a condensed text snippet for immediate context.

    HTML is stripped via BeautifulSoup. The fast-path snippet (~1200 chars)
    is returned immediately. An async LLM cleaning job is queued in the
    background — the full condensed result lands in the page store and is
    surfaced as a pending note at the start of the next agent step.

    Args:
        url:       URL to fetch.
        max_chars: Unused (kept for backwards compat — condenser caps output).

    Returns:
        JSON string with {"content": "...", "note": "LLM cleaning in progress"}
        or {"error": "..."}.
    """
    logger.info("[web] fetch  url=%s", url[:100])
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("[web] fetch error  url=%s  error=%s", url[:80], e)
        return json.dumps({"error": str(e)})

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return json.dumps({"error": f"unsupported content type: {content_type}"})

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    body = soup.find("article") or soup.find("main") or soup.find("body")
    raw_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

    # Keyword-prune the raw text immediately using the URL as a proxy goal —
    # removes blank lines, repeated nav text, and cookie notices before handing
    # to condenser. semantic_prune is used if the page is large and noisy.
    from birdclaw.llm.pruner import keyword_prune, semantic_prune
    _goal_hint = url.split("/")[-1].replace("-", " ").replace("_", " ")
    if len(raw_text) > 3000:
        raw_text = semantic_prune(raw_text, goal=_goal_hint, max_chars=2000)
    else:
        raw_text = keyword_prune(raw_text, goal=_goal_hint, max_chars=2000)

    from birdclaw.tools.condenser import condense_async
    snippet = condense_async(raw_text, url, source_tool="web_fetch")
    logger.info("[web] fetch ok  url=%s  raw=%d  snippet=%d", url[:80], len(raw_text), len(snippet))

    return json.dumps({
        "content": snippet,
        "url": url,
        "note": "Full condensed notes will be available next step.",
    })


def web_search(query: str, n: int = 5) -> str:
    """Search the web and return top results.

    Tries SearXNG first (requires local SearXNG instance configured via
    BC_SEARXNG_URL). Falls back to DuckDuckGo Instant Answers when SearXNG
    is unavailable — no API key or setup required.

    Args:
        query: Search query.
        n:     Number of results to return (default 5).

    Returns:
        JSON string with {"results": [{"title", "url", "content"}, ...]} or {"error": "..."}.
    """
    logger.info("[web] search  query=%r  n=%d", query[:60], n)
    # --- Try SearXNG first ---
    try:
        resp = httpx.get(
            f"{settings.searxng_url}/search",
            params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", [])[:n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
            })
        if results:
            logger.info("[web] search ok via SearXNG  query=%r  results=%d", query[:40], len(results))
            from birdclaw.tools.condenser import condense_async
            combined = "\n\n".join(
                f"[{r['title']}] {r['url']}\n{r['content']}" for r in results
            )
            condense_async(combined, url=f"search:{query}", source_tool="web_search")
            return json.dumps({"results": results, "source": "searxng"})
    except Exception as _searxng_err:
        logger.debug("[web] SearXNG miss  query=%r  err=%s", query[:40], _searxng_err)
        pass  # fall through to DuckDuckGo

    # --- Fallback: DuckDuckGo Instant Answers ---
    # Free, no auth, no setup. Returns AbstractText + RelatedTopics snippets.
    # Replace with SearXNG by installing it locally (BC_SEARXNG_URL=http://localhost:8888).
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=15,
            headers={"User-Agent": "BirdClaw/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"search unavailable: {e}"})
    except Exception as e:
        return json.dumps({"error": f"search error: {e}"})

    results = []

    # AbstractText is the main instant answer (Wikipedia summary etc.)
    abstract = data.get("AbstractText", "")
    abstract_url = data.get("AbstractURL", "")
    if abstract:
        results.append({
            "title": data.get("Heading", query),
            "url": abstract_url,
            "content": abstract[:500],
        })

    # RelatedTopics gives additional snippets
    for topic in data.get("RelatedTopics", [])[:n]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "url": topic.get("FirstURL", ""),
                "content": topic.get("Text", "")[:500],
            })
        if len(results) >= n:
            break

    if not results:
        return json.dumps({"error": "no results found", "note": "Install SearXNG for full web search"})

    # Condense search snippets as a single block for page store
    from birdclaw.tools.condenser import condense_async
    combined = "\n\n".join(
        f"[{r['title']}] {r['url']}\n{r['content']}" for r in results
    )
    condense_async(combined, url=f"search:{query}", source_tool="web_search")

    return json.dumps({"results": results, "source": "duckduckgo_instant"})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(Tool(
    name="web_fetch",
    description="Fetch a URL and return its text content (HTML stripped, article/main preferred).",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
            "max_chars": {"type": "integer", "description": "Max characters to return."},
        },
        "required": ["url"],
    },
    handler=web_fetch,
    tags=["fetch", "url", "web", "http", "browse", "download", "page", "link", "read"],
))

registry.register(Tool(
    name="web_search",
    description="Search the web via SearXNG. Returns title, URL, and snippet for top results.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "n": {"type": "integer", "description": "Number of results (default 5)."},
        },
        "required": ["query"],
    },
    handler=web_search,
    tags=[
        "search", "web", "google", "internet", "lookup", "find",
        "who", "what", "when", "where", "why", "how",
        "latest", "news", "docs", "documentation",
    ],
))
