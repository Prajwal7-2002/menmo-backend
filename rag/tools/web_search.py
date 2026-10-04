# rag/tools/web_search.py
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")


def web_search_tool(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    DuckDuckGo search via the `ddgs` package.
    Returns [{title, body, href}] or [] on failure (failures are logged).
    """
    if not WEB_SEARCH_ENABLED or not query.strip():
        return []
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("web search unavailable: the `ddgs` package is not installed")
        return []
    try:
        results = list(DDGS().text(query, max_results=max_results) or [])
    except Exception as e:
        logger.warning("web search failed for %r: %s", query[:80], e)
        return []

    out = []
    for r in results:
        body = r.get("body") or r.get("snippet") or ""
        title = r.get("title") or ""
        if body or title:
            out.append({"title": title, "body": body, "href": r.get("href") or r.get("url") or ""})
    return out


def web_search_chunks(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """Web results in the same chunk shape the rest of the pipeline uses."""
    return [
        {
            "id": None,
            "text": f"{r['title']}\n{r['body']}".strip(),
            "score": 0.0,
            "meta": {"source": r["href"], "title": r["title"], "type": "web"},
        }
        for r in web_search_tool(query, max_results=max_results)
    ]
