import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def web_search_tool(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    DuckDuckGo wrapper (ddgs / duckduckgo_search compatible).
    Returns list of dicts {title, body, href} or [] on failure.
    """
    try:
        # prefer ddgs if available
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception:
            # fallback to duckduckgo_search
            from duckduckgo_search import ddg_answers
            res = ddg_answers(query, related=False)
            results = res or []

        output = []
        for r in results:
            title = r.get("title") or r.get("text") or ""
            body = r.get("body") or r.get("snippet") or r.get("text") or ""
            href = r.get("href") or r.get("url") or r.get("link") or ""
            output.append({"title": title, "body": body, "href": href})
        return output
    except Exception as e:
        logger.debug("web_search_tool failed: %s", e)
        return []
