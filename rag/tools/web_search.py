# rag/tools/web_search.py
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def web_search_tool(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Simple DuckDuckGo (DDGS) wrapper that returns a list of result dicts:
    [{'title':..., 'body':..., 'href':...}, ...]
    If DDGS is unavailable it returns [].
    """
    try:
        # import inside function to avoid hard dependency in environments without ddgs
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        # Normalize results: ddgs returns dict with title/body/href in most cases
        out = []
        for r in results:
            title = r.get("title") or r.get("text") or ""
            body = r.get("body") or r.get("text") or ""
            href = r.get("href") or r.get("url") or ""
            out.append({"title": title, "body": body, "href": href})
        return out
    except Exception as e:
        logger.debug("web_search_tool failed (DDGS may be missing): %s", e)
        return []
