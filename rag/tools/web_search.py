# rag/tools/web_search.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def web_search_tool(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Simple DuckDuckGo (DDGS) wrapper that returns a list of dictionaries:
    [
        {"title": "...", "body": "...", "href": "..."}
    ]

    If DDGS is not installed or fails, it returns [] safely.
    """
    try:
        from ddgs import DDGS  # import inside to avoid dependency issues

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        output = []
        for r in results:
            output.append({
                "title": r.get("title") or r.get("text") or "",
                "body": r.get("body") or r.get("text") or "",
                "href": r.get("href") or r.get("url") or "",
            })

        return output

    except Exception as e:
        logger.debug("web_search_tool failed: %s", e)
        return []
