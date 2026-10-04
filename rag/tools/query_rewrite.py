# rag/tools/query_rewrite.py
from rag.llm import safe_completion


def _clean(out: str, original: str) -> str:
    out = (out or "").strip().strip('"').strip()
    return out if 1 < len(out) < 300 else original


def rewrite_query_tool(query: str) -> str:
    """Rephrase a query for better semantic retrieval, keeping its meaning."""
    out = safe_completion([
        {"role": "system", "content": (
            "Rewrite the search query to improve semantic document retrieval: expand abbreviations, "
            "add key synonyms, keep the same intent and scope. Return only the rewritten query.")},
        {"role": "user", "content": query},
    ], max_tokens=60)
    return _clean(out, query)
