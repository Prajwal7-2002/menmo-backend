# rag/tools/query_rewrite.py
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from rag.llm import call_llm_answer

def rewrite_query_tool(query: str, context_chunks: Optional[List[Dict[str, Any]]] = None, max_tokens: int = 48) -> str:
    """
    LLM-driven lightweight query rewrite.
    Returns rewritten query (or original query on failure).
    context_chunks: list of chunk dicts (optional) — used to give LLM a little context to rewrite better.
    """
    try:
        context = ""
        if context_chunks:
            # use up to 2 chunk texts as context
            context = "\n\n".join(c.get("text", "") for c in context_chunks[:2])

        prompt = (
            "Rewrite the user's query to improve document retrieval while keeping the same meaning. "
            "Return only the rewritten concise query (no extra commentary).\n\n"
            f"Original query:\n{query}\n\n"
        )
        if context:
            prompt += f"Context from documents:\n{context}\n\n"

        out = call_llm_answer(question=prompt, context="", mood="serious", max_tokens=max_tokens)
        if out and isinstance(out, str) and out.strip():
            return out.strip()
        return query
    except Exception as e:
        logger.exception("rewrite_query_tool failed: %s", e)
        return query
