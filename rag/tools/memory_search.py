# rag/tools/memory_search.py
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def memory_search_tool(user, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Returns scored memory entries using your existing memory.relevant_chunks().
    Returns [] on failure.
    Each item: {'text':..., 'score':..., 'meta':...}
    """
    try:
        from rag import memory
        if hasattr(memory, "relevant_chunks"):
            return memory.relevant_chunks(user, query, top_k=top_k)
        # fallback to load_vector_memory if present (but that's chat-style)
        if hasattr(memory, "load_vector_memory"):
            msgs = memory.load_vector_memory(user, query, top_k=top_k)
            # convert to simple scored items with low default score
            return [{"text": m.get("content", ""), "score": 0.0, "meta": {}} for m in msgs]
        return []
    except Exception as e:
        logger.exception("memory_search_tool failed: %s", e)
        return []
