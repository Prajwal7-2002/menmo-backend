# rag/tools/rag_search.py
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Import local retrieval functions (adjust path if needed)
try:
    from rag import retrieval
except Exception:
    # fallback relative import for some environments
    from . import retrieval  # type: ignore

def rag_search(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    max_chunks: int = 4,
) -> Dict[str, Any]:
    """
    Lightweight RAG search tool.
    Returns a dict:
      {
        "found": bool,
        "confidence": float,    # top candidate hybrid score
        "chunks": [ {...} ],    # list of chunk dicts (id,text,score,meta)
        "context": "..."        # joined context of top N chunks
      }
    This uses retrieval.retrieve_with_conf() when available for a fast preview,
    and falls back to retrieval.retrieve().
    """
    try:
        if hasattr(retrieval, "retrieve_with_conf"):
            chunks, scores, conf = retrieval.retrieve_with_conf(
                query, user_id=user_id, domain=domain, document_id=document_id, top_k=max_chunks * 2
            )
            if not chunks:
                return {"found": False, "confidence": 0.0, "chunks": [], "context": ""}
            chosen = chunks[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            return {"found": True, "confidence": float(conf), "chunks": chosen, "context": context}
        else:
            candidates = retrieval.retrieve(query, user_id=user_id, domain=domain, document_id=document_id, top_k=max_chunks*2)
            if not candidates:
                return {"found": False, "confidence": 0.0, "chunks": [], "context": ""}
            chosen = candidates[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            top_conf = float(chosen[0].get("score", 0.0)) if chosen else 0.0
            return {"found": True, "confidence": top_conf, "chunks": chosen, "context": context}
    except Exception as e:
        logger.exception("rag_search failed: %s", e)
        return {"found": False, "confidence": 0.0, "chunks": [], "context": ""}
