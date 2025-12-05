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

    Behavior:
      1) Attempt strict retrieval with the provided domain/document_id (if any).
      2) If strict retrieval returns nothing but domain/document_id were provided,
         run a relaxed (unfiltered) retrieval as a fallback and return those results
         with filtered_match=False so callers can decide how to handle them.
      3) Otherwise behave as before.

    Returns a dict:
      {
        "found": bool,
        "confidence": float,
        "chunks": [...],
        "context": "...",
        "filtered_match": True|False  # True when results matched the provided filters
      }
    """
    try:
        # Prefer retrieve_with_conf when available (returns chunks, scores, conf)
        def _use_with_conf(q, u, dom, doc, top_k):
            if hasattr(retrieval, "retrieve_with_conf"):
                return retrieval.retrieve_with_conf(q, user_id=u, domain=dom, document_id=doc, top_k=top_k)
            # fallback to old retrieve()
            chunks = retrieval.retrieve(q, user_id=u, domain=dom, document_id=doc, top_k=top_k)
            scores = [c.get("score", 0.0) for c in chunks]
            conf = max(scores) if scores else 0.0
            return chunks, scores, conf

        # 1) Strict retrieval using provided filters
        chunks, scores, conf = _use_with_conf(query, user_id, domain, document_id, max_chunks * 2)

        if chunks:
            chosen = chunks[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            return {"found": True, "confidence": float(conf), "chunks": chosen, "context": context, "filtered_match": True}

        # 2) If strict returned nothing but caller requested domain/document, try relaxed fallback
        if domain or document_id:
            logger.info("rag_search: strict filtered retrieval returned no chunks; trying relaxed (unfiltered) fallback")
            try:
                relaxed_chunks, relaxed_scores, relaxed_conf = _use_with_conf(query, user_id, None, None, max_chunks * 2)
                if relaxed_chunks:
                    chosen = relaxed_chunks[:max_chunks]
                    context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
                    # signal that these did NOT strictly match filters
                    return {
                        "found": True,
                        "confidence": float(relaxed_conf),
                        "chunks": chosen,
                        "context": context,
                        "filtered_match": False,
                    }
            except Exception as e:
                logger.exception("rag_search: relaxed fallback failed: %s", e)

        # 3) Nothing found
        return {"found": False, "confidence": 0.0, "chunks": [], "context": "", "filtered_match": False}

    except Exception as e:
        logger.exception("rag_search failed: %s", e)
        return {"found": False, "confidence": 0.0, "chunks": [], "context": "", "filtered_match": False}
