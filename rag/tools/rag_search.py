import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Import local retrieval functions (adjust path if needed)
try:
    from rag import retrieval
except Exception:
    from . import retrieval  # type: ignore


def rag_search(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    max_chunks: int = 4,
) -> Dict[str, Any]:
    """
    Corrected RAG search.

    New behaviour:
    ---------------------------------------------------------
    • If document_id is explicitly provided → STRICT retrieval ONLY.
      No fallback to other documents. No relaxed mode. No leakage.
    • If no document_id is provided → RAG behaves normally with relaxed fallback.
    ---------------------------------------------------------

    Returns:
      {
        "found": bool,
        "confidence": float,
        "chunks": [...],
        "context": "...",
        "filtered_match": True|False
      }
    """

    try:
        # internal utility
        def _use_with_conf(q, u, dom, doc, top_k):
            if hasattr(retrieval, "retrieve_with_conf"):
                return retrieval.retrieve_with_conf(
                    q, user_id=u, domain=dom, document_id=doc, top_k=top_k
                )
            chunks = retrieval.retrieve(
                q, user_id=u, domain=dom, document_id=doc, top_k=top_k
            )
            scores = [c.get("score", 0.0) for c in chunks]
            conf = max(scores) if scores else 0.0
            return chunks, scores, conf

        # -----------------------------
        # 1. STRICT RETRIEVAL
        # -----------------------------
        chunks, scores, conf = _use_with_conf(
            query, user_id, domain, document_id, max_chunks * 2
        )

        if chunks:  # strict hit
            chosen = chunks[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            return {
                "found": True,
                "confidence": float(conf),
                "chunks": chosen,
                "context": context,
                "filtered_match": True,
            }

        # -------------------------------------------------------------------
        # 2. STRICT MISS → IF document_id WAS PROVIDED, DO NOT RELAX
        # -------------------------------------------------------------------
        if document_id:
            # HARD STOP — do NOT leak into other documents
            logger.info(
                f"rag_search: strict retrieval for document_id={document_id} returned 0 results → NO RELAXED FALLBACK"
            )
            return {
                "found": False,
                "confidence": 0.0,
                "chunks": [],
                "context": "",
                "filtered_match": True,  # means: yes, filtering was applied
            }

        # -------------------------------------------------------------------
        # 3. domain provided but NO document_id → allow relaxed fallback
        # -------------------------------------------------------------------
        if domain:
            logger.info(
                "rag_search: strict domain filter returned no chunks; applying relaxed fallback"
            )
            relaxed_chunks, relaxed_scores, relaxed_conf = _use_with_conf(
                query, user_id, None, None, max_chunks * 2
            )

            if relaxed_chunks:
                chosen = relaxed_chunks[:max_chunks]
                context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
                return {
                    "found": True,
                    "confidence": float(relaxed_conf),
                    "chunks": chosen,
                    "context": context,
                    "filtered_match": False,  # indicates fallback
                }

            return {
                "found": False,
                "confidence": 0.0,
                "chunks": [],
                "context": "",
                "filtered_match": False,
            }

        # -------------------------------------------------------------------
        # 4. No filters → global relaxed search across all documents
        # -------------------------------------------------------------------
        relaxed_chunks, relaxed_scores, relaxed_conf = _use_with_conf(
            query, user_id, None, None, max_chunks * 2
        )
        if relaxed_chunks:
            chosen = relaxed_chunks[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            return {
                "found": True,
                "confidence": float(relaxed_conf),
                "chunks": chosen,
                "context": context,
                "filtered_match": False,
            }

        # nothing
        return {
            "found": False,
            "confidence": 0.0,
            "chunks": [],
            "context": "",
            "filtered_match": False,
        }

    except Exception as e:
        logger.exception("rag_search failed: %s", e)
        return {
            "found": False,
            "confidence": 0.0,
            "chunks": [],
            "context": "",
            "filtered_match": False,
        }
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Import local retrieval functions (adjust path if needed)
try:
    from rag import retrieval
except Exception:
    from . import retrieval  # type: ignore


def rag_search(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    max_chunks: int = 4,
) -> Dict[str, Any]:
    """
    Simplified, non-heuristic RAG search.

    Rules:
      - If document_id is provided → STRICT retrieval for that document only.
        No relaxed fallback to other documents.
      - Else if domain is provided → STRICT retrieval for that domain only.
        No relaxed fallback to other domains.
      - Else → global retrieval across all documents.

    The agent (LLM) decides what to do with a strict miss (e.g. try web,
    look in memory, or fall back to "I don't know").
    """

    try:
        def _use_with_conf(q, u, dom, doc, top_k):
            if hasattr(retrieval, "retrieve_with_conf"):
                return retrieval.retrieve_with_conf(
                    q, user_id=u, domain=dom, document_id=doc, top_k=top_k
                )
            chunks = retrieval.retrieve(
                q, user_id=u, domain=dom, document_id=doc, top_k=top_k
            )
            scores = [c.get("score", 0.0) for c in chunks]
            conf = max(scores) if scores else 0.0
            return chunks, scores, conf

        # -----------------------------
        # 1. Determine filter scope
        # -----------------------------
        if document_id:
            # Strict document scope: only this document.
            scope_domain = domain
            scope_doc = document_id
        elif domain:
            # Strict domain scope: only this domain.
            scope_domain = domain
            scope_doc = None
        else:
            # Global scope: all documents.
            scope_domain = None
            scope_doc = None

        chunks, scores, conf = _use_with_conf(
            query, user_id, scope_domain, scope_doc, max_chunks * 2
        )

        if chunks:
            chosen = chunks[:max_chunks]
            context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
            return {
                "found": True,
                "confidence": float(conf),
                "chunks": chosen,
                "context": context,
                "filtered_match": True,
            }

        # Strict miss in the selected scope. Do not relax here; let the agent
        # decide whether to use web or other tools.
        logger.info(
            "rag_search: strict retrieval miss (domain=%s, document_id=%s)",
            scope_domain,
            scope_doc,
        )
        return {
            "found": False,
            "confidence": 0.0,
            "chunks": [],
            "context": "",
            "filtered_match": True,
        }

    except Exception as e:
        logger.exception("rag_search failed: %s", e)
        return {
            "found": False,
            "confidence": 0.0,
            "chunks": [],
            "context": "",
            "filtered_match": False,
        }
