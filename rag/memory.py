# rag/memory.py
"""
Long-term memory stored in a separate Pinecone index.

Two kinds of records, both scoped by user:
  - type="conversation": past question/answer turns (tagged with the
    conversation and, when relevant, the document they were about, so they
    can be cleaned up when either is deleted).
  - type="memory_fact": things the user explicitly asked us to remember.

Short-term context (the last few messages of the current chat) comes from
the database, not from here; see api_app.views.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from . import vectorstore
from .embeddings import EMBED_DIM, EmbeddingError, embed_query

logger = logging.getLogger(__name__)

MEMORY_INDEX = vectorstore.MEMORY_INDEX_NAME
CONVERSATION_MIN_SCORE = 0.45
FACT_MIN_SCORE = 0.30
FACT_DUPLICATE_SCORE = 0.92


def _upsert(record_id: str, text: str, metadata: Dict[str, Any]) -> bool:
    try:
        vec = embed_query(text)
        vectorstore.upsert(MEMORY_INDEX, [{"id": record_id, "values": vec,
                                           "metadata": {**metadata, "text": text[:8000]}}])
        return True
    except (EmbeddingError, vectorstore.VectorStoreError) as e:
        logger.error("memory upsert failed: %s", e)
        return False


def _search(user, query: str, kind: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
    try:
        vec = embed_query(query)
        matches = vectorstore.query(MEMORY_INDEX, vec, top_k=top_k,
                                    flt={"type": kind, "user": str(user.id)})
    except (EmbeddingError, vectorstore.VectorStoreError) as e:
        logger.error("memory search failed: %s", e)
        return []
    return [
        {"id": m["id"], "text": m["metadata"].get("text", ""), "similarity": m["score"],
         "score": m["score"], "meta": {"type": kind}}
        for m in matches
        if m["score"] >= min_score and m["metadata"].get("text")
    ]


# ------------------------------- conversation -------------------------------

def store_conversation_turn(user, query: str, answer: str,
                            conversation_id: Optional[str] = None,
                            document_id: Optional[str] = None) -> bool:
    if not user or not getattr(user, "id", None) or not query:
        return False
    meta: Dict[str, Any] = {"type": "conversation", "user": str(user.id)}
    if conversation_id:
        meta["conversation_id"] = str(conversation_id)
    if document_id:
        meta["document_id"] = str(document_id)
    return _upsert(f"conv-{user.id}-{uuid.uuid4().hex}",
                   f"user: {query}\nassistant: {answer}", meta)


def search_conversation_memory(user, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    return _search(user, query, "conversation", top_k, CONVERSATION_MIN_SCORE)


# ----------------------------------- facts ----------------------------------

def add_memory_fact(user, text: str) -> str:
    """Store a fact. Returns "stored", "duplicate", "empty" or "failed"."""
    text = (text or "").strip()
    if not text:
        return "empty"
    if _search(user, text, "memory_fact", 3, FACT_DUPLICATE_SCORE):
        return "duplicate"
    ok = _upsert(f"fact-{user.id}-{uuid.uuid4().hex}", text,
                 {"type": "memory_fact", "user": str(user.id)})
    return "stored" if ok else "failed"


def search_memory_facts(user, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    return _search(user, query, "memory_fact", top_k, FACT_MIN_SCORE)


# ---------------------------------- cleanup ---------------------------------

def _clear(flt: Dict[str, Any]) -> int:
    try:
        n = vectorstore.delete_by_filter(MEMORY_INDEX, flt, dim=EMBED_DIM)
        logger.info("Cleared %d memory vectors for %s", n, flt)
        return n
    except Exception as e:
        logger.error("memory cleanup failed for %s: %s", flt, e)
        return 0


def clear_memory_for_document(user_id: int, document_id: str) -> int:
    return _clear({"user": str(user_id), "document_id": str(document_id)})


def clear_memory_for_conversation(user_id: int, conversation_id: str) -> int:
    return _clear({"user": str(user_id), "conversation_id": str(conversation_id)})
