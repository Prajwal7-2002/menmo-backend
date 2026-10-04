# rag/retrieval.py
"""
Hybrid document retrieval over Pinecone.

Every chunk returned carries two numbers:
  - "similarity": raw cosine similarity between query and chunk. This is an
    absolute measure and is what relevance decisions are based on.
  - "score": hybrid ranking score (dense + BM25 + optional reranker +
    feedback), used only to order results.

The old code min-max normalised dense scores inside each result set, so the
best chunk always looked like a perfect match even when nothing was relevant.
"""
import logging
import math
import os
import re
import threading
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from . import vectorstore
from .embeddings import EMBED_DIM, EmbeddingError, embed_query

logger = logging.getLogger(__name__)

TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "8"))

DENSE_WEIGHT = float(os.getenv("RAG_WEIGHT_DENSE", "0.60"))
BM25_WEIGHT = float(os.getenv("RAG_WEIGHT_BM25", "0.25"))
RR_WEIGHT = float(os.getenv("RAG_WEIGHT_RERANK", "0.15"))
FB_WEIGHT = float(os.getenv("RAG_WEIGHT_FEEDBACK", "0.05"))

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "40"))
# When a domain filter finds nothing, retry across all of the user's documents.
ALLOW_RELAXED_FALLBACK = os.getenv("ALLOW_RELAXED_FALLBACK", "true").lower() in ("1", "true", "yes")
# Opt-in: let every user search every document in the index (old behaviour).
SHARED_DOCUMENTS = os.getenv("RAG_SHARED_DOCUMENTS", "false").lower() in ("1", "true", "yes")

_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _clean(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text or "")
    return re.sub(r"\s{2,}", " ", text).strip()


def build_filter(user_id: Optional[int], domain: Optional[str] = None,
                 document_id: Optional[str] = None) -> Dict[str, Any]:
    flt: Dict[str, Any] = {}
    if not SHARED_DOCUMENTS:
        flt["user_id"] = str(user_id)
    if document_id:
        flt["document_id"] = str(document_id)
    elif domain:
        flt["domain"] = str(domain).lower()
    return flt


def _dense_candidates(q_vec: List[float], flt: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    out = []
    for m in vectorstore.query(vectorstore.DOC_INDEX_NAME, q_vec, top_k=top_k, flt=flt):
        meta = m["metadata"]
        text = _clean(meta.get("chunk_text") or meta.get("full_text") or meta.get("snippet") or "")
        out.append({"id": m["id"], "text": text, "similarity": m["score"], "meta": meta})
    return out


def retrieve(query: str, user_id: Optional[int], domain: Optional[str] = None,
             document_id: Optional[str] = None, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Return up to top_k chunks ranked by hybrid score.

    Scope: always the user's own documents (unless RAG_SHARED_DOCUMENTS),
    narrowed to `document_id` if given, else to `domain` if given. A
    document-scoped search never widens; a domain-scoped one may widen to all
    of the user's documents when ALLOW_RELAXED_FALLBACK is on.
    """
    if not query or not query.strip():
        return []
    if user_id is None and not SHARED_DOCUMENTS:
        logger.warning("retrieve() called without user_id; refusing to search")
        return []

    try:
        q_vec = embed_query(query)
    except EmbeddingError as e:
        logger.error("retrieve(): %s", e)
        return []

    n_candidates = max(10, top_k * 2)
    try:
        dense = _dense_candidates(q_vec, build_filter(user_id, domain, document_id), n_candidates)
        if not dense and domain and not document_id and ALLOW_RELAXED_FALLBACK:
            logger.info("retrieve(): no hits in domain %r, widening to all user documents", domain)
            dense = _dense_candidates(q_vec, build_filter(user_id), n_candidates)
    except vectorstore.VectorStoreError as e:
        logger.error("retrieve(): %s", e)
        return []

    dense = [d for d in dense if d["text"]]
    long_enough = [d for d in dense if len(d["text"]) >= MIN_TEXT_LEN]
    dense = long_enough or dense
    if not dense:
        return []

    # Lexical signal on the candidate set
    q_tokens = _tokens(query)
    try:
        bm25_scores = list(BM25Okapi([_tokens(d["text"]) or [""] for d in dense]).get_scores(q_tokens))
    except Exception as e:
        logger.warning("retrieve(): BM25 failed: %s", e)
        bm25_scores = [0.0] * len(dense)
    max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

    # Optional cross-encoder
    rr_norm = [0.0] * len(dense)
    if USE_RERANKER:
        try:
            rr = [float(x) for x in _get_reranker().predict([(query, d["text"]) for d in dense])]
            rr_norm = [1.0 / (1.0 + math.exp(-x)) for x in rr]  # logits -> 0..1
        except Exception as e:
            logger.warning("retrieve(): reranker failed: %s", e)

    # Feedback from thumbs up/down
    fb_map: Dict[str, float] = {}
    try:
        from .models import ChunkFeedback
        fb_map = {fb.pinecone_id: fb.score for fb in
                  ChunkFeedback.objects.filter(pinecone_id__in=[d["id"] for d in dense])}
    except Exception as e:
        logger.warning("retrieve(): failed to load ChunkFeedback: %s", e)

    combined, seen = [], set()
    for idx, d in enumerate(dense):
        key = " ".join(d["text"].split()).lower()[:300]
        if key in seen:
            continue
        seen.add(key)
        bm25_norm = bm25_scores[idx] / max_bm25
        fb = math.tanh(fb_map.get(d["id"], 0.0) / 3.0)  # -1..1, saturates after a few votes
        score = (DENSE_WEIGHT * d["similarity"] + BM25_WEIGHT * bm25_norm
                 + RR_WEIGHT * rr_norm[idx] + FB_WEIGHT * fb)
        combined.append({
            **d,
            "score": round(score, 4),
            "bm25_score_norm": round(bm25_norm, 4),
            "rerank_norm": round(rr_norm[idx], 4),
            "feedback_score": fb_map.get(d["id"], 0.0),
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("retrieve(): top similarities %s", [round(c["similarity"], 3) for c in combined[:5]])
    return combined[:top_k]


def upsert_chunks(items: List[Dict[str, Any]]) -> None:
    vectorstore.upsert(vectorstore.DOC_INDEX_NAME, items)


def delete_chunks(ids: List[str]) -> int:
    return vectorstore.delete_ids(vectorstore.DOC_INDEX_NAME, ids)


def delete_document_vectors(user_id: int, document_id: str) -> int:
    """Fallback cleanup for vectors whose ids aren't mirrored in the DB."""
    return vectorstore.delete_by_filter(
        vectorstore.DOC_INDEX_NAME,
        {"user_id": str(user_id), "document_id": str(document_id)},
        dim=EMBED_DIM,
    )
