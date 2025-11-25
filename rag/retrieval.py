# rag/retrieval.py
import os
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from .models import ChunkFeedback
from .llm import call_llm_answer   # <-- for query rewriting

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))

# ---------------------- Config ----------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# threshold: you can lower/raise with env var RAG_MIN_SCORE_THRESHOLD
MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE_THRESHOLD", "0.35"))
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "8"))

# auto-relax: if dense scores are negative, this fallback_threshold will be used
AUTO_RELAX_THRESHOLD = float(os.getenv("RAG_AUTO_RELAX_THRESHOLD", "-0.5"))

_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None

# ---------------------- Helpers ----------------------


def _get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)
    # try to log index stats (best-effort)
    try:
        stats = idx.describe_index_stats()
        logger.debug("Pinecone index stats: %s", stats)
    except Exception:
        logger.debug("Could not describe pinecone index stats (non-fatal).")
    return idx


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L6-v2")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _reranker


# ---------------------- Embeddings ----------------------


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of embeddings. If a text is empty it is skipped and an empty embedding returned."""
    if not texts:
        return []
    model = _get_model()
    # Protective: replace empty strings with a small placeholder so model doesn't crash
    safe_texts = [t if (t and t.strip()) else " " for t in texts]
    embs = model.encode(safe_texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
    return [emb.tolist() for emb in embs]


# ---------------------- Upsert Vectors ----------------------

def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100) -> None:
    if not items:
        logger.debug("upsert_vectors called with no items")
        return
    index = _get_pinecone_index()
    for i in range(0, len(items), batch_size):
        slice_ = items[i:i + batch_size]
        try:
            index.upsert(vectors=slice_)
        except Exception as e:
            logger.exception("Pinecone upsert failed for batch starting at %d: %s", i, e)
            raise


# ---------------------- Query Vectors ----------------------


def query_vectors(
    query_text: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    if not query_text or not query_text.strip():
        logger.debug("query_vectors called with empty query_text")
        return []

    index = _get_pinecone_index()

    embs = embed_texts([query_text])
    if not embs:
        logger.debug("Embedding returned empty for query '%s'", query_text)
        return []

    q_vec = embs[0]

    flt: Dict[str, Any] = {}
    if user_id:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = domain
    if document_id:
        flt["document_id"] = document_id

    try:
        res = index.query(
            vector=q_vec,
            top_k=top_k,
            filter=flt or None,
            include_metadata=True,
        )
    except Exception as e:
        logger.exception("Pinecone query failed: %s", e)
        # Try again without filters (best-effort fallback)
        try:
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
        except Exception as e2:
            logger.exception("Pinecone query fallback failed: %s", e2)
            return []

    matches = []
    for m in res.get("matches", []):
        meta = m.get("metadata") or {}
        # support older metadata key names
        text = meta.get("chunk_text") or meta.get("snippet") or meta.get("text") or ""
        matches.append({
            "id": m.get("id"),
            "text": text,
            "score": float(m.get("score", 0.0)),
            "meta": meta,
        })

    logger.debug("query_vectors returned %d matches (domain filter=%s)", len(matches), domain)
    return matches


# ---------------------- Hybrid Retrieval (Upgraded) ----------------------


def retrieve(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    logger.info("retrieve() query=%s domain=%s document_id=%s", (query[:80] + "...") if query else "", domain, document_id)

    # ========================
    # 1) Query Rewriting (best-effort)
    # ========================
    try:
        rewritten = call_llm_answer(
            question=f"Rewrite this query to improve document search accuracy: {query}",
            context="",
            mood="serious",
            max_tokens=48
        )
        effective_query = rewritten if (rewritten and len(rewritten) < 400) else query
    except Exception:
        effective_query = query

    logger.debug("effective_query=%s", effective_query)

    # ========================
    # 2) Dense retrieval (may return many)
    # ========================
    dense_matches = query_vectors(
        query_text=effective_query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max(top_k * 2, 10),
    )

    if not dense_matches:
        logger.info("No dense matches found. Returning empty list.")
        return []

    # If domain was not provided, try to auto-detect the most common domain among matches
    if domain is None:
        domains = [m["meta"].get("domain") for m in dense_matches if m["meta"].get("domain")]
        if domains:
            inferred = max(set(domains), key=domains.count)
            logger.debug("Inferred domain=%s from dense matches", inferred)
            domain = inferred

    # Attempt strict domain filtering first; if nothing left, relax and keep original dense_matches
    filtered = [m for m in dense_matches if domain is None or m["meta"].get("domain") == domain]
    if not filtered:
        logger.debug("No matches after domain filter; relaxing domain filter.")
        filtered = dense_matches

    dense_matches = filtered

    # ========================
    # 3) BM25 Sparse Scoring
    # ========================
    corpus_tokens = [m["text"].split() for m in dense_matches]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(effective_query.split())
    max_bm25 = max(bm25_scores) if len(bm25_scores) else 1.0

    # ========================
    # 4) Feedback lookup
    # ========================
    pinecone_ids = [m["id"] for m in dense_matches]
    feedback_qs = ChunkFeedback.objects.filter(pinecone_id__in=pinecone_ids)
    feedback_map = {fb.pinecone_id: fb for fb in feedback_qs}

    # ========================
    # 5) Cross-Encoder Reranking (best-effort)
    # ========================
    try:
        reranker = _get_reranker()
        passage_pairs = [(effective_query, m["text"]) for m in dense_matches]
        rerank_scores_raw = reranker.predict(passage_pairs)
        # convert to float list
        rerank_scores = [float(x) for x in rerank_scores_raw]
    except Exception as e:
        logger.exception("Reranker failed: %s", e)
        rerank_scores = [0.0] * len(dense_matches)

    # Ensure lengths align
    if len(rerank_scores) != len(dense_matches):
        logger.warning("Reranker returned %d scores but there are %d dense matches; adjusting.", len(rerank_scores), len(dense_matches))
        # fallback to neutral scores
        rerank_scores = [0.0] * len(dense_matches)

    # normalize reranker to 0..1 (per-query min-max)
    try:
        rmin = min(rerank_scores)
        rmax = max(rerank_scores)
        rspan = max(1e-6, (rmax - rmin))
        rerank_norm = [(r - rmin) / rspan for r in rerank_scores]
    except Exception:
        rerank_norm = [0.5] * len(rerank_scores)  # neutral fallback

    # ========================
    # 6) Hybrid Score (IMPROVED + normalized)
    # ========================

    # normalize dense_score across results to bring into 0..1 scale (best-effort)
    raw_dense = [float(m["score"]) for m in dense_matches]
    max_dense = max(raw_dense) if raw_dense else 1.0
    min_dense = min(raw_dense) if raw_dense else 0.0
    span_dense = max(1e-6, (max_dense - min_dense))

    combined = []
    for idx, (m, sparse, rerank_val) in enumerate(zip(dense_matches, bm25_scores, rerank_norm)):
        raw_ds = float(m["score"])
        # normalized dense in 0..1
        dense_score_norm = (raw_ds - min_dense) / span_dense if span_dense else 0.0

        # Safe normalize sparse
        try:
            sparse_val = float(sparse)
            if sparse_val != sparse_val:
                sparse_val = 0.0
        except Exception:
            sparse_val = 0.0
        sparse_score = sparse_val / (max_bm25 or 1.0)

        fb = feedback_map.get(m["id"])
        fb_score = float(fb.score) if fb else 0.0

        # Weighted hybrid (use normalized rerank)
        # Note: rerank_norm is 0..1 so weight chosen lower to not overpower dense/bm25
        hybrid = (
            0.45 * dense_score_norm +
            0.30 * sparse_score +
            0.15 * rerank_val +   # normalized reranker contribution
            0.10 * fb_score
        )

        # NaN guard
        if hybrid != hybrid:
            hybrid = 0.0

        combined.append({
            "id": m["id"],
            "text": m["text"],
            "meta": m["meta"],
            "score": hybrid,
            "dense_score_raw": raw_ds,
            "dense_score_norm": dense_score_norm,
            "bm25_score": sparse_score,
            "rerank_score": float(rerank_scores[idx]) if idx < len(rerank_scores) else 0.0,
            "rerank_norm": rerank_val,
            "feedback_score": fb_score,
        })

    # sort by combined hybrid score
    combined.sort(key=lambda x: x["score"], reverse=True)

    top_k_results = combined[:top_k]

    logger.debug("retrieve returning %d items. top score=%s", len(top_k_results), (top_k_results[0]["score"] if top_k_results else None))

    # If top raw dense score is very negative, log a warning (embedding/index mismatch suspect)
    if raw_dense and max_dense < AUTO_RELAX_THRESHOLD:
        logger.warning("Max dense score (raw) is %s which is below AUTO_RELAX_THRESHOLD=%s. Suspect embedding/index mismatch or bad text extraction.", max_dense, AUTO_RELAX_THRESHOLD)

    return top_k_results
