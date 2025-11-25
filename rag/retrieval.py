# rag/retrieval.py
import os
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
# Import CrossEncoder lazily (only used when enabled)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # pragma: no cover

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from .models import ChunkFeedback
from .llm import call_llm_answer   # <-- for optional query rewriting

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

# Feature toggles (safe defaults = off)
USE_RERANKER = str(os.getenv("USE_RERANKER", "false")).lower() in ("1", "true", "yes")
USE_QUERY_REWRITE = str(os.getenv("USE_QUERY_REWRITE", "false")).lower() in ("1", "true", "yes")

# Reranker model name (kept separate in case you want to override)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

_model: Optional[SentenceTransformer] = None
_reranker: Optional["CrossEncoder"] = None  # type: ignore[name-defined]

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
        logger.info(f"Loading embedding model from: {EMBED_MODEL_NAME}")

        # Load local model if EMBED_MODEL_NAME points to a folder
        if EMBED_MODEL_NAME.startswith("/") or os.path.isdir(EMBED_MODEL_NAME):
            logger.info("Detected local model folder → loading locally")
            _model = SentenceTransformer(EMBED_MODEL_NAME)
        else:
            # Otherwise load from hub
            logger.info("Loading model from HuggingFace Hub")
            _model = SentenceTransformer(EMBED_MODEL_NAME)

    return _model


def _get_reranker():
    global _reranker
    if not USE_RERANKER:
        raise RuntimeError("Reranker is disabled via USE_RERANKER environment toggle")
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder import failed; ensure sentence-transformers supports CrossEncoder on this platform")
    if _reranker is None:
        logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ---------------------- Embeddings ----------------------


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of embeddings. If a text is empty it is skipped and an empty embedding returned."""
    if not texts:
        return []
    model = _get_model()
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


def _extract_text_from_meta(meta: Dict[str, Any]) -> str:
    """Robustly get text from Pinecone metadata."""
    if not meta:
        return ""
    return meta.get("chunk_text") or meta.get("snippet") or meta.get("text") or meta.get("content") or ""


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
        text = _extract_text_from_meta(meta)
        # Pinecone may return different score fields depending on API/version
        raw_score = m.get("score", m.get("value", 0.0))
        try:
            raw_score = float(raw_score)
        except Exception:
            raw_score = 0.0
        matches.append({
            "id": m.get("id"),
            "text": text,
            "score": raw_score,
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

    # 1) Query Rewriting (optional)
    effective_query = query
    if USE_QUERY_REWRITE:
        try:
            rewritten = call_llm_answer(
                question=f"Rewrite this query to improve document search accuracy: {query}",
                context="",
                mood="serious",
                max_tokens=48
            )
            if rewritten and len(rewritten) < 400:
                effective_query = rewritten
        except Exception as e:
            logger.exception("Query rewrite failed: %s. Falling back to original query.", e)
            effective_query = query

    logger.debug("effective_query=%s", effective_query)

    # 2) Dense retrieval (may return many)
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

    # 3) BM25 Sparse Scoring
    corpus_texts = [m["text"] or "" for m in dense_matches]
    corpus_tokens = [txt.split() for txt in corpus_texts]
    try:
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(effective_query.split())
    except Exception as e:
        logger.exception("BM25 scoring failed: %s", e)
        bm25_scores = [0.0] * len(dense_matches)
    max_bm25 = max(bm25_scores) if len(bm25_scores) else 1.0

    # 4) Feedback lookup
    pinecone_ids = [m["id"] for m in dense_matches]
    feedback_qs = ChunkFeedback.objects.filter(pinecone_id__in=pinecone_ids)
    feedback_map = {fb.pinecone_id: fb for fb in feedback_qs}

    # 5) Cross-Encoder Reranking (best-effort & optional)
    rerank_scores = []
    if USE_RERANKER:
        try:
            reranker = _get_reranker()
            passage_pairs = [(effective_query, m["text"]) for m in dense_matches]
            rerank_raw = reranker.predict(passage_pairs)
            # convert to list of floats
            rerank_scores = [float(x) for x in rerank_raw]
        except Exception as e:
            logger.exception("Reranker failed: %s", e)
            rerank_scores = [0.0] * len(dense_matches)
    else:
        rerank_scores = [0.0] * len(dense_matches)

    # Align lengths
    if len(rerank_scores) != len(dense_matches):
        logger.warning("Reranker returned %d scores but there are %d dense matches; adjusting.", len(rerank_scores), len(dense_matches))
        rerank_scores = [0.0] * len(dense_matches)

    # Normalize reranker to 0..1 (per-query min-max) with safe fallback
    try:
        rmin = min(rerank_scores)
        rmax = max(rerank_scores)
        rspan = max(1e-6, (rmax - rmin))
        rerank_norm = [(r - rmin) / rspan for r in rerank_scores]
    except Exception:
        rerank_norm = [0.5] * len(rerank_scores)

    # 6) Hybrid Score (normalize dense + sparse + rerank)
    raw_dense = [float(m["score"]) for m in dense_matches]
    max_dense = max(raw_dense) if raw_dense else 1.0
    min_dense = min(raw_dense) if raw_dense else 0.0
    span_dense = max(1e-6, (max_dense - min_dense))

    combined = []
    for idx, (m, sparse_val, rerank_val) in enumerate(zip(dense_matches, bm25_scores, rerank_norm)):
        raw_ds = float(m["score"])
        dense_score_norm = (raw_ds - min_dense) / span_dense if span_dense else 0.0

        # safe normalize sparse (bm25)
        try:
            sparse_f = float(sparse_val)
            if sparse_f != sparse_f:
                sparse_f = 0.0
        except Exception:
            sparse_f = 0.0
        sparse_score = sparse_f / (max_bm25 or 1.0)

        fb = feedback_map.get(m["id"])
        fb_score = float(fb.score) if fb else 0.0

        # Weighted hybrid
        hybrid = (
            0.45 * dense_score_norm +
            0.30 * sparse_score +
            0.15 * rerank_val +   # normalized reranker contribution (0..1)
            0.10 * fb_score
        )

        # guard NaN / negative weirdness
        if hybrid != hybrid or hybrid is None:
            hybrid = 0.0
        # clamp to 0..1 just in case
        hybrid = max(0.0, min(1.0, hybrid))

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
