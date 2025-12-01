# rag/retrieval.py
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from .llm import call_llm_answer
from .models import ChunkFeedback

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))

# ---------------------- Config (env-driven) ----------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "8"))

# Hybrid weight defaults (change via env to tune)
DENSE_WEIGHT = float(os.getenv("RAG_WEIGHT_DENSE", "0.45"))
BM25_WEIGHT  = float(os.getenv("RAG_WEIGHT_BM25", "0.30"))
RR_WEIGHT    = float(os.getenv("RAG_WEIGHT_RERANK", "0.15"))
FB_WEIGHT    = float(os.getenv("RAG_WEIGHT_FEEDBACK", "0.10"))

USE_RERANKER = str(os.getenv("USE_RERANKER", "false")).lower() in ("1", "true", "yes")
USE_QUERY_REWRITE = str(os.getenv("USE_QUERY_REWRITE", "false")).lower() in ("1", "true", "yes")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# Minimum hybrid score to consider a chunk "relevant" (tuneable)
MIN_HYBRID_CONF = float(os.getenv("MIN_HYBRID_CONF", "0.60"))

_embedder = None
_reranker = None

# ---------------------- Pinecone index helper ----------------------
def _get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# ---------------------- Embeddings ----------------------
def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"🔄 Loading HF embedding model → {HF_EMBED_MODEL}")
        _embedder = SentenceTransformer(HF_EMBED_MODEL)
        logger.info("✅ Embedding model loaded")
    return _embedder

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        model = _get_embedder()
        safe = [t if (t and t.strip()) else " " for t in texts]
        vecs = model.encode(safe, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.error(f"❌ HF embedding failed → {e}")
        return [[0.01] * EMBED_DIM for _ in texts]

# ---------------------- Reranker (optional) ----------------------
def _get_reranker():
    global _reranker
    if not USE_RERANKER:
        raise RuntimeError("Reranker disabled via USE_RERANKER")
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        raise RuntimeError("CrossEncoder unavailable")
    logger.info(f"🔄 Loading CrossEncoder reranker → {RERANKER_MODEL}")
    _reranker = CrossEncoder(RERANKER_MODEL)
    logger.info("✅ Reranker loaded")
    return _reranker

# ---------------------- Query Vectors ----------------------
def _extract_text(meta: Dict[str, Any]) -> str:
    return meta.get("chunk_text") or meta.get("full_text") or meta.get("snippet") or ""

def query_vectors(
    query_text: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    if not query_text or not query_text.strip():
        return []
    try:
        q_embs = embed_texts([query_text])
        if not q_embs:
            return []
        q_vec = q_embs[0]
    except Exception as e:
        logger.error(f"query_vectors(): embedding failed: {e}")
        return []
    flt: Dict[str, Any] = {}
    if user_id is not None:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = domain
    if document_id:
        flt["document_id"] = document_id
    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error(f"query_vectors(): failed to get Pinecone index: {e}")
        return []
    try:
        res = index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
            filter=flt or None,
        )
    except Exception as e:
        logger.error(f"query_vectors(): Pinecone query failed (with filter): {e}")
        try:
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
        except Exception as e2:
            logger.error(f"query_vectors(): Pinecone query failed (no filter): {e2}")
            return []
    output: List[Dict[str, Any]] = []
    for m in res.get("matches", []):
        meta = m.get("metadata") or {}
        text = _extract_text(meta)
        try:
            score = float(m.get("score", 0.0))
        except Exception:
            score = 0.0
        output.append(
            {
                "id": m.get("id"),
                "text": text,
                "score": score,
                "meta": meta,
            }
        )
    return output

# ---------------------- Hybrid Retrieval ----------------------
def retrieve(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    logger.info(f"retrieve(): query={query[:80]}...")
    effective_query = query
    if USE_QUERY_REWRITE:
        try:
            rq = call_llm_answer(
                question=(
                    "Rewrite this query to optimize semantic document retrieval, "
                    "keeping the same meaning but removing noise:\n" + query
                ),
                context="",
                mood="serious",
                max_tokens=48,
            )
            if rq and len(rq) < 400:
                effective_query = rq
        except Exception as e:
            logger.warning(f"retrieve(): query rewrite failed: {e}")

    # Dense / vector search
    try:
        dense = query_vectors(
            effective_query,
            user_id=user_id,
            domain=domain,
            document_id=document_id,
            top_k=max(10, top_k * 2),
        )
    except Exception as e:
        logger.error(f"retrieve(): query_vectors failed: {e}")
        dense = []
    if not dense:
        return []

    # Domain auto-detect
    if domain is None:
        domains = [d["meta"].get("domain") for d in dense if d["meta"].get("domain")]
        if domains:
            domain = max(set(domains), key=domains.count)

    # Relax filter if empty
    filtered = [d for d in dense if (domain is None or d["meta"].get("domain") == domain)]
    if not filtered:
        filtered = dense
    dense = filtered

    # BM25 scoring
    corpus_tokens = [d["text"].split() for d in dense]
    try:
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(effective_query.split())
    except Exception as e:
        logger.warning(f"retrieve(): BM25 scoring failed: {e}")
        bm25_scores = np.zeros(len(dense))

    max_bm25 = float(bm25_scores.max()) if hasattr(bm25_scores, "max") and bm25_scores.size > 0 else 1.0

    # Feedback map
    pine_ids = [d["id"] for d in dense]
    fb_map = {
        fb.pinecone_id: fb
        for fb in ChunkFeedback.objects.filter(pinecone_id__in=pine_ids)
    }

    # Optional reranker
    if USE_RERANKER:
        try:
            pairs = [(effective_query, d["text"]) for d in dense]
            rr_model = _get_reranker()
            rr = rr_model.predict(pairs)
            rr = [float(x) for x in rr]
        except Exception as e:
            logger.warning(f"retrieve(): reranker failed: {e}")
            rr = [0.0] * len(dense)
    else:
        rr = [0.0] * len(dense)

    # normalize reranker
    if rr:
        rmin, rmax = min(rr), max(rr)
        rspan = max(1e-6, rmax - rmin)
        rr_norm = [(x - rmin) / rspan for x in rr]
    else:
        rr_norm = [0.5] * len(dense)

    # normalize dense
    raw_dense = [float(d["score"]) for d in dense]
    maxd, mind = max(raw_dense), min(raw_dense)
    dspan = max(1e-6, maxd - mind)
    combined: List[Dict[str, Any]] = []
    for idx, d in enumerate(dense):
        dense_norm = (d["score"] - mind) / dspan
        bm25_norm = bm25_scores[idx] / max_bm25 if max_bm25 else 0.0
        fb = fb_map.get(d["id"])
        fb_score = float(fb.score) if fb else 0.0
        hybrid = (
            DENSE_WEIGHT * dense_norm
            + BM25_WEIGHT * bm25_norm
            + RR_WEIGHT * rr_norm[idx]
            + FB_WEIGHT * fb_score
        )
        hybrid = max(0.0, min(1.0, hybrid))
        combined.append(
            {
                **d,
                "score": hybrid,
                "dense_score_raw": d["score"],
                "dense_score_norm": dense_norm,
                "bm25_score": float(bm25_scores[idx]) if hasattr(bm25_scores, "__len__") else 0.0,
                "bm25_score_norm": bm25_norm,
                "rerank_score": rr[idx],
                "rerank_norm": rr_norm[idx],
                "feedback_score": fb_score,
            }
        )
    combined.sort(key=lambda x: x["score"], reverse=True)
    # Return top_k with full diagnostics
    return combined[:top_k]

# ---------------------- Upsert (Needed by loader.py) ----------------------
def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100):
    """Restore compatibility for loader.py which imports this function."""
    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error(f"upsert_vectors(): failed to get Pinecone index: {e}")
        return

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logger.error(f"upsert_vectors(): batch upsert failed: {e}")


# ==========================================================
#  Agentic Retrieval Confidence Wrapper
# ==========================================================
def retrieve_with_conf(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT
):
    """
    Returns: (chunks, scores, top_hybrid_score)
    Each returned chunk has per-component scores attached (dense/bm25/rerank/feedback).
    """
    chunks = retrieve(query, user_id, domain, document_id, top_k)
    if not chunks:
        return [], [], 0.0
    scores = [c["score"] for c in chunks]
    conf = max(scores) if scores else 0.0
    return chunks, scores, conf

# small helper the agent can use
def is_confident_enough(confidence: float) -> bool:
    try:
        return float(confidence) >= float(MIN_HYBRID_CONF)
    except Exception:
        return False
