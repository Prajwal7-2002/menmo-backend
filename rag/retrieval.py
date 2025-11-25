# rag/retrieval.py
import os
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # Not required unless reranker enabled

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from .models import ChunkFeedback
from .llm import call_llm_answer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))

# ---------------------- Config ----------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "/app/models/minilm").strip()

MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE_THRESHOLD", "0.35"))
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "8"))
AUTO_RELAX_THRESHOLD = float(os.getenv("RAG_AUTO_RELAX_THRESHOLD", "-0.5"))

USE_RERANKER = str(os.getenv("USE_RERANKER", "false")).lower() in ("1", "true", "yes")
USE_QUERY_REWRITE = str(os.getenv("USE_QUERY_REWRITE", "false")).lower() in ("1", "true", "yes")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

_model = None
_reranker = None

# ---------------------- Helpers ----------------------

def _get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# ---------------------- MODEL LOADING (IMPORTANT) ----------------------

def _get_model() -> SentenceTransformer:
    """
    Force loading only from local folder.
    Prevent all HuggingFace Hub downloads.
    """
    global _model
    if _model is None:
        model_path = (EMBED_MODEL_NAME or "").strip()

        logger.info(f"HF_EMBED_MODEL resolved to: '{model_path}'")

        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"❌ Local model not found at '{model_path}'. "
                "HF_EMBED_MODEL must point to /app/models/minilm"
            )

        logger.info("✔ Loading MiniLM from local model folder (no downloads)")
        _model = SentenceTransformer(model_path)

    return _model

def _get_reranker():
    global _reranker
    if not USE_RERANKER:
        raise RuntimeError("Reranker disabled via USE_RERANKER")
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder unavailable in this environment")

    if _reranker is None:
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker

# ---------------------- Embeddings ----------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_model()
    safe = [t if (t and t.strip()) else " " for t in texts]
    embs = model.encode(safe, convert_to_numpy=True)
    return [e.tolist() for e in embs]

# ---------------------- Upsert ----------------------

def upsert_vectors(items: List[Dict[str, Any]], batch_size=100):
    index = _get_pinecone_index()
    for i in range(0, len(items), batch_size):
        index.upsert(vectors=items[i:i+batch_size])

# ---------------------- Query Vectors ----------------------

def _extract_text(meta):
    return meta.get("chunk_text") or meta.get("text") or meta.get("content") or ""

def query_vectors(query_text, user_id=None, domain=None, document_id=None, top_k=TOP_K_DEFAULT):
    if not query_text.strip():
        return []

    q_vec = embed_texts([query_text])[0]

    flt = {}
    if user_id:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = domain
    if document_id:
        flt["document_id"] = document_id

    index = _get_pinecone_index()

    try:
        res = index.query(vector=q_vec, top_k=top_k, include_metadata=True, filter=flt or None)
    except Exception:
        res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)

    output = []
    for m in res.get("matches", []):
        meta = m.get("metadata") or {}
        text = _extract_text(meta)
        try:
            score = float(m.get("score", 0))
        except:
            score = 0.0

        output.append({
            "id": m["id"],
            "text": text,
            "score": score,
            "meta": meta
        })

    return output

# ---------------------- Hybrid Retrieval ----------------------

def retrieve(query, user_id=None, domain=None, document_id=None, top_k=TOP_K_DEFAULT):
    logger.info(f"retrieve(): query={query[:80]}...")

    # Optional: Query rewrite
    effective_query = query
    if USE_QUERY_REWRITE:
        try:
            rq = call_llm_answer(
                question=f"Rewrite: {query}",
                context="",
                mood="serious",
                max_tokens=48
            )
            if rq and len(rq) < 400:
                effective_query = rq
        except:
            pass

    dense = query_vectors(effective_query, user_id, domain, document_id, top_k=max(10, top_k * 2))
    if not dense:
        return []

    # Domain auto-detect
    if domain is None:
        domains = [d["meta"].get("domain") for d in dense if d["meta"].get("domain")]
        if domains:
            domain = max(set(domains), key=domains.count)

    # Strict filter → relax if empty
    filtered = [d for d in dense if domain is None or d["meta"].get("domain") == domain]
    if not filtered:
        filtered = dense

    dense = filtered

    # BM25 scoring
    corpus_tokens = [d["text"].split() for d in dense]
    try:
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(effective_query.split())
    except:
        bm25_scores = [0.0] * len(dense)

    max_bm25 = max(bm25_scores) if bm25_scores else 1.0

    # Feedback scores
    pine_ids = [d["id"] for d in dense]
    fb_map = {fb.pinecone_id: fb for fb in ChunkFeedback.objects.filter(pinecone_id__in=pine_ids)}

    # Optional reranker
    if USE_RERANKER:
        try:
            pairs = [(effective_query, d["text"]) for d in dense]
            rr = _get_reranker().predict(pairs)
            rr = [float(x) for x in rr]
        except:
            rr = [0.0] * len(dense)
    else:
        rr = [0.0] * len(dense)

    # Normalize reranker
    if rr:
        rmin, rmax = min(rr), max(rr)
        rspan = max(1e-6, rmax - rmin)
        rr_norm = [(x - rmin) / rspan for x in rr]
    else:
        rr_norm = [0.5] * len(dense)

    # Normalize dense
    raw_dense = [float(d["score"]) for d in dense]
    maxd, mind = max(raw_dense), min(raw_dense)
    dspan = max(1e-6, maxd - mind)

    combined = []
    for idx, d in enumerate(dense):
        dense_norm = (d["score"] - mind) / dspan

        bm25_norm = bm25_scores[idx] / max_bm25 if max_bm25 else 0.0

        fb = fb_map.get(d["id"])
        fb_score = float(fb.score) if fb else 0.0

        hybrid = (
            0.45 * dense_norm +
            0.30 * bm25_norm +
            0.15 * rr_norm[idx] +
            0.10 * fb_score
        )

        hybrid = max(0.0, min(1.0, hybrid))

        combined.append({
            **d,
            "score": hybrid,
            "dense_score_raw": d["score"],
            "dense_score_norm": dense_norm,
            "bm25_score": bm25_norm,
            "rerank_score": rr[idx],
            "rerank_norm": rr_norm[idx],
            "feedback_score": fb_score,
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
