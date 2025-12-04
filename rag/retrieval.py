# rag/retrieval.py
import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

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
BM25_WEIGHT = float(os.getenv("RAG_WEIGHT_BM25", "0.35"))
RR_WEIGHT = float(os.getenv("RAG_WEIGHT_RERANK", "0.15"))
FB_WEIGHT = float(os.getenv("RAG_WEIGHT_FEEDBACK", "0.10"))

# Options
USE_RERANKER = str(os.getenv("USE_RERANKER", "false")).lower() in ("1", "true", "yes")
USE_QUERY_REWRITE = str(os.getenv("USE_QUERY_REWRITE", "false")).lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# Strictness
MIN_HYBRID_CONF = float(os.getenv("MIN_HYBRID_CONF", "0.15"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "80"))

# Internal caches
_embedder = None
_reranker = None

# ---------------------- Helpers ----------------------


def _get_pinecone_index():
    """Return a Pinecone Index object (raises if API key missing)."""
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _get_embedder():
    """Lazy-load HF SentenceTransformer embedder. Raises if model not installed."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            logger.exception("sentence-transformers not available: %s", e)
            raise
        logger.info("Loading embedding model: %s", HF_EMBED_MODEL)
        _embedder = SentenceTransformer(HF_EMBED_MODEL)
        logger.info("Embedding model loaded")
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts with HF model. On failure returns small constant vectors to avoid crashes.
    """
    if not texts:
        return []
    try:
        model = _get_embedder()
        safe = [t if (t and t.strip()) else " " for t in texts]
        vecs = model.encode(safe, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.error("embed_texts(): embedder failed: %s", e)
        # fallback tiny non-zero vectors
        return [[0.01] * EMBED_DIM for _ in texts]


def _extract_text(meta: Dict[str, Any]) -> str:
    txt = (meta or {}).get("chunk_text") or (meta or {}).get("full_text") or (meta or {}).get("snippet") or ""
    txt = re.sub(r'[\x00-\x1f\x7f]+', ' ', txt)
    txt = re.sub(r'\s{2,}', ' ', txt).strip()
    return txt


def _build_filter(user_id: Optional[int], domain: Optional[str], document_id: Optional[str]) -> Dict[str, Any]:
    flt: Dict[str, Any] = {}
    if user_id is not None:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = str(domain).lower()
    if document_id:
        flt["document_id"] = str(document_id)
    return flt


# ---------------------- Pinecone query / retrieval ----------------------


def query_vectors(query_text: str,
                  user_id: Optional[int] = None,
                  domain: Optional[str] = None,
                  document_id: Optional[str] = None,
                  top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Query Pinecone for semantic matches. Returns list of dicts: {id, metadata, score, ...}
    This function tries a filtered query first; if it fails, falls back to unfiltered query.
    """
    if not query_text or not query_text.strip():
        return []

    try:
        q_embs = embed_texts([query_text])
        if not q_embs:
            return []
        q_vec = q_embs[0]
    except Exception as e:
        logger.error("query_vectors(): embedding failed: %s", e)
        return []

    flt = _build_filter(user_id, domain, document_id)

    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error("query_vectors(): failed to get Pinecone index: %s", e)
        return []

    try:
        if flt:
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True, filter=flt)
        else:
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    except Exception as e:
        logger.warning("query_vectors(): Pinecone query error (will attempt fallback without filter): %s", e)
        try:
            res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
        except Exception as e2:
            logger.error("query_vectors(): Pinecone fallback failed: %s", e2)
            return []

    # Normalize across possible response shapes (SDK versions)
    matches = []
    if isinstance(res, dict):
        matches = res.get("matches", []) or res.get("vectors", [])
    else:
        # try attribute access (older/newer clients)
        try:
            matches = res.matches
        except Exception:
            try:
                matches = res.vectors
            except Exception:
                matches = []

    output: List[Dict[str, Any]] = []
    for m in matches:
        # m can be dict with 'id','score','metadata' or object with attributes
        if isinstance(m, dict):
            mid = m.get("id")
            score = m.get("score") or m.get("value") or 0.0
            meta = m.get("metadata") or {}
        else:
            mid = getattr(m, "id", None)
            score = getattr(m, "score", 0.0)
            meta = getattr(m, "metadata", {}) or {}
        try:
            score = float(score)
        except Exception:
            score = 0.0
        txt = _extract_text(meta)
        output.append({"id": mid, "text": txt, "score": score, "meta": meta})
    return output


def retrieve(query: str,
             user_id: Optional[int] = None,
             domain: Optional[str] = None,
             document_id: Optional[str] = None,
             top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
     - optional query rewrite (disabled by default)
     - dense vector search (query_vectors)
     - strict post-filtering by domain/document_id (to prevent leaks)
     - BM25 scoring on the returned small set
     - optional reranking
     - combine dense,bm25,rerank,feedback into hybrid score
     - dedupe by normalized text
    Returns top_k combined chunks (each with keys: id, text, meta, score, ...).
    """
    logger.info("retrieve(): q=%s...", query[:140])
    effective_query = query

    if USE_QUERY_REWRITE:
        try:
            from .llm import call_llm_answer
            rq = call_llm_answer(question=("Rewrite this query to optimize semantic retrieval:\n" + query),
                                 context="", mood="serious", max_tokens=48)
            if rq and len(rq) < 400:
                effective_query = rq
                logger.debug("retrieve(): query rewritten")
        except Exception as e:
            logger.warning("retrieve(): query rewrite failed: %s", e)

    try:
        dense = query_vectors(effective_query, user_id=user_id, domain=domain, document_id=document_id,
                              top_k=max(10, top_k * 2))
    except Exception as e:
        logger.error("retrieve(): query_vectors failed: %s", e)
        dense = []

    if not dense:
        return []

    # ----------------------------
    # ENFORCE STRICT FILTERING
    # ----------------------------
    # If caller requested a domain or document_id we must ensure final candidates match metadata.
    if domain or document_id:
        strict_filtered = []
        for d in dense:
            meta = (d.get("meta") or {}) or {}
            meta_domain = str(meta.get("domain", "")).lower()
            meta_docid = str(meta.get("document_id", ""))

            if domain and meta_domain != str(domain).lower():
                continue
            if document_id and meta_docid != str(document_id):
                continue
            strict_filtered.append(d)

        # If the caller asked for domain/document and nothing matches, return empty
        if (domain or document_id) and not strict_filtered:
            logger.info("retrieve(): strict filter matched 0 candidates — returning empty")
            return []
        dense = strict_filtered

    # Filter out trivially short chunks early
    filtered_long = [d for d in dense if len((d.get("text") or "").strip()) >= MIN_TEXT_LEN]
    if filtered_long:
        dense = filtered_long
    else:
        logger.debug("retrieve(): no chunk >= MIN_TEXT_LEN; keeping original filtered set")

    # BM25 (on small dense set)
    try:
        corpus_tokens = [d["text"].split() for d in dense]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(effective_query.split())
    except Exception as e:
        logger.warning("retrieve(): BM25 failed: %s", e)
        bm25_scores = np.zeros(len(dense))

    max_bm25 = float(bm25_scores.max()) if hasattr(bm25_scores, "max") and bm25_scores.size > 0 else 1.0

    # feedback map
    pine_ids = [d["id"] for d in dense]
    try:
        fb_map = {fb.pinecone_id: fb for fb in ChunkFeedback.objects.filter(pinecone_id__in=pine_ids)}
    except Exception as e:
        logger.warning("retrieve(): failed to load ChunkFeedback: %s", e)
        fb_map = {}

    # optional reranker
    rr = [0.0] * len(dense)
    if USE_RERANKER and dense:
        try:
            from sentence_transformers import CrossEncoder
            rr_model = CrossEncoder(RERANKER_MODEL)
            pairs = [(effective_query, d["text"]) for d in dense]
            rr = [float(x) for x in rr_model.predict(pairs)]
        except Exception as e:
            logger.warning("retrieve(): reranker failed: %s", e)
            rr = [0.0] * len(dense)

    # normalize and combine
    if rr:
        rmin, rmax = min(rr), max(rr)
        rspan = max(1e-6, rmax - rmin)
        rr_norm = [(x - rmin) / rspan for x in rr]
    else:
        rr_norm = [0.5] * len(dense)

    raw_dense = [float(d["score"]) for d in dense]
    maxd, mind = max(raw_dense), min(raw_dense)
    dspan = max(1e-6, maxd - mind)

    combined: List[Dict[str, Any]] = []
    seen_norm_texts = set()
    for idx, d in enumerate(dense):
        dense_norm = (d["score"] - mind) / dspan if dspan else 0.0
        bm25_norm = (bm25_scores[idx] / max_bm25) if max_bm25 else 0.0
        fb = fb_map.get(d["id"])
        fb_score = float(fb.score) if fb else 0.0
        hybrid = DENSE_WEIGHT * dense_norm + BM25_WEIGHT * bm25_norm + RR_WEIGHT * rr_norm[idx] + FB_WEIGHT * fb_score
        hybrid = max(0.0, min(1.0, hybrid))

        # dedupe by normalized snippet (first 300 chars)
        norm_text = ' '.join((d.get("text") or "").split()).lower()[:300]
        if norm_text in seen_norm_texts:
            continue
        seen_norm_texts.add(norm_text)

        combined.append({
            **d,
            "score": hybrid,
            "dense_score_raw": d["score"],
            "dense_score_norm": dense_norm,
            "bm25_score": float(bm25_scores[idx]) if hasattr(bm25_scores, "__len__") else 0.0,
            "bm25_score_norm": bm25_norm,
            "rerank_score": rr[idx] if rr else 0.0,
            "rerank_norm": rr_norm[idx],
            "feedback_score": fb_score,
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("retrieve(): top scores: %s", ", ".join(f"{round(x['score'], 3)}" for x in combined[:6]))
    return combined[:top_k]


def retrieve_with_conf(query: str, user_id: Optional[int] = None, domain: Optional[str] = None,
                       document_id: Optional[str] = None, top_k: int = TOP_K_DEFAULT) -> Tuple[List[Dict[str, Any]], List[float], float]:
    chunks = retrieve(query, user_id, domain, document_id, top_k)
    if not chunks:
        return [], [], 0.0
    scores = [c["score"] for c in chunks]
    conf = max(scores) if scores else 0.0
    return chunks, scores, conf


def is_confident_enough(confidence: float) -> bool:
    try:
        return float(confidence) >= float(MIN_HYBRID_CONF)
    except Exception:
        return False


# ---------------------- Upsert / utilities ----------------------


def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100):
    """
    items: list of dicts in the Pinecone upsert shape accepted by your SDK variant.
    E.g.: {"id": "xxx", "values": [...], "metadata": {...}}
    """
    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error("upsert_vectors(): cannot get Pinecone index: %s", e)
        return
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            logger.exception("upsert_vectors(): batch upsert failed: %s", e)


def scan_missing_metadata(limit_per_page: int = 1000) -> Dict[str, int]:
    """
    Diagnostic: scan the index and report how many vectors are missing domain/document_id/user_id metadata.
    Returns a summary dict. Also writes lists to files in CWD: missing_domain.txt, missing_docid.txt
    NOTE: depending on Pinecone SDK version you may need to adapt pagination; this supports common dict shape.
    """
    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error("scan_missing_metadata(): cannot get Pinecone index: %s", e)
        return {"error": 1}

    missing_domain = []
    missing_docid = []
    cursor = None
    while True:
        try:
            page = index.list(limit=limit_per_page, cursor=cursor)
        except Exception as e:
            logger.exception("scan_missing_metadata(): list failed: %s", e)
            break

        vectors = page.get("vectors", []) if isinstance(page, dict) else getattr(page, "vectors", []) or []
        for v in vectors:
            meta = v.get("metadata") or {}
            if not meta.get("domain"):
                missing_domain.append(v.get("id"))
            if not meta.get("document_id"):
                missing_docid.append(v.get("id"))

        cursor = page.get("cursor") if isinstance(page, dict) else getattr(page, "cursor", None)
        if not cursor:
            break

    open("missing_domain.txt", "w").write("\n".join(missing_domain))
    open("missing_docid.txt", "w").write("\n".join(missing_docid))
    logger.info("scan_missing_metadata: missing_domain=%d missing_docid=%d", len(missing_domain), len(missing_docid))
    return {"missing_domain": len(missing_domain), "missing_docid": len(missing_docid)}


def delete_vectors_by_ids(ids: List[str], batch_size: int = 100):
    """Delete vector ids from Pinecone in batches. Returns number deleted (best-effort)."""
    if not ids:
        return 0
    try:
        index = _get_pinecone_index()
    except Exception as e:
        logger.error("delete_vectors_by_ids(): cannot get Pinecone index: %s", e)
        return 0
    deleted = 0
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            index.delete(ids=batch)
            deleted += len(batch)
        except Exception as e:
            logger.exception("delete_vectors_by_ids(): delete batch failed: %s", e)
    logger.info("delete_vectors_by_ids(): requested delete %d ids, attempted %d", len(ids), deleted)
    return deleted
