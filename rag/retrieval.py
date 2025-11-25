# rag/retrieval.py

import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from .models import ChunkFeedback
from .llm import call_llm_answer   # <-- for query rewriting

# ---------------------- Config ----------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE_THRESHOLD", "0.35"))
TOP_K_DEFAULT = 8

_model: SentenceTransformer | None = None

# NEW: Cross-Encoder ReRanker (lightweight & fast)
_reranker: CrossEncoder | None = None


# ---------------------- Helpers ----------------------

def _get_pinecone_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _reranker


# ---------------------- Embeddings ----------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_model()
    embs = model.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
    return [emb.tolist() for emb in embs]


# ---------------------- Upsert Vectors ----------------------

def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100) -> None:
    if not items:
        return
    index = _get_pinecone_index()
    for i in range(0, len(items), batch_size):
        index.upsert(vectors=items[i:i + batch_size])


# ---------------------- Query Vectors ----------------------

def query_vectors(
    query_text: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    index = _get_pinecone_index()
    q_vec = embed_texts([query_text])[0]

    flt: Dict[str, Any] = {}

    if user_id:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = domain
    if document_id:
        flt["document_id"] = document_id

    res = index.query(
        vector=q_vec,
        top_k=top_k,
        filter=flt or None,
        include_metadata=True,
    )

    matches = []
    for m in res.get("matches", []):
        meta = m.get("metadata") or {}
        text = meta.get("chunk_text") or meta.get("snippet") or ""
        matches.append({
            "id": m.get("id"),
            "text": text,
            "score": float(m.get("score", 0.0)),
            "meta": meta,
        })

    return matches


# ---------------------- Hybrid Retrieval (Upgraded) ----------------------

def retrieve(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    # ========================
    # 🔥 1. Query Rewriting
    # ========================
    try:
        rewritten = call_llm_answer(
            question=f"Rewrite this query to improve document search accuracy: {query}",
            context="",
            mood="serious",
            max_tokens=48
        )

        if rewritten and len(rewritten) < 200:
            effective_query = rewritten
        else:
            effective_query = query
    except Exception:
        effective_query = query

    # ========================
    # 🔥 2. Dense retrieval
    # ========================
    dense_matches = query_vectors(
        query_text=effective_query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max(top_k * 2, 10),
    )

    if not dense_matches:
        return []

    # Domain auto-lock
    if domain is None:
        domains = [m["meta"].get("domain") for m in dense_matches]
        if domains:
            domain = max(set(domains), key=domains.count)

    dense_matches = [m for m in dense_matches if m["meta"].get("domain") == domain]

    if not dense_matches:
        return []

    # ========================
    # 🔥 3. BM25 Sparse Scoring
    # ========================
    corpus_tokens = [m["text"].split() for m in dense_matches]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(effective_query.split())

    max_bm25 = max(bm25_scores) if len(bm25_scores) else 1.0

    # ========================
    # 🔥 4. Feedback lookup
    # ========================
    pinecone_ids = [m["id"] for m in dense_matches]
    feedback_qs = ChunkFeedback.objects.filter(pinecone_id__in=pinecone_ids)
    feedback_map = {fb.pinecone_id: fb for fb in feedback_qs}

    # ========================
    # 🔥 5. Cross-Encoder Reranking
    # ========================
    reranker = _get_reranker()
    passage_pairs = [(effective_query, m["text"]) for m in dense_matches]
    rerank_scores = reranker.predict(passage_pairs)

    # ========================
    # 🔥 6. Hybrid Score (IMPROVED)
    # ========================
    combined = []
    for m, sparse, rerank in zip(dense_matches, bm25_scores, rerank_scores):

        dense_score = float(m["score"])

        # Safe normalize sparse
        try:
            sparse_val = float(sparse)
            if sparse_val != sparse_val:  # NaN check
                sparse_val = 0.0
        except:
            sparse_val = 0.0

        sparse_score = sparse_val / (max_bm25 or 1.0)

        fb = feedback_map.get(m["id"])
        fb_score = fb.score if fb else 0.0

        # NEW hybrid score (best practice)
        hybrid = (
            0.40 * dense_score +
            0.35 * sparse_score +
            0.20 * float(rerank) +
            0.05 * fb_score
        )

        # NaN guard
        if hybrid != hybrid:
            hybrid = 0.0

        combined.append({
            "id": m["id"],
            "text": m["text"],
            "meta": m["meta"],
            "score": hybrid,
            "dense_score": dense_score,
            "bm25_score": sparse_score,
            "rerank_score": float(rerank),
            "feedback_score": fb_score,
        })

    combined.sort(key=lambda x: x["score"], reverse=True)

    return combined[:top_k]
