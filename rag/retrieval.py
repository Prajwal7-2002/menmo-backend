# rag/retrieval.py

import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from .models import ChunkFeedback

# ---------------------- Config ----------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE_THRESHOLD", "0.35"))
TOP_K_DEFAULT = 8

_model: SentenceTransformer | None = None


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


# ---------------------- Pinecone Query ----------------------

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
        flt["document_id"] = document_id   # NEW

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


# ---------------------- Hybrid Retrieval ----------------------

def retrieve(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,   # NEW
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    # 1. Dense retrieval (with document filter)
    dense_matches = query_vectors(
        query_text=query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,   # NEW
        top_k=max(top_k * 2, 10),
    )

    if not dense_matches:
        return []

    # Domain locking
    if domain is None:
        domains = [m["meta"].get("domain") for m in dense_matches]
        if domains:
            domain = max(set(domains), key=domains.count)

    dense_matches = [m for m in dense_matches if m["meta"].get("domain") == domain]

    if not dense_matches:
        return []

    # 2. BM25 sparse scoring
    corpus_tokens = [m["text"].split() for m in dense_matches]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query.split())
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0

    # 3. Combine dense + sparse + feedback
    combined = []
    pinecone_ids = [m["id"] for m in dense_matches]
    feedback_qs = ChunkFeedback.objects.filter(pinecone_id__in=pinecone_ids)
    feedback_map = {fb.pinecone_id: fb for fb in feedback_qs}

    for m, sparse in zip(dense_matches, bm25_scores):
        dense_score = float(m["score"])
        sparse_score = float(sparse) / max_bm25

        fb = feedback_map.get(m["id"])
        fb_score = fb.score if fb else 0.0

        hybrid = 0.5 * dense_score + 0.5 * sparse_score + 0.05 * fb_score

        combined.append({
            "id": m["id"],
            "text": m["text"],
            "meta": m["meta"],
            "score": hybrid,
            "dense_score": dense_score,
            "bm25_score": sparse_score,
            "feedback_score": fb_score,
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
