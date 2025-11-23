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

# We'll still reuse HF_EMBED_MODEL env var just as the model name,
# but we are NOT calling the Hugging Face Inference API anymore.
EMBED_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# threshold on hybrid score (dense + BM25) for hallucination blocking
MIN_SCORE_THRESHOLD = float(os.getenv("RAG_MIN_SCORE_THRESHOLD", "0.35"))

TOP_K_DEFAULT = 8

# cache the model so it loads only once
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
        # this loads a small, fast sentence-transformer model
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


# ---------------------- Local embeddings ----------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed text using a local SentenceTransformer model.

    This REPLACES the old Hugging Face Inference API call, so
    there is no more HTTP 410 error.
    """
    if not texts:
        return []

    model = _get_model()
    # returns a numpy array of shape (len(texts), dim)
    embs = model.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)

    # convert numpy arrays -> plain Python lists for Pinecone
    return [emb.tolist() for emb in embs]


# ---------------------- Pinecone upsert/query ----------------------

def upsert_vectors(items: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """
    Upsert in batches to avoid Pinecone's 2MB request limit.
    Each vector (id + values + metadata) must fit into the limit.
    """
    if not items:
        return

    index = _get_pinecone_index()

    # chunk into batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        index.upsert(vectors=batch)



def query_vectors(
    query_text: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone using dense embeddings.
    Returns list of matches with metadata & vector score.
    """
    index = _get_pinecone_index()

    # local embedding instead of HF API
    q_vec = embed_texts([query_text])[0]

    flt: Dict[str, Any] = {}
    if user_id is not None:
        flt["user_id"] = str(user_id)
    if domain:
        flt["domain"] = domain

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
        matches.append(
            {
                "id": m.get("id"),
                "text": text,
                "score": float(m.get("score", 0.0)),  # dense score
                "meta": meta,
            }
        )
    return matches


# ---------------------- Hybrid retrieval (dense + BM25) ----------------------

def retrieve(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:

    # 1. Initial dense retrieval
    dense_matches = query_vectors(query, user_id=user_id, domain=domain, top_k=max(top_k * 2, 10))
    if not dense_matches:
        return []

    # --- DOMAIN LOCKING ----------------------------------------------------
    if domain is None:
        domains = [m["meta"].get("domain") for m in dense_matches if "meta" in m]
        if domains:
            domain = max(set(domains), key=domains.count)

    dense_matches = [m for m in dense_matches if m["meta"].get("domain") == domain]
    if not dense_matches:
        return []
    # -----------------------------------------------------------------------

    # 2. BM25 sparse scoring on filtered set
    corpus_tokens = [m["text"].split() for m in dense_matches]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query.split())

    if bm25_scores is None or bm25_scores.size == 0:
        max_bm25 = 1.0
    else:
        max_bm25 = float(bm25_scores.max()) or 1.0

    # 3. Combine dense + sparse + feedback
    combined: List[Dict[str, Any]] = []

    # Load feedback for all pinecone IDs in one DB call
    pinecone_ids = [m["id"] for m in dense_matches]
    feedback_qs = ChunkFeedback.objects.filter(pinecone_id__in=pinecone_ids)
    feedback_map = {fb.pinecone_id: fb for fb in feedback_qs}

    FEEDBACK_WEIGHT = 0.05  # safe value; can tune if needed

    for m, bm_s in zip(dense_matches, bm25_scores):
        dense = float(m["score"])
        sparse = float(bm_s) / max_bm25

        fb_obj = feedback_map.get(m["id"])
        fb_score = fb_obj.score if fb_obj else 0.0

        hybrid = 0.5 * dense + 0.5 * sparse
        hybrid += FEEDBACK_WEIGHT * fb_score  # ⭐ feedback learning

        combined.append(
            {
                "id": m["id"],
                "text": m["text"],
                "meta": m["meta"],
                "dense_score": dense,
                "bm25_score": sparse,
                "feedback_score": fb_score,
                "score": hybrid,
            }
        )

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
