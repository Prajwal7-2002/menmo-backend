# rag/vectorstore.py
"""
Thin Pinecone wrapper shared by document retrieval and memory.

Normalises the SDK's response objects to plain dicts so callers don't need
to guess between `res.matches` / `res["matches"]` / `m.score` / `m["score"]`.
"""
import logging
import math
import os
import threading
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DOC_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")
MEMORY_INDEX_NAME = os.getenv("PINECONE_MEMORY_INDEX", "neurostack-memory")

_client = None
_indexes: Dict[str, Any] = {}
_lock = threading.Lock()


class VectorStoreError(RuntimeError):
    pass


def get_index(name: str):
    global _client
    if name in _indexes:
        return _indexes[name]
    with _lock:
        if name not in _indexes:
            if not PINECONE_API_KEY:
                raise VectorStoreError("PINECONE_API_KEY not set")
            if _client is None:
                from pinecone import Pinecone
                _client = Pinecone(api_key=PINECONE_API_KEY)
            _indexes[name] = _client.Index(name)
    return _indexes[name]


def _field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def query(index_name: str, vector: List[float], top_k: int,
          flt: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Returns [{"id", "score", "metadata"}]. Raises VectorStoreError on failure.

    There is deliberately no "retry without the filter" fallback: the filter
    is what keeps one user's data out of another user's answers.
    """
    index = get_index(index_name)
    kwargs: Dict[str, Any] = {"vector": vector, "top_k": top_k, "include_metadata": True}
    if flt:
        kwargs["filter"] = flt
    try:
        res = index.query(**kwargs)
    except Exception as e:
        raise VectorStoreError(f"query on {index_name} failed: {e}") from e

    out = []
    for m in (_field(res, "matches") or []):
        try:
            score = float(_field(m, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        out.append({
            "id": _field(m, "id"),
            "score": score,
            "metadata": dict(_field(m, "metadata") or {}),
        })
    return out


def upsert(index_name: str, items: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """Raises VectorStoreError if any batch fails, so callers can report it."""
    index = get_index(index_name)
    for i in range(0, len(items), batch_size):
        try:
            index.upsert(vectors=items[i:i + batch_size])
        except Exception as e:
            raise VectorStoreError(f"upsert to {index_name} failed at batch {i // batch_size}: {e}") from e


def delete_ids(index_name: str, ids: Iterable[str], batch_size: int = 100) -> int:
    ids = [i for i in ids if i]
    if not ids:
        return 0
    index = get_index(index_name)
    for i in range(0, len(ids), batch_size):
        index.delete(ids=ids[i:i + batch_size])
    return len(ids)


def delete_by_filter(index_name: str, flt: Dict[str, Any], dim: int, max_rounds: int = 20) -> int:
    """
    Delete every vector matching a metadata filter.

    Serverless indexes don't support delete-by-filter, so we page through
    matches with a filtered query (any non-zero probe vector works because
    the filter, not similarity, selects the rows) and delete by id.
    """
    probe = [1.0 / math.sqrt(dim)] * dim
    seen = set()
    for _ in range(max_rounds):
        matches = query(index_name, probe, top_k=1000, flt=flt)
        # Deletes are eventually consistent, so a re-query can return ids we
        # already deleted; stop once a page brings nothing new.
        ids = [m["id"] for m in matches if m.get("id") and m["id"] not in seen]
        if not ids:
            break
        delete_ids(index_name, ids)
        seen.update(ids)
        if len(matches) < 1000:
            break
    return len(seen)
