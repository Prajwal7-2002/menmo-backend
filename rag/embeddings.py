# rag/embeddings.py
"""
Single shared embedding model for documents and memory.

Previously retrieval.py and memory.py each loaded their own copy of the
model (double the RAM per worker) and retrieval silently returned constant
dummy vectors when loading failed, which then got indexed. Now there is one
lazily-loaded, thread-safe instance and failures raise.
"""
import logging
import os
import threading
from typing import List

logger = logging.getLogger(__name__)

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

_model = None
_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    pass


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except Exception as e:
                    raise EmbeddingError(f"sentence-transformers not installed: {e}") from e
                logger.info("Loading embedding model: %s", HF_EMBED_MODEL)
                try:
                    _model = SentenceTransformer(HF_EMBED_MODEL)
                except Exception as e:
                    raise EmbeddingError(f"failed to load {HF_EMBED_MODEL}: {e}") from e
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed texts as unit-length vectors (so dot product == cosine)."""
    if not texts:
        return []
    safe = [t if (t and t.strip()) else " " for t in texts]
    try:
        vecs = get_model().encode(
            safe,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"encode failed: {e}") from e
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
