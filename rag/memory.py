# rag/memory.py  -> Clean, Isolated Memory System (Conversation + Facts)

import os
import uuid
from typing import List, Dict, Any

from pinecone import Pinecone

# ==========================================================
# ENV CONFIG
# ==========================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Document vector index (separate)
DOC_INDEX = os.getenv("PINECONE_INDEX_NAME", "neurostack-rag")

# Memory vector index (fully isolated)
MEMORY_INDEX = os.getenv("PINECONE_MEMORY_INDEX", "neurostack-memory")

# ==========================================================
# INTERNAL: Lazy embedding model
# ==========================================================
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ==========================================================
# INTERNAL: Pinecone memory index connection
# ==========================================================
def get_memory_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(MEMORY_INDEX)


# ==========================================================
# 1️⃣  CONVERSATION MEMORY
# ==========================================================
def store_conversation_turn(user, query: str, answer: str):
    """Store conversation memory vectors in memory index."""
    try:
        text = f"user:{query}\nassistant:{answer}"
        vec = get_embedder().encode(text).tolist()

        index = get_memory_index()
        index.upsert([{
            "id": f"conv-{user.id}-{uuid.uuid4()}",
            "values": vec,
            "metadata": {
                "type": "conversation",
                "user": str(user.id),
                "text": text
            }
        }])

    except Exception as e:
        print("❌ Failed storing conversation memory:", e)


def load_vector_memory(user, query: str, top_k: int = 6):
    """Retrieve conversation memory ONLY from memory index."""
    try:
        q_vec = get_embedder().encode(query).tolist()
        index = get_memory_index()

        response = index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "conversation", "user": str(user.id)}
        )

        messages = []
        for m in response.get("matches", []):
            if m.get("score", 0) < 0.40:
                continue

            txt = m["metadata"]["text"]
            if "\nassistant:" in txt:
                u, a = txt.split("\nassistant:")
                messages.append({"role": "user", "content": u.replace("user:", "").strip()})
                messages.append({"role": "assistant", "content": a.strip()})

        return messages

    except Exception as e:
        print("❌ Failed retrieving conversation memory:", e)
        return []


# ==========================================================
# 2️⃣  FACT MEMORY (Long-term memory)
# ==========================================================
def _fact_id(user):
    return f"fact-{user.id}-{uuid.uuid4().hex}"


def memory_exists(user, text: str) -> bool:
    """Semantic dedupe for facts."""
    try:
        q_vec = get_embedder().encode(text).tolist()
        index = get_memory_index()

        response = index.query(
            vector=q_vec,
            top_k=5,
            include_metadata=True,
            filter={"type": "memory_fact", "user": str(user.id)}
        )

        for m in response.get("matches", []):
            stored = m["metadata"]["text"].strip().lower()
            if stored == text.strip().lower():
                return True

        return False

    except Exception:
        return False


def add_memory_fact(user, text: str) -> bool:
    """Store long-term memory fact into memory index."""
    try:
        text = text.strip()
        if not text:
            return False

        if memory_exists(user, text):
            return False

        vec = get_embedder().encode(text).tolist()
        index = get_memory_index()

        index.upsert([{
            "id": _fact_id(user),
            "values": vec,
            "metadata": {
                "type": "memory_fact",
                "user": str(user.id),
                "text": text
            }
        }])

        return True

    except Exception as e:
        print("❌ add_memory_fact failed:", e)
        return False


# ==========================================================
# 3️⃣  MEMORY RECALL
# ==========================================================
def relevant_chunks(user, query: str, top_k: int = 4):
    """Retrieve long-term memory facts."""
    try:
        q_vec = get_embedder().encode(query).tolist()
        index = get_memory_index()

        response = index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "memory_fact", "user": str(user.id)}
        )

        results = []
        for m in response.get("matches", []):
            results.append({
                "text": m["metadata"]["text"],
                "score": float(m["score"]),
                "meta": {"type": "memory_fact"}
            })

        return results

    except Exception as e:
        print("❌ relevant_chunks failed:", e)
        return []


# ==========================================================
# 4️⃣ MEMORY CLEANUP WHEN DOCUMENT IS DELETED
# ==========================================================
def clear_memory_for_document(user_id: int, document_id: str):
    """Deletes memory vectors linked to a deleted document."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(MEMORY_INDEX)

        response = index.query(
            vector=[0.0] * 384,
            top_k=500,
            include_metadata=True,
            filter={
                "user": str(user_id),
                "document_id": str(document_id)
            }
        )

        ids = [m["id"] for m in response.get("matches", [])]

        if ids:
            index.delete(ids=ids)

        print(f"🧹 Cleared {len(ids)} memory vectors for document {document_id}")

    except Exception as e:
        print(f"⚠ clear_memory_for_document failed: {e}")
