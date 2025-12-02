# rag/memory.py  -> Hybrid Memory System (Conversation + Facts)

import os, uuid
from typing import List, Dict, Any
from pinecone import Pinecone

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME")

# ==========================================================
# INTERNAL: Load embedding model lazily
# ==========================================================
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ==========================================================
# 1️⃣  CONVERSATION MEMORY (Existing Feature)
# ==========================================================

def store_conversation_turn(user, query: str, answer: str):
    """
    Store chat history as vector memory.
    """
    try:
        model = get_embedder()
        text = f"user:{query}\nassistant:{answer}"
        vec = model.encode(text).tolist()

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        index.upsert([{
            "id": f"conv-{user.id}-{uuid.uuid4()}",
            "values": vec,
            "metadata": {
                "type":"conversation",
                "user":str(user.id),
                "text":text
            }
        }])

    except Exception as e:
        print("❌ Failed storing conversation vector:", e)


def load_vector_memory(user, query: str, top_k: int = 6) -> List[Dict[str, str]]:
    """
    Retrieve past conversation for chat continuity.
    """
    try:
        model = get_embedder()
        q_vec = model.encode(query).tolist()

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        response = index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
            filter={"type":"conversation", "user":str(user.id)}
        )

        messages = []
        for m in response.get("matches", []):
            if m.get("score", 0) < 0.40:
                continue

            txt = m["metadata"]["text"]
            if "\nassistant:" in txt:
                u, a = txt.split("\nassistant:")
                u = u.replace("user:", "").strip()
                a = a.strip()
                messages.append({"role": "user",      "content": u})
                messages.append({"role": "assistant", "content": a})

        return messages

    except Exception as e:
        print("❌ Failed retrieving conversation memory:", e)
        return []


# ==========================================================
# 2️⃣  FACT MEMORY (NEW) – Clean, Persistent, Deduped
# ==========================================================

def _fact_id(user, text):
    return "fact-" + str(user.id) + "-" + uuid.uuid4().hex


def add_memory_fact(user, text: str) -> bool:
    """
    Store a clean memory fact (not a conversation turn).
    Deduped properly. Stored in Pinecone.
    """
    try:
        text = text.strip()
        if not text:
            return False

        # Check duplicates
        if memory_exists(user, text):
            return False

        model = get_embedder()
        vec = model.encode(text).tolist()

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        index.upsert([{
            "id": _fact_id(user, text),
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


def memory_exists(user, text: str) -> bool:
    """
    Check if a memory fact already exists for dedupe.
    """
    try:
        model = get_embedder()
        q_vec = model.encode(text).tolist()

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

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
    except:
        return False


# ==========================================================
# 3️⃣  FACT MEMORY RECALL (NEW)
# ==========================================================

def relevant_chunks(user, query: str, top_k: int = 4):
    """
    Retrieve FACT memory (not conversation memory).
    Used by agentic_answer for reasoning.
    """
    try:
        model = get_embedder()
        q_vec = model.encode(query).tolist()

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

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
