# rag/memory.py  -> Vector Conversation Memory System
from typing import List, Dict
from pinecone import Pinecone
import os, uuid

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME")


# ==========================================================
# 1️⃣  Store Conversation Turns as Embeddings
# ==========================================================

def store_conversation_turn(user, query: str, answer: str):
    """
    Save memory inside Pinecone, not in DB.
    Stored as semantic vector embeddings so AI can recall intent later.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        text = f"user:{query}\nassistant:{answer}"
        vec  = model.encode(text).tolist()

        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        index.upsert([
            {
                "id": f"conv-{user.id}-{uuid.uuid4()}",
                "values": vec,
                "metadata": {
                    "type":"conversation",
                    "user":str(user.id),
                    "text":text
                }
            }
        ])

    except Exception as e:
        print("❌ Failed storing conversation vector:", e)



# ==========================================================
# 2️⃣  Retrieve Semantic Conversation Memory
# ==========================================================

def load_vector_memory(user, query: str, top_k: int = 6) -> List[Dict[str, str]]:
    """
    Retrieves similar past messages based on semantic relevance.
    Returns chat-style history list used directly by RAG + Agent.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        query_vec = model.encode(query).tolist()

        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        response = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter={"type":"conversation", "user":str(user.id)}
        )

        messages = []
        for match in response.get("matches", []):
            txt = match["metadata"]["text"]

            # split into user + assistant
            if "\nassistant:" in txt:
                u, a = txt.split("\nassistant:")
                u = u.replace("user:", "").strip()
                a = a.strip()

                messages.append({"role":"user","content":u})
                messages.append({"role":"assistant","content":a})

        return messages

    except Exception as e:
        print("❌ Failed retrieving conversation memory:", e)
        return []
