# ========================= rag/agent.py ========================= #

import os
import logging
from typing import List, Dict, Any, Optional
from ddgs import DDGS
from .pipeline import run_rag

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------- DuckDuckGo Search ---------------------- #

def _internet_search_raw(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results found."
        return "\n\n".join(f"- {r['title']}\n  {r['body']}" for r in results)
    except Exception as e:
        return f"Search failed: {e}"


# ----------------------- RAG Wrapper ---------------------------- #

def _rag_call(query, user_id=None, domain=None, document_id=None, mood="neutral", history=None):
    try:
        return run_rag(query=query, user_id=user_id, domain=domain,
                       document_id=document_id, mood=mood, history=history)
    except Exception as e:
        return {"answer": f"RAG error: {e}", "validated": False}


# ------------------- Simple Agent (your default) ---------------- #

class SimpleAgent:
    """
    1) Try RAG (local docs)
    2) If agent_mode OFF = stop
    3) If agent_mode ON => search Web
    4) If still weak => Groq LLM reasoning fallback
    """

    def __init__(self, user_id, domain, document_id, mood, history, enabled=False):
        self.user_id = user_id
        self.domain = domain
        self.document_id = document_id
        self.mood = mood
        self.history = history or []
        self.enabled = enabled    # 🔥 controls agent mode

    def run(self, query: str) -> str:

        # Step 1 — RAG lookup
        rag = _rag_call(query, self.user_id, self.domain, self.document_id, self.mood, self.history)

        if rag.get("validated") and not self.enabled:   # only block local if agent off
            return rag["answer"]


        # Step 2 — If agent disabled STOP EARLY
        if not self.enabled:
            return "❗ No match in documents. Agent Mode disabled."

        # Step 3 — Web search
        web = _internet_search_raw(query)
        if "No web" not in web:
            return web + "\n\n🌐 Web Search Result (Agent Mode)"

        # Step 4 — Groq LLM fallback
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            out = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Answer using external knowledge."},
                    {"role": "user", "content": query}
                ]
            )
            return out.choices[0].message.content + "\n\n🧠 LLM Reasoning Fallback"
        except:
            return "❌ Agent could not answer."


# ------------------------ REQUIRED FIX ------------------------- #

def build_agent(user_id=None, domain=None, document_id=None,
                mood="neutral", history=None, agent_enabled=False):
    """
    ALWAYS return SimpleAgent now. (Stable mode)
    """
    return SimpleAgent(
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        mood=mood,
        history=history,
        enabled=agent_enabled
    )
