# rag/agent.py
import os
import logging
from typing import List, Dict, Any, Optional

from duckduckgo_search import DDGS  # 🔁 instead of Tavily

# LangChain imports
try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import tool
    from langchain_groq import ChatGroq
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
except Exception:
    create_react_agent = None
    AgentExecutor = None
    tool = None
    ChatGroq = None
    DuckDuckGoSearchAPIWrapper = None

from .pipeline import run_rag


logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------------ DuckDuckGo search ------------------------ #

def _internet_search_raw(query: str) -> str:
    """Plain DDG search without LangChain (used by fallback agent)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results found via DuckDuckGo."
        # Compact string summary
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            url = r.get("href", "")
            lines.append(f"- {title}\n  {body}\n  {url}")
        return "DuckDuckGo results:\n\n" + "\n\n".join(lines)
    except Exception as e:
        logger.exception("DuckDuckGo search failed")
        return f"Internet search failed: {e}"


def _rag_call(query: str,
              user_id: Optional[int] = None,
              domain: Optional[str] = None,
              document_id: Optional[str] = None,
              mood: str = "neutral",
              history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    try:
        return run_rag(
            query=query,
            user_id=user_id,
            domain=domain,
            document_id=document_id,
            mood=mood,
            history=history,
        )
    except Exception as e:
        logger.exception("run_rag failed")
        return {"answer": f"RAG failed: {e}", "validated": False, "chunks": [], "reason": "exception"}


def _looks_document_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    doc_keys = ["document", "pdf", "page", "section", "uploaded", "report", "appendix", "policy", "manual", "chapter"]
    return any(k in q for k in doc_keys) or (len(q.split()) <= 3 and q.endswith("?"))


def _looks_web_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    web_keys = ["latest", "news", "202", "update", "statute", "price", "rate", "current", "today"]
    return any(k in q for k in web_keys)


# ---------------- LangChain Tools ---------------- #

if tool is not None and DuckDuckGoSearchAPIWrapper is not None:
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

    @tool
    def internet_search(query: str) -> str:
        """Use DuckDuckGo to search the web for up-to-date information."""
        try:
            return ddg_wrapper.run(query)
        except Exception:
            # fallback to raw helper
            return _internet_search_raw(query)

    @tool
    def rag_search(query: str,
                   user_id: int = None,
                   domain: str = None,
                   document_id: str = None,
                   mood: str = "neutral",
                   history: Optional[List[Dict[str, str]]] = None) -> str:
        """Use RAG over user documents to answer document-grounded questions."""
        res = _rag_call(query, user_id=user_id, domain=domain,
                        document_id=document_id, mood=mood, history=history)
        validated = res.get("validated", False)
        answer = res.get("answer", "") or ""
        if validated:
            return f"[RAG_VALIDATED]\n{answer}"
        chunks = res.get("chunks", [])
        fb = chunks[0]["text"] if chunks else ""
        return f"[RAG_LOW_RELEVANCE]\n{answer}\n\nTop chunk:\n{fb}"
else:
    # fallback simple callables
    def internet_search(query: str) -> str:
        return _internet_search_raw(query)

    def rag_search(query: str,
                   user_id: int = None,
                   domain: str = None,
                   document_id: str = None,
                   mood: str = "neutral",
                   history: Optional[List[Dict[str, str]]] = None) -> str:
        res = _rag_call(query, user_id=user_id, domain=domain,
                        document_id=document_id, mood=mood, history=history)
        if res.get("validated"):
            return res.get("answer", "")
        chunks = res.get("chunks", [])
        fb = chunks[0]["text"] if chunks else ""
        return (res.get("answer") or "") + ("\n\nTop chunk:\n" + fb if fb else "")


# ---------------- Build Agent ---------------- #

def build_agent(user_id: int = None,
                domain: Optional[str] = None,
                document_id: Optional[str] = None,
                mood: str = "neutral",
                history: Optional[List[Dict[str, str]]] = None):
    """Return an object with .run(query) – LangChain agent if available; fallback otherwise."""
    # Load preferences if possible
    try:
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user_obj = None
        if user_id:
            user_obj = User.objects.filter(id=user_id).first()
        prefs = load_user_preferences(user_obj) if user_obj else {}
    except Exception:
        prefs = {}

    default_mood = prefs.get("default_mood", mood)

    # If we have LangChain & Groq bindings, use full agent
    if create_react_agent and AgentExecutor and ChatGroq and DuckDuckGoSearchAPIWrapper:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

        tools = [rag_search, internet_search]

        system_instruction = (
            "You are a document-grounded copilot. "
            "Use the rag_search tool for queries about user-uploaded documents. "
            "Use the internet_search tool for web/time-sensitive queries. "
            f"Default mood: {default_mood}. Keep answers concise and grounded."
        )

        try:
            agent_chain = create_react_agent(llm=llm, tools=tools, system_message=system_instruction)
            executor = AgentExecutor(agent_chain=agent_chain)
        except Exception as e:
            logger.exception("LangChain agent creation failed, falling back to SimpleAgent.")
            executor = None

        if executor is not None:
            class LCAgent:
                def __init__(self, executor, user_id, domain, document_id, mood, history):
                    self.executor = executor
                    self.user_id = user_id
                    self.domain = domain
                    self.document_id = document_id
                    self.mood = mood
                    self.history = history or []

                def run(self, query: str) -> str:
                    try:
                        inputs = {
                            "input": query,
                            "user_id": self.user_id,
                            "domain": self.domain,
                            "document_id": self.document_id,
                            "mood": self.mood,
                            "history": self.history,
                        }
                        out = self.executor.run(inputs)
                        return out if isinstance(out, str) else str(out)
                    except Exception as e:
                        logger.exception("LCAgent.run failed")
                        return f"Agent error: {e}"

            return LCAgent(executor, user_id, domain, document_id, default_mood, history)

    # Fallback: simple, non-LangChain agent
    class SimpleAgent:
        def __init__(self, user_id, domain, document_id, mood, history):
            self.user_id = user_id
            self.domain = domain
            self.document_id = document_id
            self.mood = mood
            self.history = history or []

        def run(self, query: str) -> str:
            q = (query or "").strip()
            if not q:
                return "No query provided."

            # --- 1) Try RAG first always ---
            rag_res = _rag_call(q, user_id=self.user_id, domain=self.domain,
                                document_id=self.document_id, mood=self.mood, history=self.history)

            if rag_res.get("validated"):
                return rag_res.get("answer") or "No answer found in documents."

            # --- 2) If RAG failed → try web ---
            web = _internet_search_raw(q)

            if "No web results" not in web:
                return f"{web}\n\n(No strong evidence from documents, using web search.)"

            # --- 3) If RAG + Web both fail → GENERIC LLM FALLBACK 🚀 ---
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system","content":"You are an intelligent copilot. Use reasoning."},
                        {"role": "user","content": q}
                    ]
                )
                return completion.choices[0].message.content
            except Exception:
                return "I could not find the answer, even with fallback LLM."
