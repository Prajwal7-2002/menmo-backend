# backend/rag/agent.py
import os
import logging
from typing import List, Dict, Any, Optional

from tavily import TavilyClient

# LangChain v1.x imports
try:
    # core agent creation
    from langchain.agents import create_react_agent, AgentExecutor
    # tools decorator
    from langchain.tools import tool
    # groq LLM binding
    from langchain_groq import ChatGroq
except Exception as _e:
    # we'll fallback later if imports fail
    create_react_agent = None
    AgentExecutor = None
    tool = None
    ChatGroq = None

# local imports (project)
from .pipeline import run_rag
from .memory import load_user_preferences

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------------
# Internet search tool (Tavily)
# ------------------------
def _internet_search_raw(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Search tool unavailable: missing TAVILY_API_KEY."
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=query, max_results=5)
        return str(res)
    except Exception as e:
        logger.exception("Tavily search failed")
        return f"Internet search failed: {e}"


# ------------------------
# RAG wrapper
# ------------------------
def _rag_call(query: str, user_id: Optional[int] = None, domain: Optional[str] = None,
              document_id: Optional[str] = None, mood: str = "neutral",
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


# ------------------------
# Heuristics
# ------------------------
def _looks_document_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    doc_keys = ["document", "pdf", "page", "section", "uploaded", "report", "appendix", "policy", "manual", "chapter"]
    for k in doc_keys:
        if k in q:
            return True
    if len(q.split()) <= 3 and q.endswith("?"):
        return True
    return False


def _looks_web_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    web_keys = ["latest", "news", "202", "update", "statute", "price", "rate", "current", "today"]
    for k in web_keys:
        if k in q:
            return True
    return False


# ------------------------
# LangChain tool wrappers (if LC available) or plain-call wrappers
# ------------------------
# We provide both: langchain tool decorator versions (if available) and plain functions.
if tool is not None:
    @tool
    def internet_search(query: str) -> str:
        return _internet_search_raw(query)

    @tool
    def rag_search(query: str, user_id: int = None, domain: str = None, document_id: str = None,
                   mood: str = "neutral", history: Optional[List[Dict[str, str]]] = None) -> str:
        res = _rag_call(query, user_id=user_id, domain=domain, document_id=document_id, mood=mood, history=history)
        # agent tools should return strings; include a short marker for confidence
        validated = res.get("validated", False)
        answer = res.get("answer", "") or ""
        if validated:
            return f"[RAG_VALIDATED]\n{answer}"
        # include fallback chunk if any
        chunks = res.get("chunks", [])
        fallback = (chunks[0]["text"] if chunks else "")
        return f"[RAG_LOW_RELEVANCE]\n{answer}\n\nTop chunk:\n{fallback}"
else:
    # fallback plain functions
    def internet_search(query: str) -> str:
        return _internet_search_raw(query)

    def rag_search(query: str, user_id: int = None, domain: str = None, document_id: str = None,
                   mood: str = "neutral", history: Optional[List[Dict[str, str]]] = None) -> str:
        res = _rag_call(query, user_id=user_id, domain=domain, document_id=document_id, mood=mood, history=history)
        if res.get("validated"):
            return res.get("answer", "")
        chunks = res.get("chunks", [])
        fallback = (chunks[0]["text"] if chunks else "")
        return (res.get("answer") or "") + ("\n\nTop chunk:\n" + fallback if fallback else "")


# ------------------------
# Build agent (LangChain-based if available)
# ------------------------
def build_agent(user_id: int = None,
                domain: Optional[str] = None,
                document_id: Optional[str] = None,
                mood: str = "neutral",
                history: Optional[List[Dict[str, str]]] = None):
    """
    Returns an agent-like object with .run(query) method.
    If LangChain agent creation is available in the environment, uses create_react_agent + AgentExecutor.
    Otherwise falls back to a lightweight in-file agent that follows the same decision logic.
    """

    # Prepare defaults and inject preferences
    # Prepare defaults and inject preferences
    prefs = {}
    try:
        # load_user_preferences expects a Django User object. If caller passed an ID,
        # try to import the user model and fetch the user instance, otherwise fall back.
        if user_id is None:
            prefs = {}
        else:
            try:
                # user_id might be an object already (User), or an int/str id
                from django.contrib.auth import get_user_model
                User = get_user_model()
                if hasattr(user_id, "id") or hasattr(user_id, "pk"):
                    user_obj = user_id
                else:
                    # try fetching from DB
                    user_obj = User.objects.filter(id=user_id).first()
                if user_obj:
                    prefs = load_user_preferences(user_obj) or {}
                else:
                    prefs = {}
            except Exception:
                # if anything goes wrong, keep prefs empty
                prefs = {}
    except Exception:
        prefs = {}


    default_mood = prefs.get("default_mood", mood)

    # If LangChain's create_react_agent is available, build an LC agent
    if create_react_agent is not None and AgentExecutor is not None and ChatGroq is not None:
        # Build the Groq LLM
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

        # Compose tools list (use the tool-wrapped functions if they exist)
        tools = []
        # rag_search and internet_search are functions or decorated tools depending on tool availability
        try:
            tools.append(rag_search if callable(rag_search) else rag_search)  # already a callable
            tools.append(internet_search if callable(internet_search) else internet_search)
        except Exception:
            # fallback to plain functions
            tools = [rag_search, internet_search]

        # Create a simple system instruction
        system_instruction = (
            "You are a document-grounded copilot. Use the RAG_Search tool for anything about user-uploaded documents. "
            "Use Internet_Search for web/time-sensitive queries. Keep answers concise. Respect requested mood/style."
            f" Default mood: {default_mood}."
        )

        try:
            agent_chain = create_react_agent(llm=llm, tools=tools, system_message=system_instruction)
            executor = AgentExecutor(agent_chain=agent_chain)
        except Exception as e:
            logger.exception("LangChain agent creation failed, falling back to local agent.")
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
                    # Build context and run via executor
                    try:
                        # Pass structured input so tools that need user_id/domain can use it.
                        inputs = {
                            "input": query,
                            "user_id": self.user_id,
                            "domain": self.domain,
                            "document_id": self.document_id,
                            "mood": self.mood,
                            "history": self.history,
                        }
                        out = self.executor.run(inputs)
                        # AgentExecutor.run may return dict or str; normalize
                        return out if isinstance(out, str) else str(out)
                    except Exception as e:
                        logger.exception("LCAgent run failed")
                        return f"Agent error: {e}"

            return LCAgent(executor, user_id, domain, document_id, default_mood, history)

    # ------------------------
    # Fallback: lightweight in-file agent (same decision logic)
    # ------------------------
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

            try:
                # If document id provided or query looks like document question -> RAG first
                if self.document_id or _looks_document_query(q):
                    rag_res = _rag_call(q, user_id=self.user_id, domain=self.domain,
                                        document_id=self.document_id, mood=self.mood, history=self.history)
                    if rag_res.get("validated"):
                        return rag_res.get("answer") or "No answer found in documents."
                    # fallback to web
                    web = _internet_search_raw(q)
                    chunks = rag_res.get("chunks", [])
                    fallback_text = chunks[0]["text"] if chunks else ""
                    return (
                        f"{web}\n\n(Also searched your docs but found low relevance; top doc snippet below)\n\n{fallback_text}"
                    )

                # If web query -> internet search
                if _looks_web_query(q):
                    web = _internet_search_raw(q)
                    rag_res = _rag_call(q, user_id=self.user_id, domain=self.domain,
                                        document_id=self.document_id, mood=self.mood, history=self.history)
                    if rag_res.get("validated"):
                        return f"{web}\n\n(Also found relevant info in your documents):\n\n{rag_res.get('answer')}"
                    return web

                # Default: RAG first, fallback to web
                rag_res = _rag_call(q, user_id=self.user_id, domain=self.domain,
                                    document_id=self.document_id, mood=self.mood, history=self.history)
                if rag_res.get("validated"):
                    return rag_res.get("answer") or "No answer found in documents."
                web = _internet_search_raw(q)
                return f"{web}\n\n(Your docs had low relevance — returned web result.)"
            except Exception as e:
                logger.exception("SimpleAgent run failed")
                return f"Agent error: {e}"

    return SimpleAgent(user_id, domain, document_id, default_mood, history)
