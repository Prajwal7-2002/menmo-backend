# rag/tools/langchain_tools.py
"""
LangChain tool wrappers and WrappedLLM.

Key changes:
- tool_rag now expects/forwards domain and document_id from JSON payload.
- tool_memory and tool_web accept JSON payload strings and forward domain/document_id.
- get_langchain_tools builds LangChain Tool objects as before but tools expect JSON input with domain/document_id.
- WrappedLLM._generate ensures system prompt enforces RAG-first policy.
"""

from typing import Any, Dict, List
import json
import logging

logger = logging.getLogger(__name__)

from rag.tools.rag_search import rag_search
from rag.tools.web_search import web_search_tool
from rag.tools.memory_search import memory_search_tool
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.fallback_llm import fallback_llm_tool
from rag.llm import call_llm_answer
from rag.tools.evaluate_context import evaluate_context_tool

# Try imports; degrade gracefully
try:
    from langchain.tools import Tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    LANGCHAIN_AVAILABLE = True
except Exception:
    Tool = None
    BaseChatModel = object
    HumanMessage = SystemMessage = AIMessage = object
    ChatResult = ChatGeneration = object
    LANGCHAIN_AVAILABLE = False

# ---------------- Wrapped LLM ----------------
class WrappedLLM(BaseChatModel):
    model_name = "groq_react_chat"

    @property
    def _llm_type(self):
        return "chat-groq"

    def _convert_messages(self, messages):
        result = []
        for m in messages:
            # Messages may be dict-like or langchain message types
            try:
                mrole = getattr(m, "type", None) or getattr(m, "role", None)
                content = getattr(m, "content", None) or str(m)
                if mrole == "system" or isinstance(m, SystemMessage):
                    result.append({"role": "system", "content": content})
                elif mrole == "assistant" or isinstance(m, AIMessage):
                    result.append({"role": "assistant", "content": content})
                else:
                    result.append({"role": "user", "content": content})
            except Exception:
                result.append({"role": "user", "content": str(m)})
        return result

    def _generate(self, messages, stop=None, **kwargs):
        msgs = self._convert_messages(messages)
        # Enforce RAG-first guard
        msgs.insert(0, {
            "role": "system",
            "content": (
                "You are an agent that MUST prefer internal document evidence and memory. "
                "Use tools for retrieval and rewriting. "
                "Only use web_search when internal sources are insufficient and you explicitly state why. "
                "Format tool calls exactly as: Action: <tool_name>\\nAction Input: <JSON string or plain text>\\n"
                "When done, output: Final Answer: <answer>"
            )
        })

        text = call_llm_answer(messages=msgs, max_tokens=512) or "Final Answer: I don’t know."
        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])

    def invoke(self, input_data, **kwargs):
        if isinstance(input_data, dict):
            input_data = input_data.get("input", "")
        res = self.generate(messages=[[HumanMessage(content=input_data)]])
        return res.generations[0].message.content


# ---------------- Tool wrappers ----------------

def tool_rag(input_str: str) -> Dict[str, Any]:
    """
    input_str should be JSON with fields:
      { "query": "...", "user_id": ..., "domain": "...", "document_id": "..." }
    Returns rag_search(...) result (dict).
    """
    try:
        payload = json.loads(input_str)
        return rag_search(
            query=payload.get("query", ""),
            user_id=payload.get("user_id"),
            domain=payload.get("domain"),
            document_id=payload.get("document_id"),
        )
    except Exception:
        # fallback: treat input as raw query (no domain/document)
        return rag_search(input_str, None, None, None)


def tool_memory(input_str: str, user_obj=None) -> Any:
    """
    Expects JSON {"query": "...", "domain": "...", "document_id": "..."} or raw query.
    """
    try:
        payload = json.loads(input_str)
        return memory_search_tool(user=user_obj, query=payload.get("query", ""), domain=payload.get("domain"), document_id=payload.get("document_id"))
    except Exception:
        return memory_search_tool(user=user_obj, query=input_str, domain=None, document_id=None)


def tool_rewrite(input_str: str) -> str:
    """
    Expects JSON {"query": "...", "context_chunks": [...], "domain": "...", "document_id": "..."} or raw query.
    """
    try:
        payload = json.loads(input_str)
        return rewrite_query_tool(
            query=payload.get("query", ""),
            context_chunks=payload.get("context_chunks", []),
            domain=payload.get("domain"),
            document_id=payload.get("document_id"),
        )
    except Exception:
        return rewrite_query_tool(query=input_str, context_chunks=[])


def tool_fallback(input_str: str) -> str:
    """
    Expects JSON {"query": "...", "context": "...", "domain":"...", "document_id":"..."} or raw query.
    """
    try:
        payload = json.loads(input_str)
        return fallback_llm_tool(query=payload.get("query", ""), context=payload.get("context", ""), domain=payload.get("domain"), document_id=payload.get("document_id"))
    except Exception:
        return fallback_llm_tool(query=input_str, context="", domain=None, document_id=None)


def tool_web(input_str: str) -> Any:
    # expects JSON {"query": "..."} or raw
    try:
        payload = json.loads(input_str)
        return web_search_tool(payload.get("query", ""))
    except Exception:
        return web_search_tool(input_str)


def tool_evaluate(input_str: str) -> Dict[str, Any]:
    """
    Accepts JSON list of chunks or a JSON string: {"chunks": [...]}
    Returns evaluate_context_tool output.
    """
    try:
        p = json.loads(input_str)
        chunks = p if isinstance(p, list) else p.get("chunks", []) or []
    except Exception:
        return {"quality": "empty", "score": 0.0, "reason": "bad_input"}
    return evaluate_context_tool(chunks)


# ---------------- Registry builder (allow_web control) ----------------

def get_langchain_tools(user_obj=None, allow_web: bool = True):
    """
    Returns LangChain Tool list or fallback dict depending on availability.
    Tools expect JSON strings as input so agent can include domain/document_id.
    """
    if not LANGCHAIN_AVAILABLE:
        tools = {
            "rag_search": tool_rag,
            "memory_search": lambda q: tool_memory(q, user_obj),
            "rewrite_query": tool_rewrite,
            "fallback_llm": tool_fallback,
            "evaluate_context": tool_evaluate,
        }
        if allow_web:
            tools["web_search"] = tool_web
        return tools

    tool_list = [
        Tool(name="rag_search", func=tool_rag, description="Primary RAG retrieval. Input JSON: {query,user_id,domain,document_id}"),
        Tool(name="memory_search", func=lambda q: tool_memory(q, user_obj), description="User memory lookup. Input JSON {query,domain,document_id}"),
        Tool(name="rewrite_query", func=tool_rewrite, description="Rewrite a bad query for better retrieval. Input JSON {query,context_chunks,domain,document_id}"),
        Tool(name="fallback_llm", func=tool_fallback, description="Safe fallback LLM that respects context. Input JSON {query,context,domain,document_id}"),
        Tool(name="evaluate_context", func=tool_evaluate, description="Return quality/score for retrieved chunks. Input JSON {chunks: [...]}"),
    ]
    if allow_web:
        tool_list.append(Tool(name="web_search", func=tool_web, description="Search the web (agent decides). Input: JSON {query}"))
    return tool_list
