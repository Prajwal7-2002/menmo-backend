# rag/tools/langchain_tools.py
"""
LangChain tool adapters + ChatModel wrapper for Groq.
This module exposes:
 - WrappedLLM : ChatModel wrapper adapted to call rag/web/memory tools via LangChain Agent
 - Tool wrappers: rag, web, memory, rewrite, fallback
 - get_langchain_tools(user_obj) : returns list/dict of tools depending on LangChain availability

Behavioral policy (Option A — Strict Agentic RAG):
 - Prefer rag_search -> memory_search -> rewrite_query
 - Use web_search ONLY when RAG + memory cannot answer OR query explicitly asks for external/current facts
 - Agent MUST include tool outputs and always finish with "Final Answer: <answer>"
"""

from typing import Any, Dict, List
import json
import os
import re
import logging

logger = logging.getLogger(__name__)

# Local tools (import lazily/at runtime)
from rag.tools.rag_search import rag_search
from rag.tools.web_search import web_search_tool
from rag.tools.memory_search import memory_search_tool
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.fallback_llm import fallback_llm_tool
from rag.llm import call_llm_answer

# LangChain imports (optional)
try:
    from langchain.tools import Tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    # CRITICAL: Need these imports for the ChatModel return type
    from langchain_core.outputs import ChatResult, ChatGeneration
    LANGCHAIN_AVAILABLE = True
except Exception:
    Tool = None
    BaseChatModel = object
    HumanMessage = object
    SystemMessage = object
    AIMessage = object
    ChatResult = object
    ChatGeneration = object
    LANGCHAIN_AVAILABLE = False

# ---------------------------
# Utility: decide if web_search is likely needed
# ---------------------------
WEB_TRIGGERS = [
    r"\bwho is\b",
    r"\bwho was\b",
    r"\bcurrent\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\bnews\b",
    r"\blatest\b",
    r"\bupdated\b",
    r"\b202\d\b",  # years
    r"\b202[0-9]-\b",
    r"\bhow many\b",
    r"\blatest version\b",
    r"\bpopulation\b",
    r"\bpresident\b",
    r"\bprime minister\b",
    r"\bdeadline\b",
    r"\blatest release\b",
]

def _looks_like_external_query(q: str) -> bool:
    if not q:
        return False
    low = q.lower()
    for pat in WEB_TRIGGERS:
        if re.search(pat, low):
            return True
    # if user asked for URLs or "sources" prefer web
    if any(k in low for k in ("source", "sources", "url", "evidence", "cite", "citation")):
        return True
    # numeric/time-specific requests often need web/freshness
    if any(tok in low for tok in ("today", "yesterday", "tomorrow")):
        return True
    return False

# ====================================================================
# WrappedLLM — Proper ChatModel wrapper that calls call_llm_answer
# and injects a strict ReAct/system prompt to prefer RAG-first behavior.
# ====================================================================
class WrappedLLM(BaseChatModel):
    model_name = "groq_react_chat"

    @property
    def _llm_type(self):
        return "chat-groq"

    def _convert_messages(self, messages):
        """Convert LangChain messages & strings → Groq format."""
        groq_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                groq_msgs.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                groq_msgs.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_msgs.append({"role": "assistant", "content": msg.content})
            else:
                # Plain strings fallback → assume user input
                groq_msgs.append({"role": "user", "content": str(msg)})
        return groq_msgs

    def _inject_enforced_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepend an enforced system message that instructs the agent to:
         - Try rag_search first, then memory_search, then rewrite_query
         - Use web_search ONLY when rag+memory cannot answer or the query clearly requests fresh/external info
         - Always output tool call outputs, and end with Final Answer: ...
        """
        enforced = {
            "role": "system",
            "content": (
                "AGENT INSTRUCTIONS (Strict RAG-first):\n"
                "1) Try to answer using ONLY tool outputs in this order: rag_search -> memory_search -> rewrite_query.\n"
                "2) Use web_search ONLY if rag_search+memos are insufficient or if the query explicitly asks for current/external facts (dates, 'who is', 'today', 'latest', 'news', etc.).\n"
                "3) If you call tools, include the tool name and raw output in your reasoning trace. Example:\n"
                "   Thought: ...\n"
                "   Action: rag_search\n"
                "   Action Input: {\"query\": \"...\"}\n"
                "   Observation: <paste tool output here>\n"
                "4) If no tool is needed, respond with: Final Answer: <your answer>\n"
                "5) NEVER hallucinate. If not found, say: \"I don’t know based on the available documentation.\" or use fallback_llm.\n"
            )
        }
        return [enforced] + messages

    # ------------ REQUIRED: _generate() ------------
    def _generate(self, messages, stop=None, **kwargs):
        groq_msgs = self._convert_messages(messages)
        groq_msgs = self._inject_enforced_system(groq_msgs)

        # 2) Call backend LLM wrapper
        text = call_llm_answer(messages=groq_msgs, max_tokens=512) or "Final Answer: I don't know based on the available documentation."

        generation = ChatGeneration(
            message=AIMessage(content=text)
        )
        return ChatResult(generations=[generation])

    def invoke(self, input_data, **kwargs):
        if isinstance(input_data, dict):
            input_data = input_data.get("input", "")

        result = self.generate(
            messages=[[HumanMessage(content=input_data)]]
        )

        return result.generations[0].message.content

# ====================================================================
# Tool wrappers (Robust to parsing failures)
# Each wrapper adapts input and always returns simple safe output.
# ====================================================================

def tool_rag(input_str: str) -> Dict[str, Any]:
    try:
        payload = json.loads(input_str)
        return rag_search(
            query=payload.get("query", ""),
            user_id=payload.get("user_id"),
            domain=payload.get("domain"),
            document_id=payload.get("document_id"),
        )
    except Exception:
        # If raw string, pass it as query
        return rag_search(query=input_str, user_id=None, domain=None, document_id=None)

def tool_web(input_str: str) -> Any:
    """
    web_search_tool is available as a tool, but we prefer the agent decide when to call it.
    Keep wrapper simple — the agent prompt will restrict web usage.
    """
    try:
        payload = json.loads(input_str)
        q = payload.get("query") or payload.get("q") or ""
    except Exception:
        q = input_str
    return web_search_tool(q)

def tool_memory(input_str: str, user_obj=None) -> Any:
    try:
        payload = json.loads(input_str)
        q = payload.get("query", "")
    except Exception:
        q = input_str
    return memory_search_tool(user=user_obj, query=q)

def tool_rewrite(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return rewrite_query_tool(
            query=payload.get("query", ""),
            context_chunks=payload.get("context_chunks", [])
        )
    except Exception:
        return rewrite_query_tool(query=input_str, context_chunks=[])

def tool_fallback(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return fallback_llm_tool(
            query=payload.get("query", ""),
            context=payload.get("context", "")
        )
    except Exception:
        return fallback_llm_tool(query=input_str, context="")

# ====================================================================
# Tool registry
# - If LangChain not available we return a dict of callables (useful for tests)
# - If available we return list[Tool]
# ====================================================================
def get_langchain_tools(user_obj=None):
    if not LANGCHAIN_AVAILABLE:
        return {
            "rag_search": tool_rag,
            "web_search": tool_web,
            "memory_search": lambda q: tool_memory(q, user_obj=user_obj),
            "rewrite_query": tool_rewrite,
            "fallback_llm": tool_fallback,
        }

    return [
        Tool(name="rag_search", func=tool_rag,
             description="RAG retrieval. Input JSON or string."),
        Tool(name="web_search", func=tool_web,
             description="Search the web. (Agent should only call this when necessary)"),
        Tool(name="memory_search", func=lambda q: tool_memory(q, user_obj=user_obj),
             description="Search user memory."),
        Tool(name="rewrite_query", func=tool_rewrite,
             description="Rewrite query to improve retrieval."),
        Tool(name="fallback_llm", func=tool_fallback,
             description="Fallback direct answer."),
    ]
