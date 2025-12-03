# rag/tools/langchain_tools.py

"""
LangChain tool wrappers + LLM adapter for Groq.
Fully compatible with LangChain 0.2.x and your ReAct agent.
"""

from typing import Any, Dict, List
import json
import os

# Local tools
from rag.tools.rag_search import rag_search
from rag.tools.web_search import web_search_tool
from rag.tools.memory_search import memory_search_tool
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.fallback_llm import fallback_llm_tool
from rag.llm import call_llm_answer

# ----------------------------------------------------------
# LangChain imports (safe fallback if missing)
# ----------------------------------------------------------
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


# ====================================================================
# 🧠 1. WrappedLLM — CORRECT ChatModel for LC 0.2.x + Groq
# ====================================================================

class WrappedLLM(BaseChatModel):
    """
    Wrapper for Groq API so LangChain can use it as a ChatModel.
    Provides:
    - correct _generate() returning ChatResult
    - correct message conversion
    - correct invoke() handling
    """

    model_name = "groq_react_chat"

    @property
    def _llm_type(self) -> str:
        return "chat-groq"

    def _convert_messages(self, messages):
        """
        Convert LangChain message types → Groq dict format.
        """
        groq_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                groq_msgs.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                groq_msgs.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                groq_msgs.append({"role": "assistant", "content": msg.content})
            else:
                groq_msgs.append({"role": "user", "content": str(msg)})
        return groq_msgs

    # ------------ CRITICAL: Correct _generate() implementation ------------
    def _generate(self, messages, stop=None, **kwargs):
        """
        LangChain ReAct agent calls _generate() via .generate().
        We wrap Groq output into ChatResult → ChatGeneration.
        """

        groq_msgs = self._convert_messages(messages)

        # Enforce ReAct format
        groq_msgs.insert(0, {
            "role": "system",
            "content": (
                "FOLLOW THIS EXACT FORMAT:\n"
                "Thought: <reason>\n"
                "Action: <tool>\n"
                "Action Input: <input>\n\n"
                "If no tool needed:\n"
                "Final Answer: <answer>"
            )
        })

        # Call Groq through your wrapper
        text = call_llm_answer(messages=groq_msgs, max_tokens=512)
        if not text:
            text = "Final Answer: I don't know."

        # Wrap Groq output into LangChain result
        generation = ChatGeneration(
            message=AIMessage(content=text)
        )
        return ChatResult(generations=[generation])

    # ------------ CRITICAL: invoke() wrapper (used by AgentExecutor) ------
    def invoke(self, input_data, **kwargs):
        if isinstance(input_data, dict):
            input_data = input_data.get("input", "")

        result = self.generate(messages=[[HumanMessage(content=input_data)]])

        # Return only answer text
        return result.generations[0].message.content



# ====================================================================
# 🧰 2. Tool wrappers (all safe, no parsing failures)
# ====================================================================

def tool_rag(input_str: str) -> Dict[str, Any]:
    try:
        payload = json.loads(input_str)
        return rag_search(
            query=payload.get("query", input_str),
            user_id=payload.get("user_id"),
            domain=payload.get("domain"),
            document_id=payload.get("document_id"),
        )
    except Exception:
        return rag_search(query=input_str, user_id=None, domain=None, document_id=None)


def tool_web(input_str: str) -> Any:
    return web_search_tool(input_str)


def tool_memory(input_str: str, user_obj=None) -> Any:
    return memory_search_tool(user=user_obj, query=input_str)


def tool_rewrite(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return rewrite_query_tool(
            query=payload.get("query", input_str),
            context_chunks=payload.get("context_chunks", [])
        )
    except Exception:
        return rewrite_query_tool(query=input_str, context_chunks=[])


def tool_fallback(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return fallback_llm_tool(
            query=payload.get("query", input_str),
            context=payload.get("context", "")
        )
    except Exception:
        return fallback_llm_tool(query=input_str, context="")


# ====================================================================
# 🧰 3. Tool registry (for agent)
# ====================================================================

def get_langchain_tools(user_obj=None):
    """
    Returns real LC Tool objects if LC available.
    Returns python callables otherwise.
    """

    if not LANGCHAIN_AVAILABLE:
        return {
            "rag_search": tool_rag,
            "web_search": tool_web,
            "memory_search": lambda q: tool_memory(q, user_obj=user_obj),
            "rewrite_query": tool_rewrite,
            "fallback_llm": tool_fallback,
        }

    return [
        Tool(
            name="rag_search",
            func=tool_rag,
            description="RAG retrieval. Input JSON or string."
        ),
        Tool(
            name="web_search",
            func=tool_web,
            description="Search the web."
        ),
        Tool(
            name="memory_search",
            func=lambda q: tool_memory(q, user_obj=user_obj),
            description="Search user memory."
        ),
        Tool(
            name="rewrite_query",
            func=tool_rewrite,
            description="Rewrite user query for clarity."
        ),
        Tool(
            name="fallback_llm",
            func=tool_fallback,
            description="Fallback direct answer."
        ),
    ]
