# rag/tools/langchain_tools.py
from typing import Any, Dict, List
import json

from rag.tools.rag_search import rag_search
from rag.tools.web_search import web_search_tool
from rag.tools.memory_search import memory_search_tool
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.fallback_llm import fallback_llm_tool
from rag.llm import call_llm_answer

# Try to import LangChain types; degrade gracefully if unavailable
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


class WrappedLLM(BaseChatModel):
    """
    Controlled LLM adapter used by the React agent.
    Converts LangChain messages to our call_llm_answer format and returns ChatResult.
    """
    model_name = "groq_react_chat"

    @property
    def _llm_type(self):
        return "chat-groq"

    def _convert_messages(self, messages):
        result = []
        for m in messages:
            if isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
            else:
                result.append({"role": "user", "content": str(m)})
        return result

    def _generate(self, messages, stop=None, **kwargs):
        msgs = self._convert_messages(messages)
        # Add a lightweight system guard to encourage tool usage or final answer
        msgs.insert(0, {
            "role": "system",
            "content": (
                "You may call tools when helpful. "
                "If you choose not to call a tool, output 'Final Answer: <your answer>'. "
                "If you call a tool, follow the format: Action: <tool_name>\\nAction Input: <input>"
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


# --- Tool wrappers (defensive) ---
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
        return rag_search(input_str, None, None, None)


def tool_memory(input_str: str, user_obj=None) -> Any:
    return memory_search_tool(user=user_obj, query=input_str)


def tool_rewrite(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return rewrite_query_tool(
            query=payload.get("query", ""),
            context_chunks=payload.get("context_chunks", []),
        )
    except Exception:
        return rewrite_query_tool(query=input_str, context_chunks=[])


def tool_fallback(input_str: str) -> str:
    try:
        payload = json.loads(input_str)
        return fallback_llm_tool(query=payload.get("query", ""), context=payload.get("context", ""))
    except Exception:
        return fallback_llm_tool(query=input_str, context="")


def tool_web(input_str: str) -> Any:
    # web_search_tool should be defensive by itself
    return web_search_tool(input_str)


# --- Registry builder; caller can control web availability ---
def get_langchain_tools(user_obj=None, allow_web: bool = True):
    """
    Returns either a list of Tool objects (if LangChain is installed) or a fallback dict of functions.
    allow_web: if False -> web_search is omitted (useful for offline / strict environments)
    """
    if not LANGCHAIN_AVAILABLE:
        tools = {
            "rag_search": tool_rag,
            "memory_search": lambda q: tool_memory(q, user_obj),
            "rewrite_query": tool_rewrite,
            "fallback_llm": tool_fallback,
        }
        if allow_web:
            tools["web_search"] = tool_web
        return tools

    # Build LangChain Tool list
    tool_list = [
        Tool(name="rag_search", func=tool_rag, description="Primary RAG retrieval."),
        Tool(name="memory_search", func=lambda q: tool_memory(q, user_obj), description="User memory lookup."),
        Tool(name="rewrite_query", func=tool_rewrite, description="Rewrite a bad query for retrieval."),
        Tool(name="fallback_llm", func=tool_fallback, description="Safe fallback LLM; prefers context.")
    ]
    if allow_web:
        tool_list.append(Tool(name="web_search", func=tool_web, description="Search the web (agent decides when)."))
    return tool_list
