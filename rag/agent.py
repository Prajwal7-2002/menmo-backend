# rag/agent.py
"""
Agent orchestration for Neurostack.

Key changes:
- AgenticRAG accepts and stores domain and document_id and forwards them to tools.
- When running manual tool loop, all tool calls include domain & document_id in payloads.
- Safe fallbacks if LangChain is unavailable.
- Agent output normalization kept.
"""

from typing import Dict, Any, List, Optional
import logging
import json
import time

logger = logging.getLogger(__name__)

# Try to import langchain agent utilities; degrade gracefully.
try:
    from langchain.agents.react.base import create_react_agent
    from langchain.agents import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Local tool wrappers
try:
    from rag.tools.langchain_tools import (
        WrappedLLM,
        get_langchain_tools,
        tool_rag,
        tool_rewrite,
        tool_evaluate,
        tool_fallback,
        tool_memory,
        tool_web,
    )
except Exception:
    # If the module can't be imported for some reason, define safe no-op placeholders
    def tool_rag(payload):
        return {"chunks": [], "found": False}

    def tool_rewrite(payload):
        return payload if isinstance(payload, str) else ""

    def tool_evaluate(payload):
        return {"quality": "empty", "score": 0.0}

    def tool_fallback(payload):
        return "I don’t know based on the available documentation."

    def tool_memory(payload, user_obj=None):
        return []

    def tool_web(payload):
        return {}

    class WrappedLLM:
        pass

    def get_langchain_tools(user_obj=None, allow_web: bool = True):
        return {}

# Django user fallback
try:
    from django.contrib.auth import get_user_model
    User = get_user_model()
except Exception:
    User = None

SYSTEM_PROMPT = """
You are NeuroStack Agent. Your job is to answer user queries using internal documents (RAG) and memory.
You must attempt to retrieve evidence and only use the web if internal sources are insufficient.
If unsure after retries, respond conservatively: "I don’t know based on the available documentation."
When using tools follow the specified tool format. Final output must contain 'Final Answer:'
"""

class AgenticRAG:
    """
    Orchestrates RAG + optional LangChain agent.
    Important: domain and document_id are stored and forwarded to tools to avoid cross-document leakage.
    """

    def __init__(
        self,
        user_id: Optional[int] = None,
        user_obj=None,
        allow_web: bool = True,
        max_retries: int = 3,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        agent_enabled: bool = True,
    ):
        self.user_id = user_id
        self.user = user_obj
        self.allow_web = allow_web
        self.max_retries = max_retries or 1
        self.domain = domain
        self.document_id = document_id
        self.agent_enabled = agent_enabled
        self._executor = None

        if LANGCHAIN_AVAILABLE and agent_enabled:
            try:
                self._build_agent()
            except Exception as e:
                logger.exception("Failed to build LangChain agent: %s", e)
                self._executor = None

    def _build_agent(self):
        # Create a WrappedLLM and tools; ensure tools created receive user_obj and allow_web
        llm = WrappedLLM()
        tools = get_langchain_tools(user_obj=self.user, allow_web=self.allow_web)
        tool_names = ", ".join(t.name for t in tools) if isinstance(tools, list) else ", ".join(tools.keys())
        tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools) if isinstance(tools, list) else "\n".join(f"- {k}" for k in tools.keys())

        from langchain_core.prompts import PromptTemplate

        REACT_PROMPT = PromptTemplate.from_template("""
{system_prompt}

You have access to tools:
{tools}

Tool names: {tool_names}

Use:
Thought: <reason>
Action: <tool_name>
Action Input: <input>

Observation: <tool output>

If finished:
Final Answer: <your answer>

Question: {input}
{agent_scratchpad}
""")

        prompt = REACT_PROMPT.partial(system_prompt=SYSTEM_PROMPT, tools=tools_block, tool_names=tool_names)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
            max_execution_time=90.0
        )

    def _extract_final_answer(self, raw: str) -> str:
        if not raw:
            return ""
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()
        # fallback last non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()

    def _rag_tool_call(self, query: str) -> List[Dict[str, Any]]:
        """
        Wrap tool_rag with domain/document_id/user_id context consistently.
        """
        payload = {
            "query": query,
            "user_id": self.user_id,
            "domain": self.domain,
            "document_id": self.document_id,
        }
        try:
            res = tool_rag(json.dumps(payload))
            # tool_rag may return dict with 'chunks' or list; normalize
            if isinstance(res, dict):
                return res.get("chunks", []) or []
            elif isinstance(res, list):
                return res
            else:
                return []
        except Exception as e:
            logger.exception("tool_rag() failed: %s", e)
            return []

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        High-level orchestration. Prefer internal docs; rewrite / rerank / memory / web fallback.
        """
        # If LangChain executor available, hand off but ensure tools see the domain/document context:
        if self._executor:
            try:
                # Pass a wrapped input containing domain/document metadata so tools can pick it up
                wrapped_input = json.dumps({
                    "input": user_query,
                    "user_id": self.user_id,
                    "domain": self.domain,
                    "document_id": self.document_id,
                })
                res = self._executor.invoke({"input": wrapped_input})
                raw = res.get("output", "") or ""
                final = self._extract_final_answer(raw)
                return {
                    "answer": final or "I don’t know based on the available documentation.",
                    "mode": "agent",
                    "confidence": 1.0,
                    "chunks": [],
                    "trace": [{"tool": "langchain_agent", "result": raw}],
                    "steps": 1
                }
            except Exception as e:
                logger.exception("Agent execution error: %s", e)
                # fallthrough to manual loop

        # Manual lightweight agent loop if no executor or executor failed:
        refined_query = user_query
        trace: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_retries) + 1):
            # Optionally rewrite on second+ attempts or if query is too short
            if attempt > 1 or len(refined_query.split()) < 3:
                try:
                    rewrite_payload = json.dumps({"query": refined_query, "context_chunks": [] , "domain": self.domain, "document_id": self.document_id})
                    new_q = tool_rewrite(rewrite_payload)
                    if new_q and isinstance(new_q, str) and len(new_q) > 1:
                        refined_query = new_q
                        trace.append({"step": "rewrite", "attempt": attempt, "result": refined_query})
                except Exception as e:
                    logger.debug("rewrite failed: %s", e)

            # RAG retrieval
            try:
                chunks = self._rag_tool_call(refined_query) or []
            except Exception as e:
                logger.exception("rag retrieval failed: %s", e)
                chunks = []

            trace.append({"step": "rag", "attempt": attempt, "chunks_count": len(chunks)})

            # Evaluate retrieved chunks quality
            try:
                eval_input = json.dumps({"chunks": chunks})
                eval_res = tool_evaluate(eval_input)
            except Exception as e:
                logger.exception("evaluate failed: %s", e)
                eval_res = {"quality": "empty", "score": 0.0}

            trace.append({"step": "evaluate", "attempt": attempt, "eval": eval_res})

            if eval_res.get("quality") == "good":
                # produce final answer using fallback LLM with context
                try:
                    top_context = "\n\n---\n\n".join(c.get("text", "") for c in (chunks or [])[:4])
                    fallback_payload = json.dumps({"query": user_query, "context": top_context, "domain": self.domain, "document_id": self.document_id})
                    final = tool_fallback(fallback_payload)
                except Exception as e:
                    logger.exception("fallback generation failed: %s", e)
                    final = "I don’t know based on the available documentation."
                trace.append({"step": "finalized", "attempt": attempt})
                return {"answer": final, "mode": "agent", "confidence": float(eval_res.get("score", 1.0)), "chunks": chunks, "trace": trace, "steps": attempt}

            # Try memory search once per attempt
            try:
                mem_payload = json.dumps({"query": refined_query, "domain": self.domain, "document_id": self.document_id})
                mem = tool_memory(mem_payload, user_obj=self.user)
                if mem:
                    trace.append({"step": "memory_search", "attempt": attempt, "result_count": len(mem) if isinstance(mem, list) else 1})
                    mem_eval = tool_evaluate(json.dumps({"chunks": mem if isinstance(mem, list) else [mem]}))
                    if mem_eval.get("quality") == "good":
                        top_context = "\n\n---\n\n".join((mem or [])[:4] if isinstance(mem, list) else [str(mem)])
                        final = tool_fallback(json.dumps({"query": user_query, "context": top_context}))
                        return {"answer": final, "mode": "agent", "confidence": float(mem_eval.get("score", 1.0)), "chunks": mem if isinstance(mem, list) else [], "trace": trace, "steps": attempt}
            except Exception:
                logger.debug("memory path failed (continuing)")

            # If allowed and last attempt, try web as last resort
            if self.allow_web and attempt == self.max_retries:
                try:
                    web_res = tool_web(json.dumps({"query": refined_query}))
                    trace.append({"step": "web_search", "attempt": attempt, "result": "ok"})
                    final = tool_fallback(json.dumps({"query": user_query, "context": json.dumps(web_res)}))
                    return {"answer": final, "mode": "agent_web", "confidence": 0.5, "chunks": [], "trace": trace, "steps": attempt}
                except Exception:
                    logger.debug("web search failed")

            # Otherwise ask rewrite tool for a stronger query and retry
            try:
                rewrite_payload = json.dumps({"query": refined_query, "context_chunks": chunks or [], "domain": self.domain, "document_id": self.document_id})
                refined_query = tool_rewrite(rewrite_payload) or refined_query
                trace.append({"step": "rewrite_retry", "attempt": attempt, "new_query": refined_query})
            except Exception:
                trace.append({"step": "rewrite_failed", "attempt": attempt})

            time.sleep(0.2)

        # Exhausted retries
        return {"answer": "I don’t know based on the available documentation.", "mode": "agent", "confidence": 0.0, "chunks": [], "trace": trace, "steps": self.max_retries}


def build_agent(user_id=None, agent_enabled=True, allow_web: bool = True, domain: Optional[str] = None, document_id: Optional[str] = None, **kwargs):
    """
    Helper to build the agent; will attempt to fetch a Django user object if possible.
    """
    user_obj = None
    if User and user_id:
        try:
            user_obj = User.objects.get(id=user_id)
        except Exception:
            user_obj = None
    return AgenticRAG(user_id=user_id, user_obj=user_obj, allow_web=allow_web, max_retries=kwargs.get("max_retries", 3), domain=domain, document_id=document_id, agent_enabled=agent_enabled)
