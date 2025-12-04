# rag/agent.py
from typing import Dict, Any, List, Optional
import logging
import json
import time

from rag.tools.langchain_tools import WrappedLLM, get_langchain_tools
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

try:
    from langchain.agents.react.base import create_react_agent
    from langchain.agents import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Fallback for Django User
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


class AgenticRAG:
    def __init__(self, user_id: Optional[int] = None, user_obj=None, allow_web: bool = True, max_retries: int = 3):
        self.user_id = user_id
        self.user = user_obj
        self.allow_web = allow_web
        self.max_retries = max_retries
        self._executor = None

        if LANGCHAIN_AVAILABLE:
            try:
                self._build_agent()
            except Exception as e:
                logger.exception("Failed to build LangChain agent: %s", e)
                self._executor = None

    def _build_agent(self):
        llm = WrappedLLM()
        tools = get_langchain_tools(user_obj=self.user, allow_web=self.allow_web)
        tool_names = ", ".join(t.name for t in tools) if isinstance(tools, list) else ", ".join(tools.keys())
        tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools) if isinstance(tools, list) else "\n".join(f"- {k}" for k in tools.keys())

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
        # fallback last line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        High-level orchestration. Agent manages rewrite -> rag_search -> evaluate_context -> retry.
        If LangChain executor present, we let it run; otherwise, emulate a controlled loop using tool wrappers.
        """
        if self._executor:
            # Let the agent run directly (it will call tools in the registry)
            try:
                res = self._executor.invoke({"input": user_query})
                raw = res.get("output", "") or ""
                final = self._extract_final_answer(raw)
                return {"answer": final or "I don’t know based on the available documentation.", "mode": "agent", "confidence": 1.0, "chunks": [], "trace": [{"tool": "agent", "result": raw}], "steps": 1}
            except Exception as e:
                logger.exception("Agent execution error: %s", e)
                # fallback to manual loop below
        # If no executor, perform manual agentic loop using local tools (lightweight)
        from rag.tools.langchain_tools import tool_rewrite, tool_rag, tool_evaluate, tool_fallback, tool_memory, tool_web

        refined_query = user_query
        trace: List[Dict[str, Any]] = []
        for attempt in range(1, max(1, self.max_retries) + 1):
            # 1) Optionally rewrite if query looks messy (simple heuristic)
            if len(refined_query.split()) < 2 or attempt > 1:
                try:
                    rewrite_input = json.dumps({"query": refined_query, "context_chunks": []})
                    new_q = tool_rewrite(rewrite_input)
                    if new_q and isinstance(new_q, str) and len(new_q) > 1:
                        refined_query = new_q
                        trace.append({"step": "rewrite", "attempt": attempt, "result": refined_query})
                except Exception:
                    pass

            # 2) RAG retrieval
            try:
                rag_payload = json.dumps({ "query": refined_query,"user_id": self.user_id,"domain": getattr(self, "domain", None),"document_id": getattr(self, "document_id", None)
                    })
                chunks = tool_rag(rag_payload)

                # ensure chunks are list-like when returned by rag_search wrapper
                if isinstance(chunks, dict) and chunks.get("chunks"):
                    chunks = chunks.get("chunks")
            except Exception as e:
                logger.exception("rag search failed: %s", e)
                chunks = []

            trace.append({"step": "rag", "attempt": attempt, "chunks_count": len(chunks)})

            # 3) Evaluate
            try:
                eval_in = json.dumps({"chunks": chunks})
                eval_res = tool_evaluate(eval_in)
            except Exception as e:
                eval_res = {"quality": "empty", "score": 0.0, "reason": f"eval_error:{str(e)}"}

            trace.append({"step": "evaluate", "attempt": attempt, "eval": eval_res})

            # 4) Decide
            qual = eval_res.get("quality", "empty")
            if qual == "good":
                # produce final answer using fallback_llm or a direct call with context
                try:
                    # Build compact context from top chunks (agent could pass more)
                    top_context = "\n\n---\n\n".join(c.get("text", "") for c in (chunks or [])[:4])
                    ans = tool_fallback(json.dumps({"query": user_query, "context": top_context}))
                    final = ans or "I don’t know based on the available documentation."
                except Exception:
                    final = "I don’t know based on the available documentation."
                trace.append({"step": "finalized", "attempt": attempt})
                return {"answer": final, "mode": "agent", "confidence": float(eval_res.get("score", 1.0)), "chunks": chunks, "trace": trace, "steps": attempt}
            else:
                # weak or empty => maybe use memory_search, rewrite differently, or use web if allowed
                # Try memory search once
                try:
                    mem = tool_memory(user_query, user_obj=self.user)
                    if mem:
                        trace.append({"step": "memory_search", "attempt": attempt, "result": mem})
                        # evaluate memory (very simply)
                        mem_eval = tool_evaluate(json.dumps({"chunks": mem if isinstance(mem, list) else [mem]}))
                        if mem_eval.get("quality") == "good":
                            # finalize with memory
                            top_context = "\n\n---\n\n".join((mem or [])[:4] if isinstance(mem, list) else [str(mem)])
                            final = tool_fallback(json.dumps({"query": user_query, "context": top_context}))
                            return {"answer": final, "mode": "agent", "confidence": float(mem_eval.get("score", 1.0)), "chunks": mem if isinstance(mem, list) else [], "trace": trace, "steps": attempt}
                except Exception:
                    pass

                # If allowed and agent decides, call web_search as last resort only on last attempt
                if self.allow_web and attempt == self.max_retries:
                    try:
                        web_res = tool_web(user_query)
                        trace.append({"step": "web_search", "attempt": attempt, "result": "ok"})
                        final = tool_fallback(json.dumps({"query": user_query, "context": json.dumps(web_res)}))
                        return {"answer": final, "mode": "agent_web", "confidence": 0.5, "chunks": [], "trace": trace, "steps": attempt}
                    except Exception:
                        pass

                # otherwise rewrite and retry
                # We ask rewrite tool for a stronger/shorter query
                try:
                    rewrite_input = json.dumps({"query": refined_query, "context_chunks": chunks or []})
                    refined_query = tool_rewrite(rewrite_input) or refined_query
                    trace.append({"step": "rewrite_retry", "attempt": attempt, "new_query": refined_query})
                except Exception:
                    trace.append({"step": "rewrite_failed", "attempt": attempt})
                # small sleep to avoid hammering
                time.sleep(0.2)

        # After retries exhausted: conservative fallback
        return {"answer": "I don’t know based on the available documentation.", "mode": "agent", "confidence": 0.0, "chunks": [], "trace": trace, "steps": self.max_retries}


def build_agent(user_id=None, agent_enabled=True, allow_web: bool = True, **kwargs):
    user_obj = None
    if User:
        try:
            user_obj = User.objects.get(id=user_id)
        except Exception:
            user_obj = None
    return AgenticRAG(user_id=user_id, user_obj=user_obj, allow_web=allow_web)
