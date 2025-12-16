from typing import Dict, Any, List, Optional
import logging
import json
import time

logger = logging.getLogger(__name__)

# Try to import langchain agent utilities; degrade gracefully.
try:
    # Modern LangChain exposes create_react_agent from langchain.agents
    from langchain.agents import create_react_agent, AgentExecutor
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Optional flag to turn on the LangChain ReAct agent.
# By default we KEEP THIS OFF and use the manual AgenticRAG.run() pipeline,
# because the LangChain path is more brittle across versions.
USE_REACT_AGENT = str(__import__("os").environ.get("USE_REACT_AGENT", "0")).lower() in ("1", "true", "yes")

# Local tool wrappers (from rag.tools.langchain_tools)
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
    def tool_rag(payload):
        return {"chunks": [], "found": False}
    def tool_rewrite(payload):
        try:
            data = json.loads(payload)
            return data.get("query", "")
        except Exception:
            return ""
    def tool_evaluate(payload):
        return {"quality": "empty", "score": 0.0}
    def tool_fallback(payload):
        try:
            data = json.loads(payload)
            ctx = data.get("context", "")
            if ctx:
                return f"Based on retrieved context: {ctx[:800]}"
            return "I don’t know based on the available documentation."
        except Exception:
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

# rag_search direct call
try:
    from rag.tools.rag_search import rag_search
except Exception:
    rag_search = None

# embed helper
try:
    from rag.retrieval import embed_texts
    import numpy as _np
except Exception:
    embed_texts = None
    _np = None

SYSTEM_PROMPT = """
You are NeuroStack Agent. Your job is to answer user queries using internal documents (RAG) and memory.
You must attempt to retrieve evidence and only use the web if internal sources are insufficient.
If unsure after retries, respond conservatively: "I don’t know based on the available documentation."
When using tools follow the specified tool format. Final output must contain 'Final Answer:' when using LangChain agent.
"""

SEMANTIC_MIN_SCORE = float(__import__("os").environ.get("SEMANTIC_MIN_SCORE", 0.40))
SEMANTIC_AGG_METHOD = __import__("os").environ.get("SEMANTIC_AGG", "max")  # "max" or "mean"


class AgenticRAG:
    def __init__(
        self,
        user_id: Optional[int] = None,
        user_obj=None,
        allow_web: bool = True,
        max_retries: int = 3,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        agent_enabled: bool = True,
        question_intent: Optional[str] = None,
    ):
        self.user_id = user_id
        self.user = user_obj
        self.allow_web = allow_web
        self.max_retries = max_retries or 1
        self.domain = domain
        self.document_id = document_id
        self.agent_enabled = agent_enabled
        # Optional hint from the router about what kind of
        # question this is ("doc_summary", "doc_lookup", etc.).
        self.question_intent: Optional[str] = question_intent
        self._executor = None

        # Only build the LangChain ReAct agent when explicitly enabled.
        # Otherwise, rely on the manual tool-orchestrated pipeline in run().
        if LANGCHAIN_AVAILABLE and agent_enabled and USE_REACT_AGENT:
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
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()

    def _rag_tool_call(self, query: str) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "user_id": self.user_id,
            "domain": self.domain,
            "document_id": self.document_id,
        }
        try:
            res = tool_rag(json.dumps(payload))
            if isinstance(res, dict):
                return res.get("chunks", []) or []
            elif isinstance(res, list):
                return res
            else:
                try:
                    parsed = json.loads(res)
                    if isinstance(parsed, dict):
                        return parsed.get("chunks", []) or []
                except Exception:
                    pass
                return []
        except Exception as e:
            logger.exception("tool_rag() failed: %s", e)
            return []

    def _compute_semantic_score(self, query: str, chunks: List[Dict[str, Any]]) -> float:
        try:
            if not embed_texts or not _np or not chunks:
                return 0.0
            texts = [query] + [c.get("text", "") for c in chunks if c.get("text")]
            if len(texts) < 2:
                return 0.0
            vecs = embed_texts(texts)
            arr = _np.array(vecs)
            qv = arr[0]
            cvs = arr[1:]
            def cos(a, b):
                na = _np.linalg.norm(a)
                nb = _np.linalg.norm(b)
                if na == 0 or nb == 0:
                    return 0.0
                return float(_np.dot(a, b) / (na * nb))
            sims = [cos(qv, c) for c in cvs]
            if not sims:
                return 0.0
            if SEMANTIC_AGG_METHOD == "mean":
                return float(sum(sims) / len(sims))
            return float(max(sims))
        except Exception as e:
            logger.debug("semantic scoring failed: %s", e)
            return 0.0

    def _call_web_fallback(self, query: str, max_chunks: int = 4) -> Dict[str, Any]:
        """
        Web-only fallback. Used when internal RAG + memory are insufficient.
        """
        try:
            tw = tool_web(json.dumps({
                "query": query,
                "user_id": self.user_id,
                "domain": self.domain,
                "document_id": self.document_id,
                "max_chunks": max_chunks,
            }))
            if tw and isinstance(tw, dict) and tw.get("found"):
                return tw
        except Exception as e:
            logger.exception("_call_web_fallback failed: %s", e)
        return {"found": False, "chunks": [], "context": "", "confidence": 0.0}

    def run(self, user_query: str) -> Dict[str, Any]:
        # LangChain executor path
        if self._executor:
            try:
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

        refined_query = user_query
        trace: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_retries) + 1):
            # rewrite early if short or after first attempt, but avoid changing strict doc queries
            if self.document_id is None and (attempt > 1 or len(refined_query.split()) < 3):
                try:
                    rewrite_payload = json.dumps({
                        "query": refined_query,
                        "context_chunks": [],
                        "domain": self.domain,
                        "document_id": self.document_id,
                    })
                    new_q = tool_rewrite(rewrite_payload)
                    if new_q and isinstance(new_q, str) and len(new_q) > 1 and new_q != refined_query:
                        refined_query = new_q
                        trace.append({"step": "rewrite", "attempt": attempt, "result": refined_query})
                except Exception:
                    pass


            # RAG retrieval
            try:
                chunks = self._rag_tool_call(refined_query) or []
            except Exception as e:
                logger.exception("rag retrieval failed: %s", e)
                chunks = []

            trace.append({"step": "rag", "attempt": attempt, "chunks_count": len(chunks)})

            # If there is no internal context at all and we are NOT in strict
            # document mode, prefer to go directly to web (when allowed)
            # instead of returning an empty/fallback answer.
            if not chunks and self.allow_web and self.document_id is None:
                try:
                    web_res = self._call_web_fallback(refined_query, max_chunks=4)
                    trace.append({
                        "step": "no_rag_chunks_trigger_web",
                        "attempt": attempt,
                        "result_found": web_res.get("found", False),
                    })
                    if web_res.get("found"):
                        context = web_res.get("context", "") or "\n\n---\n\n".join(
                            c.get("text", "") for c in web_res.get("chunks", [])[:4]
                        )
                        try:
                            final = tool_fallback(json.dumps({
                                "query": user_query,
                                "context": context,
                                "domain": self.domain,
                                "document_id": self.document_id,
                            }))
                        except Exception:
                            final = context[:800] or "I don't know based on the available documentation."
                        return {
                            "answer": final,
                            "mode": "agent_web",
                            "source": "web",
                            "answer_source": "web",
                            "confidence": float(web_res.get("confidence", 0.5)),
                            "chunks": web_res.get("chunks", []),
                            "trace": trace,
                            "steps": attempt,
                        }
                except Exception:
                    # If web also fails, fall through to the normal evaluator logic
                    pass

            # evaluate chunks
            try:
                eval_input = json.dumps({"chunks": chunks})
                eval_res = tool_evaluate(eval_input)
            except Exception:
                eval_res = {"quality": "empty", "score": 0.0}

            trace.append({"step": "evaluate", "attempt": attempt, "eval": eval_res})

            # semantic score
            sem_score = self._compute_semantic_score(user_query, chunks)
            trace.append({"step": "semantic_score", "attempt": attempt, "value": sem_score})

            # Accept document answers more eagerly in strict doc mode
            # (document_id is set). For open-agent mode (no document_id)
            # keep a stricter rule.
            if self.document_id is not None:
                # For strict document questions:
                # - doc_summary: allow answers whenever some context exists and
                #   evaluator is at least "weak" (used for high-level summaries).
                # - doc_lookup or unknown: keep stricter rules so we do not
                #   hallucinate specific values.
                qi = (self.question_intent or "").strip().lower()
                if qi == "doc_summary":
                    accept = eval_res.get("quality") in ("good", "weak") and bool(chunks)
                else:
                    # - If evaluator says "good", always allow a doc-based answer.
                    # - If evaluator says "weak", require a strong semantic match
                    #   to avoid hallucinating specific values.
                    if eval_res.get("quality") == "good":
                        accept = True
                    else:
                        accept = eval_res.get("quality") == "weak" and sem_score >= 0.60
            else:
                # Open agent mode: only accept when context is clearly good
                # and aligned with the question.
                accept = eval_res.get("quality") == "good" and sem_score >= SEMANTIC_MIN_SCORE

            if accept:
                try:
                    top_context = "\n\n---\n\n".join(c.get("text", "") for c in (chunks or [])[:4])
                    fallback_payload = json.dumps({
                        "query": user_query,
                        "context": top_context,
                        "domain": self.domain,
                        "document_id": self.document_id,
                    })
                    final = tool_fallback(fallback_payload)
                except Exception:
                    final = "I don’t know based on the available documentation."
                trace.append({"step": "finalized", "attempt": attempt})
                return {
                    "answer": final,
                    "mode": "agent",
                    "source": "document",
                    "answer_source": "document",
                    "confidence": float(eval_res.get("score", 1.0)),
                    "chunks": chunks,
                    "trace": trace,
                    "steps": attempt,
                }


            # If evaluator good or weak but semantic low -> try web fallback
            if eval_res.get("quality") in ("good", "weak") and sem_score < SEMANTIC_MIN_SCORE and self.allow_web:
                trace.append({"step": "semantic_low_trigger_web", "attempt": attempt, "sem_score": sem_score})
                try:
                    web_res = self._call_web_fallback(refined_query, max_chunks=4)
                    trace.append({"step": "web_search", "attempt": attempt, "result_found": web_res.get("found", False)})
                    if web_res.get("found"):
                        context = web_res.get("context", "") or "\n\n---\n\n".join(
                            c.get("text", "") for c in web_res.get("chunks", [])[:4]
                        )
                        try:
                            final = tool_fallback(json.dumps({
                                "query": user_query,
                                "context": context,
                                "domain": self.domain,
                                "document_id": self.document_id,
                            }))
                        except Exception:
                            final = context[:800] or "I don't know based on the available documentation."
                        return {
                            "answer": final,
                            "mode": "agent_web",
                            "source": "web",
                            "answer_source": "web",
                            "confidence": float(web_res.get("confidence", 0.5)),
                            "chunks": web_res.get("chunks", []),
                            "trace": trace,
                            "steps": attempt,
                        }
                except Exception:
                    pass

            # In open-agent mode (no document_id), if chunks are weak but
            # semantically strong for a general question, prefer web over
            # returning "I don't know".
            if self.document_id is None and eval_res.get("quality") == "weak" and sem_score >= SEMANTIC_MIN_SCORE and self.allow_web:
                trace.append({"step": "weak_but_semantic_high_trigger_web", "attempt": attempt, "sem_score": sem_score})
                try:
                    web_res = self._call_web_fallback(refined_query, max_chunks=4)
                    trace.append({"step": "web_search", "attempt": attempt, "result_found": web_res.get("found", False)})
                    if web_res.get("found"):
                        context = web_res.get("context", "") or "\n\n---\n\n".join(
                            c.get("text", "") for c in web_res.get("chunks", [])[:4]
                        )
                        try:
                            final = tool_fallback(json.dumps({
                                "query": user_query,
                                "context": context,
                                "domain": self.domain,
                                "document_id": self.document_id,
                            }))
                        except Exception:
                            final = context[:800] or "I don't know based on the available documentation."
                        return {
                            "answer": final,
                            "mode": "agent_web",
                            "source": "web",
                            "answer_source": "web",
                            "confidence": float(web_res.get("confidence", 0.5)),
                            "chunks": web_res.get("chunks", []),
                            "trace": trace,
                            "steps": attempt,
                        }
                except Exception:
                    pass


            # memory
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
                pass

            # last attempt -> web fallback
            if self.allow_web and attempt == self.max_retries:
                try:
                    # Final safety net: try web search one more time.
                    web_res = self._call_web_fallback(refined_query, max_chunks=4)
                    trace.append({"step": "web_search_last_attempt", "attempt": attempt, "found": web_res.get("found", False)})
                    if web_res.get("found"):
                        context = web_res.get("context", "") or "\n\n---\n\n".join(c.get("text", "") for c in web_res.get("chunks", [])[:4])
                        try:
                            final = tool_fallback(json.dumps({"query": user_query, "context": context, "domain": self.domain, "document_id": self.document_id}))
                        except Exception:
                            final = context[:800] or "I don't know based on the available documentation."
                        return {
                            "answer": final,
                            "mode": "agent_web",
                            "source": "web",
                            "answer_source": "web",
                            "confidence": float(web_res.get("confidence", 0.5)),
                            "chunks": web_res.get("chunks", []),
                            "trace": trace,
                            "steps": attempt,
                        }
                except Exception:
                    pass

            # rewrite for next attempt (with context), but never broaden strict doc queries
            if self.document_id is None:
                try:
                    rewrite_payload = json.dumps({
                        "query": refined_query,
                        "context_chunks": chunks or [],
                        "domain": self.domain,
                        "document_id": self.document_id,
                    })
                    new_ref = tool_rewrite(rewrite_payload) or refined_query
                    if new_ref != refined_query:
                        refined_query = new_ref
                        trace.append({"step": "rewrite_retry", "attempt": attempt, "new_query": refined_query})
                except Exception:
                    trace.append({"step": "rewrite_failed", "attempt": attempt})


        # exhausted retries
        return {
            "answer": "I don't know based on the available documentation.",
            "mode": "agent",
            "source": "fallback",
            "answer_source": "fallback",
            "confidence": 0.0,
            "chunks": [],
            "trace": trace,
            "steps": self.max_retries,
        }


def build_agent(
    user_id=None,
    agent_enabled=True,
    allow_web: bool = True,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    **kwargs,
):
    user_obj = None
    if User and user_id:
        try:
            user_obj = User.objects.get(id=user_id)
        except Exception:
            user_obj = None
    return AgenticRAG(
        user_id=user_id,
        user_obj=user_obj,
        allow_web=allow_web,
        max_retries=kwargs.get("max_retries", 3),
        domain=domain,
        document_id=document_id,
        agent_enabled=agent_enabled,
        question_intent=kwargs.get("question_intent"),
    )
