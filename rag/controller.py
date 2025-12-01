# rag/controller.py
import json
from typing import List, Dict, Any, Optional

from rag.tools.rag_search import rag_search  # presumed wrapper over pipeline.run_rag or retrieval
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.web_search import web_search_tool
from rag.tools.memory_search import memory_search_tool
from rag.tools.fallback_llm import fallback_llm_tool
from rag.llm import call_llm_answer
from rag.pipeline import run_rag

# Planner prompt: agent is free to choose tools. We include last tool outputs and chunk diagnostics.
PLANNER_PROMPT = """
You are an autonomous agent controller. Available tools (name + brief):
- rag_search(query) -> returns document chunks and diagnostics
- rewrite_query(query, context_chunks) -> improves retrieval queries
- web_search(query) -> returns web results (title+body)
- memory_search(query) -> returns user memory entries
- fallback_llm(query, context) -> run an unconstrained LLM call for reasoning/creative output

You must return strict JSON with:
{
  "action": "<one of: use_rag | rewrite_and_retry | use_web | use_memory | fallback_llm | finish>",
  "payload": { ... }   # args for the chosen action
}

When recommending use_rag, only do so if the found chunks look sufficiently relevant. If the query is a simple greeting or casual talk, return "finish" with a short friendly reply and mark it as chat. NEVER output anything else outside the JSON.
"""

def _ask_planner(history: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"User Query: {query}"},
    ]
    # attach history (tool outputs) for planner context
    for h in history:
        messages.append({"role": "assistant", "content": json.dumps(h)})
    raw = call_llm_answer(
        question="Which action should the agent take? Output JSON only.",
        context="\n\n".join(m["content"] for m in messages),
        mood="neutral",
        max_tokens=400
    )
    try:
        return json.loads(raw)
    except Exception:
        # safe fallback
        return {"action": "fallback_llm", "payload": {"query": query, "context": ""}}

def agentic_answer(
    query: str,
    user_id: Optional[int],
    domain: Optional[str],
    document_id: Optional[str],
    user=None,
    max_steps: int = 6
) -> Dict[str, Any]:
    """
    High-level agent controller. It uses the planner to choose tools and
    returns a final 'answer' and trace.
    """
    working_trace: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    for step in range(max_steps):
        plan = _ask_planner(working_trace, query)
        action = plan.get("action")
        payload = plan.get("payload", {})

        if action == "finish":
            final_answer = payload.get("answer") or payload.get("message") or ""
            working_trace.append({"step": step, "tool": "finish", "result": final_answer})
            break

        if action == "use_rag":
            # call run_rag as a tool
            rag_res = run_rag(query, user_id, domain, document_id, mood=payload.get("mood", "neutral"))
            working_trace.append({"step": step, "tool": "rag_search", "result": rag_res})
            # if rag returned validated answer -> finish
            if rag_res.get("validated"):
                final_answer = rag_res.get("answer")
                break
            # else let planner decide next step: planner gets this rag_res as part of trace
            continue

        if action == "rewrite_and_retry":
            new_q = rewrite_query_tool(query=payload.get("query", query), context_chunks=payload.get("context_chunks", []))
            working_trace.append({"step": step, "tool": "rewrite_query", "result": new_q})
            query = new_q or query
            continue

        if action == "use_memory":
            mem = memory_search_tool(user=user, query=payload.get("query", query))
            working_trace.append({"step": step, "tool": "memory_search", "result": mem})
            continue

        if action == "use_web":
            web = web_search_tool(payload.get("query", query))
            working_trace.append({"step": step, "tool": "web_search", "result": web})
            # planner may finish after seeing web results
            continue

        if action == "fallback_llm":
            res = fallback_llm_tool(query=payload.get("query", query), context=payload.get("context", ""))
            working_trace.append({"step": step, "tool": "fallback_llm", "result": res})
            final_answer = res if isinstance(res, str) else (res.get("answer") if isinstance(res, dict) else str(res))
            break

        # Unknown action: record and ask planner again
        working_trace.append({"step": step, "tool": "unknown", "result": {"action": action, "payload": payload}})
        # allow planner to reconsider

    return {
        "answer": final_answer or "",
        "trace": working_trace,
        "steps": len(working_trace),
    }
