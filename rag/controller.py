# controller.py
import json
import logging
from typing import List, Dict, Any, Optional

from .llm import call_llm_answer
from .pipeline import run_rag
from .tools.rag_search import rag_search
from .tools.query_rewrite import rewrite_query_tool
from .tools.web_search import web_search_tool
from .tools.memory_search import memory_search_tool
from .tools.fallback_llm import fallback_llm_tool

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """
You are an autonomous agent controller.

Available tools:
- use_rag → rag_search(query)
- rewrite_and_retry → rewrite_query(query, context_chunks)
- use_web → web_search(query)
- use_memory → memory_search(query)
- fallback_llm → fallback_llm(query, context)
- finish → return final answer

You MUST return strictly valid JSON:
{
  "action": "<use_rag | rewrite_and_retry | use_web | use_memory | fallback_llm | finish>",
  "payload": { ... }
}
"""

def _ask_planner(history: List[Dict[str, Any]], query: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert messages into a single context string and include last RAG trace,
    domain and short memory so the planner can reason with retrieval evidence.
    """
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"User Query:\n{query}"}
    ]

    # Attach full trace entries (safe-guarded)
    for item in history:
        try:
            messages.append({"role": "assistant", "content": json.dumps(item)})
        except Exception:
            continue

    # Attach the latest RAG result preview (if any)
    last_rag = None
    for item in reversed(history):
        if item.get("tool") in ("rag_search", "retrieve", "rag_search_after_rewrite"):
            last_rag = item.get("result")
            break

    if last_rag:
        preview_chunks = []
        for c in last_rag.get("chunks", [])[:3]:
            preview_chunks.append({
                "id": c.get("id"),
                "snippet": (c.get("text") or "")[:300],
                "score": c.get("score")
            })
        preview = {
            "rag_preview": preview_chunks,
            "rag_confidence": float(last_rag.get("confidence", last_rag.get("top_score", 0.0) or 0.0)),
            "rag_answer_exists": bool(last_rag.get("answer"))
        }
        messages.append({"role": "assistant", "content": "RAG_PREVIEW: " + json.dumps(preview)})

    if domain:
        messages.append({"role": "assistant", "content": f"ACTIVE_DOMAIN: {domain}"})

    full_context = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    raw = call_llm_answer(
        question="Decide the next agent action. Return JSON only.",
        context=full_context,
        mood="neutral",
        max_tokens=300
    )

    logger.debug("Planner raw response: %s", raw)

    try:
        return json.loads(raw)
    except Exception:
        # best-effort fallback detection
        if isinstance(raw, str):
            text = raw.lower()
            if "use_rag" in text:
                return {"action": "use_rag", "payload": {"query": query}}
            if "rewrite" in text or "rewrite_and_retry" in text:
                return {"action": "rewrite_and_retry", "payload": {"query": query, "context_chunks": []}}
            if "use_web" in text:
                return {"action": "use_web", "payload": {"query": query}}
            if "finish" in text or "final" in text:
                # attempt to extract a finishing answer
                return {"action": "finish", "payload": {"answer": raw}}
        return {"action": "fallback_llm", "payload": {"query": query}}

def agentic_answer(
    query: str,
    user_id: Optional[int],
    domain: Optional[str],
    document_id: Optional[str],
    user=None,
    max_steps: int = 8
) -> Dict[str, Any]:
    """
    Main agentic loop. Planner is given short RAG previews and the working_trace
    is enriched with RAG results so subsequent planner decisions are evidence-aware.
    """
    working_trace: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    for step in range(max_steps):
        logger.debug("Agent loop step %d — current query: %s", step, (query or "")[:160])

        plan = _ask_planner(working_trace, query, domain=domain)
        action = plan.get("action")
        payload = plan.get("payload", {})

        logger.debug("Planner chose action=%s payload=%s", action, payload)

        # FINISH
        if action == "finish":
            final_answer = payload.get("answer") or payload.get("message") or ""
            working_trace.append({"step": step, "tool": "finish", "result": final_answer})
            break

        # RAG
        if action == "use_rag":
            rag_res = run_rag(
                query,
                user_id,
                domain,
                document_id,
                mood=payload.get("mood", "neutral"),
                history=working_trace
            )

            # expanded trace entry for planner consumption
            working_trace.append({
                "step": step,
                "tool": "rag_search",
                "result": {
                    "answer": rag_res.get("answer", ""),
                    "validated": rag_res.get("validated", False),
                    "confidence": float(rag_res.get("confidence", rag_res.get("top_score", 0.0) or 0.0)),
                    "chunks": [
                        {"id": c.get("id"), "snippet": (c.get("text") or "")[:400], "score": c.get("score")}
                        for c in rag_res.get("chunks", [])[:6]
                    ]
                }
            })

            if rag_res.get("validated"):
                final_answer = rag_res.get("answer")
                break
            # let planner decide next step with RAG trace available
            continue

        # Query Rewriting (immediately re-run retrieval after rewrite)
        if action == "rewrite_and_retry":
            new_q = rewrite_query_tool(
                query=payload.get("query", query),
                context_chunks=payload.get("context_chunks", [])
            )
            working_trace.append({"step": step, "tool": "rewrite_query", "result": new_q})

            if new_q:
                query = new_q
                # immediate retrieval after rewrite
                rag_res = run_rag(
                    query, user_id, domain, document_id, mood=payload.get("mood", "neutral"), history=working_trace
                )
                working_trace.append({
                    "step": step,
                    "tool": "rag_search_after_rewrite",
                    "result": {
                        "answer": rag_res.get("answer", ""),
                        "validated": rag_res.get("validated", False),
                        "confidence": float(rag_res.get("confidence", rag_res.get("top_score", 0.0) or 0.0)),
                        "chunks": [
                            {"id": c.get("id"), "snippet": (c.get("text") or "")[:300], "score": c.get("score")}
                            for c in rag_res.get("chunks", [])[:6]
                        ]
                    }
                })
                if rag_res.get("validated"):
                    final_answer = rag_res.get("answer")
                    break
            continue

        # Memory
        if action == "use_memory":
            mem = memory_search_tool(user=user, query=payload.get("query", query))
            working_trace.append({"step": step, "tool": "memory_search", "result": mem})

            if isinstance(mem, list):
                combined = "\n".join(m.get("text") if isinstance(m, dict) else str(m) for m in mem)
            else:
                combined = str(mem)

            summary = fallback_llm_tool(
                query="Summarize the user's past conversation in one helpful sentence:",
                context=combined
            )

            final_answer = summary if isinstance(summary, str) else str(summary)
            break

        # Web
        if action == "use_web":
            web_results = web_search_tool(payload.get("query", query))
            working_trace.append({"step": step, "tool": "web_search", "result": web_results})

            summary = fallback_llm_tool(
                query="Summarize the following web search results into a short news update:",
                context=json.dumps(web_results, indent=2)
            )

            final_answer = summary if isinstance(summary, str) else str(summary)
            break

        # Fallback LLM
        if action == "fallback_llm":
            res = fallback_llm_tool(
                query=payload.get("query", query),
                context=payload.get("context", "")
            )
            working_trace.append({"step": step, "tool": "fallback_llm", "result": res})

            if isinstance(res, str):
                final_answer = res
            elif isinstance(res, dict):
                final_answer = res.get("answer") or res.get("output") or str(res)
            else:
                final_answer = str(res)

            break

        # Unknown action — record and continue
        working_trace.append({
            "step": step,
            "tool": "unknown",
            "result": {"action": action, "payload": payload}
        })

    return {
    "answer": final_answer or "",
    "mode": "agent",
    "confidence": 1.0,
    "chunks": [],
    "trace": working_trace,
    "steps": len(working_trace)
}

