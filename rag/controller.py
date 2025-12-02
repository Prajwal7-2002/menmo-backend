# controller.py (patched)
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

You MUST return strictly valid JSON with the fields:
{
  "action": "<use_rag | rewrite_and_retry | use_web | use_memory | fallback_llm | finish>",
  "payload": { ... }
}

Only return JSON. Do not include explanations or extra text.
"""

_ALLOWED_ACTIONS = {"use_rag", "rewrite_and_retry", "use_web", "use_memory", "fallback_llm", "finish"}


def _ask_planner(history: List[Dict[str, Any]], query: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a compact, well-structured planner prompt and parse its JSON reply robustly.
    Avoid passing giant JSON dumps to the LLM (causes hallucinated / non-JSON responses).
    """
    # Basic system + user
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"User Query:\n{query}"}
    ]

    # Add small, safe summaries of recent tool steps (no full dumps)
    # Include only last N steps to keep prompt small
    MAX_HISTORY_STEPS = 6
    for item in history[-MAX_HISTORY_STEPS:]:
        tool = item.get("tool", "unknown")
        step = item.get("step", None)
        # short description if available
        if isinstance(item.get("result"), dict) and "answer" in item.get("result"):
            snippet = (item["result"].get("answer") or "")[:200]
            snippet_text = f"ANSWER_SNIPPET: {snippet}"
        else:
            snippet_text = ""
        messages.append({
            "role": "assistant",
            "content": f"TOOL_STEP: {tool} (step {step}) {snippet_text}"
        })

    # Attach a tiny RAG preview (only IDs + short snippets + score)
    last_rag = None
    for item in reversed(history):
        if item.get("tool") in ("rag_search", "rag_search_after_rewrite", "retrieve"):
            last_rag = item.get("result")
            break

    if last_rag:
        preview_chunks = []
        for c in (last_rag.get("chunks") or [])[:3]:
            preview_chunks.append({
                "id": c.get("id"),
                "snippet": (c.get("snippet") or (c.get("text") or "")[:180])[:180],
                "score": float(c.get("score") or 0.0)
            })
        preview = {
            "rag_preview": preview_chunks,
            "rag_confidence": float(last_rag.get("confidence", last_rag.get("top_score", 0.0) or 0.0)),
            "rag_answer_exists": bool(last_rag.get("answer"))
        }
        messages.append({"role": "assistant", "content": "RAG_PREVIEW: " + json.dumps(preview)})

    if domain:
        messages.append({"role": "assistant", "content": f"ACTIVE_DOMAIN: {domain}"})

    # Build the final compact context string for the LLM
    full_context = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    # Ask planner (smaller token budget improves JSON consistency)
    raw = call_llm_answer(
        question="Decide the next agent action. Return JSON only.",
        context=full_context,
        mood="neutral",
        max_tokens=200
    )

    # Log raw planner output for debugging (helps diagnose non-JSON replies)
    logger.debug("Planner raw response: %s", raw)

    # Robust JSON parse + validation
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            action = parsed.get("action")
            if action in _ALLOWED_ACTIONS:
                # ensure payload exists
                payload = parsed.get("payload", {}) or {}
                return {"action": action, "payload": payload}
    except Exception:
        # fall through to best-effort text heuristics
        pass

    # Best-effort string heuristics (if LLM returned plain text)
    if isinstance(raw, str):
        text = raw.lower()
        if "use_rag" in text or "rag" in text:
            return {"action": "use_rag", "payload": {"query": query}}
        if "rewrite" in text or "rewrite_and_retry" in text:
            return {"action": "rewrite_and_retry", "payload": {"query": query, "context_chunks": []}}
        if "use_web" in text or "web" in text:
            return {"action": "use_web", "payload": {"query": query}}
        if "use_memory" in text or "memory" in text:
            return {"action": "use_memory", "payload": {"query": query}}
        if "finish" in text or "final" in text or "i think" in text:
            # try to extract a finishing answer (use raw as payload.answer)
            return {"action": "finish", "payload": {"answer": raw}}

    # Last-resort safe fallback: call fallback_llm flow
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
    Main agentic loop. Planner is given compact RAG previews and
    short tool-step summaries so it can decide the next action safely.
    Ensures final responses are always non-empty and returned in a consistent shape.
    """
    working_trace: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    for step in range(max_steps):
        logger.debug("Agent loop step %d — current query: %s", step, (query or "")[:160])

        plan = _ask_planner(working_trace, query, domain=domain)
        action = plan.get("action")
        payload = plan.get("payload", {}) or {}

        logger.debug("Planner chose action=%s payload=%s", action, payload)

        # FINISH action: planner gives final answer
                # FINISH
        if action == "finish":
            final_answer = payload.get("answer") or payload.get("message") or ""
            working_trace.append({"step": step, "tool": "finish", "result": final_answer})

            # If planner finished but answer is empty or a generic "no docs" message,
            # automatically try web search + summarize as a best-effort fallback.
            try:
                low = (final_answer or "").strip().lower()
                generic_no_doc = "don" in low and "know" in low and "document" in low  # loose check
                if (not final_answer or not final_answer.strip()) or generic_no_doc:
                    # run web search
                    web_results = web_search_tool(query)
                    working_trace.append({"step": step, "tool": "web_search", "result": web_results})

                    # summarize web results using fallback LLM
                    summary = fallback_llm_tool(
                        query="Provide a short factual answer to the user's question using the web search results:",
                        context=json.dumps(web_results, indent=2)
                    )

                    if isinstance(summary, str) and summary.strip():
                        final_answer = summary
                        working_trace.append({"step": step, "tool": "web_summary", "result": final_answer})
            except Exception as e:
                logger.exception("Automatic web fallback failed: %s", e)

            break

        # RAG action
        if action == "use_rag":
            rag_res = run_rag(
                query,
                user_id,
                domain,
                document_id,
                mood=payload.get("mood", "neutral"),
                history=working_trace
            )

            working_trace.append({
                "step": step,
                "tool": "rag_search",
                "result": {
                    "answer": rag_res.get("answer", ""),
                    "validated": rag_res.get("validated", False),
                    "confidence": float(rag_res.get("confidence", rag_res.get("top_score", 0.0) or 0.0)),
                    "chunks": [
                        {
                            "id": c.get("id"),
                            "snippet": (c.get("text") or "")[:300],
                            "score": c.get("score")
                        }
                        for c in rag_res.get("chunks", [])[:5]
                    ]
                }
            })

            if rag_res.get("validated"):
                final_answer = rag_res.get("answer")
                break
            # allow planner to pick next action using this trace
            continue

        # Query rewrite + immediate retrieval
        if action == "rewrite_and_retry":
            new_q = rewrite_query_tool(
                query=payload.get("query", query),
                context_chunks=payload.get("context_chunks", []) or []
            )
            working_trace.append({"step": step, "tool": "rewrite_query", "result": new_q or ""})

            if new_q:
                query = new_q
                rag_res = run_rag(
                    query,
                    user_id,
                    domain,
                    document_id,
                    mood=payload.get("mood", "neutral"),
                    history=working_trace
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
                            for c in rag_res.get("chunks", [])[:5]
                        ]
                    }
                })
                if rag_res.get("validated"):
                    final_answer = rag_res.get("answer")
                    break
            continue

        # Memory retrieval
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

        # Web search
        if action == "use_web":
            web_results = web_search_tool(payload.get("query", query) or query)
            working_trace.append({"step": step, "tool": "web_search", "result": web_results})

            summary = fallback_llm_tool(
                query="Summarize the following web search results into a short news update:",
                context=json.dumps(web_results, indent=2) if isinstance(web_results, (list, dict)) else str(web_results)
            )

            final_answer = summary if isinstance(summary, str) else str(summary)
            break

        # Fallback LLM: general answer
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

        # Unknown action: record and let planner adapt
        working_trace.append({
            "step": step,
            "tool": "unknown",
            "result": {"action": action, "payload": payload}
        })

    # Ensure final_answer is never empty — provide a friendly fallback instead of empty string
    if not final_answer or not str(final_answer).strip():
        # call fallback LLM with a tiny prompt to try to create a friendly message
        try:
            fallback_text = fallback_llm_tool(
                query="Provide a brief helpful response to the user's query:",
                context=query
            )
            final_answer = fallback_text if isinstance(fallback_text, str) and fallback_text.strip() else "I couldn't determine the answer — can you rephrase or give more detail?"
        except Exception:
            final_answer = "I couldn't determine the answer — can you rephrase or give more detail?"

    # Unified response shape for frontend stability
    return {
        "answer": final_answer,
        "mode": "agent",
        "confidence": 1.0,
        "chunks": [],
        "trace": working_trace,
        "steps": len(working_trace)
    }
