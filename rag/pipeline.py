# rag/pipeline.py
from typing import Optional, Dict, Any, List
import logging

from .retrieval import retrieve_with_conf, is_confident_enough
from .llm import call_llm_answer
from .tools.evaluate_context import evaluate_context_tool
from .tools.rag_search import rag_search
from .tools.fallback_llm import fallback_llm_tool

logger = logging.getLogger(__name__)

REPHRASE_MSG = "I don’t know based on the available documentation. Please rephrase or ask something more specific."

# overall threshold used as a last resort
MIN_ACCEPT_CONF = float(__import__("os").environ.get("MIN_ACCEPT_CONF", 0.18))

def _log_and_response(query: str, answer: str, chunks: List[Dict[str, Any]], top_score: float, validated: bool=False):
    # preserve previous QueryLog behavior if you have it; else return dict
    try:
        from api_app.models import QueryLog
        log = QueryLog.objects.create(query=query, answer=answer, top_score=top_score, chunks=chunks)
        qid = str(log.id)
    except Exception:
        qid = None
    return {"answer": answer, "validated": validated, "confidence": float(top_score or 0.0), "chunks": chunks, "query_id": qid}

def run_rag(query: str, user_id: Optional[int] = None, domain: Optional[str] = None,
            document_id: Optional[str] = None, mood: str = "neutral", history: Optional[List[Dict[str, str]]] = None,
            max_chunks: int = 4) -> Dict[str, Any]:
    """
    Retrieval -> context evaluation -> LLM answer flow.
    Uses evaluate_context_tool to accept/reject retrieval results.
    """

    # 1) retrieve candidates (with diagnostic score)
    chunks, scores, conf = retrieve_with_conf(query, user_id=user_id, domain=domain, document_id=document_id, top_k=max_chunks * 2)
    if not chunks:
        # as a last resort call fallback llm with available minimal context
        fallback = fallback_llm_tool(query, context="")
        return _log_and_response(query, fallback, [], 0.0, validated=False)

    chosen = chunks[:max_chunks]
    # evaluate the retrieval context quality
    eval_result = evaluate_context_tool(chosen)
    logger.info("run_rag: retrieval_conf=%s eval=%s", conf, eval_result)

    # Accept if retrieval confidence OR evaluator says 'good'
    accepted = False
    if eval_result.get("quality") == "good":
        accepted = True
    elif is_confident_enough(conf) or float(conf or 0.0) >= float(MIN_ACCEPT_CONF):
        accepted = True

    if not accepted:
        # not safe to ground LLM — fallback to short safe answer via fallback_llm
        context = "\n\n---\n\n".join(c.get("text","") for c in chosen)
        fallback = fallback_llm_tool(query, context=context)
        return _log_and_response(query, fallback, chosen, float(conf or 0.0), validated=False)

    # build context (optionally prepend history)
    context = "\n\n---\n\n".join([c.get("text","") for c in chosen])
    if history:
        try:
            memory = "\n".join(f"{m['role']}: {m['content']}" for m in history if isinstance(m, dict) and m.get('content'))
            if memory:
                context = memory + "\n\n---\n\n" + context
        except Exception:
            pass

    # final LLM call
    try:
        answer = call_llm_answer(question=query, context=context, mood=mood)
        final = answer.strip() if answer else (chosen[0]["text"] if chosen else REPHRASE_MSG)
    except Exception as e:
        logger.exception("run_rag: final LLM failed: %s", e)
        final = chosen[0]["text"] if chosen else REPHRASE_MSG

    return _log_and_response(query, final, chosen, float(conf or 0.0), validated=True)
