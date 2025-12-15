from typing import Optional, Dict, Any, List
import logging
import numpy as np

from .retrieval import retrieve_with_conf, is_confident_enough, embed_texts
from .llm import call_llm_answer
from .tools.evaluate_context import evaluate_context_tool
from .tools.rag_search import rag_search
from .tools.fallback_llm import fallback_llm_tool

logger = logging.getLogger(__name__)

REPHRASE_MSG = "I don’t know based on the available documentation. Please rephrase or ask something more specific."
MIN_ACCEPT_CONF = float(__import__("os").environ.get("MIN_ACCEPT_CONF", 0.18))
SEMANTIC_THRESHOLD = float(__import__("os").environ.get("SEMANTIC_THRESHOLD", 0.40))


def _log_and_response(query: str, answer: str, chunks: List[Dict[str, Any]], top_score: float, validated: bool = False):
    """Creates QueryLog entry when available."""
    try:
        from api_app.models import QueryLog
        log = QueryLog.objects.create(query=query, answer=answer, top_score=top_score, chunks=chunks)
        qid = str(log.id)
    except Exception:
        qid = None

    return {
        "answer": answer,
        "validated": validated,
        "confidence": float(top_score or 0.0),
        "chunks": chunks,
        "query_id": qid
    }


def semantic_match(query: str, chunks: List[Dict[str, Any]]) -> float:
    """
    Compute semantic match between query and chunk texts using embed_texts.
    Returns max cosine similarity or 0.0 on error.
    """
    try:
        if not chunks:
            return 0.0
        # embed query and chunks
        texts = [query] + [c.get("text", "") for c in chunks]
        vecs = embed_texts(texts)
        arr = np.array(vecs)
        qv = arr[0]
        cvs = arr[1:]
        def cos(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        sims = [cos(qv, c) for c in cvs]
        return float(max(sims)) if sims else 0.0
    except Exception as e:
        logger.debug("semantic_match failed: %s", e)
        return 0.0


def run_rag(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    mood: str = "neutral",
    history: Optional[List[Dict[str, str]]] = None,
    max_chunks: int = 4
) -> Dict[str, Any]:

    logger.info("RAG pipeline invoked for query: %s", query)

    # 1) Primary RAG retrieval
    chunks, scores, conf = retrieve_with_conf(
        query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max_chunks * 2
    )

    # If nothing retrieved, try web fallback
    if not chunks:
        logger.info("RAG returned no chunks → trying web search fallback")
        try:
            web = rag_search(query, user_id=user_id, domain=domain, document_id=document_id, max_chunks=max_chunks)
        except Exception as e:
            logger.exception("web search failed: %s", e)
            web = {"found": False}

        if web.get("found") and web.get("chunks"):
            context = web.get("context", "")
            try:
                answer = call_llm_answer(question=query, context=context, mood=mood)
                final = answer.strip() if answer else context[:400]
            except Exception:
                final = context[:400]
            return _log_and_response(query, final, web["chunks"], float(web.get("confidence", 0.0)), validated=True)

        fallback = fallback_llm_tool(query, context="")
        return _log_and_response(query, fallback, [], 0.0, validated=False)

    # Evaluate chunk quality
    chosen = chunks[:max_chunks]
    eval_result = evaluate_context_tool(chosen)
    logger.info("run_rag → eval=%s  conf=%s", eval_result, conf)

    # Compute semantic score
    sem_score = semantic_match(query, chosen)
    logger.info("Semantic similarity score: %s", sem_score)

    # Acceptance rules
    accepted = False
    # Accept if evaluator says good and semantic passes threshold
    if eval_result.get("quality") == "good" and sem_score >= SEMANTIC_THRESHOLD:
        accepted = True
    # Accept if retrieval confidence is high (legacy behavior)
    elif is_confident_enough(conf) or float(conf or 0.0) >= MIN_ACCEPT_CONF:
        # but double-check semantic relevance — if confidence is high but semantic low, reject and go to web
        if sem_score >= SEMANTIC_THRESHOLD:
            accepted = True
        else:
            accepted = False
    # Accept if semantic relevance alone passes threshold
    elif sem_score >= SEMANTIC_THRESHOLD:
        accepted = True

    if not accepted:
        logger.info("Chunks exist but semantically irrelevant or failed eval → trying web search")
        web = rag_search(query, user_id=user_id, domain=domain, document_id=document_id, max_chunks=max_chunks)
        if web.get("found") and web.get("chunks"):
            context = web.get("context", "")
            try:
                answer = call_llm_answer(question=query, context=context, mood=mood)
                final = answer.strip() if answer else context[:400]
            except Exception:
                final = context[:400]
            return _log_and_response(query, final, web["chunks"], float(web.get("confidence", 0.0)), validated=True)

        # web failed — fallback to LLM but mark unvalidated
        context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
        fallback = fallback_llm_tool(query, context=context)
        return _log_and_response(query, fallback, chosen, float(conf or 0.0), validated=False)

    # Build final LLM context and answer
    context = "\n\n---\n\n".join(c.get("text", "") for c in chosen)
    if history:
        try:
            hist = "\n".join(f"{m['role']}: {m['content']}" for m in history if m.get("content"))
            if hist:
                context = hist + "\n\n---\n\n" + context
        except Exception:
            pass

    try:
        answer = call_llm_answer(question=query, context=context, mood=mood)
        final = answer.strip() if answer else (chosen[0]["text"] if chosen else REPHRASE_MSG)
    except Exception as e:
        logger.exception("Final LLM generation failed: %s", e)
        final = chosen[0]["text"] if chosen else REPHRASE_MSG

    return _log_and_response(query, final, chosen, float(conf or 0.0), validated=True)
