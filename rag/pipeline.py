from typing import Optional, Dict, Any, List
import logging

from .retrieval import retrieve_with_conf, is_confident_enough
from .llm import call_llm_answer
from api_app.models import QueryLog

logger = logging.getLogger(__name__)

def _log_and_response(query: str, answer: str, chunks: List[Dict[str,Any]], top_score: float, validated: bool=False):
    log = QueryLog.objects.create(query=query, answer=answer, top_score=top_score, chunks=chunks)
    return {"answer": answer, "validated": validated, "confidence": float(top_score or 0.0), "chunks": chunks, "query_id": str(log.id)}

def run_rag(query: str, user_id: Optional[int]=None, domain: Optional[str]=None,
            document_id: Optional[str]=None, mood: str="neutral", history: Optional[List[Dict[str,str]]]=None,
            max_chunks: int=4) -> Dict[str, Any]:

    # Retrieve with diagnostics
    chunks, scores, conf = retrieve_with_conf(query, user_id, domain, document_id, top_k=max_chunks*2)
    if not chunks:
        return _log_and_response(query, "", [], 0.0, validated=False)

    candidates = chunks
    top = candidates[0]
    text = top.get("text", "").strip()
    text_len = len(text)
    bm25_norm = float(top.get("bm25_score_norm") or top.get("bm25_norm") or 0.0)

    logger.info(f"run_rag: conf={conf:.4f}  top_text_len={text_len}  bm25_norm={bm25_norm:.4f}")

    # GATING
    if is_confident_enough(conf):
        accepted = True
    else:
        MIN_TEXT_LEN = 40
        accepted = (text_len >= MIN_TEXT_LEN) or (bm25_norm >= 0.40)

    if not accepted:
        logger.info("run_rag: gating rejected (insufficient confidence/bm25/text_len)")
        return _log_and_response(query, "", candidates[:max_chunks], conf, validated=False)

    # build context
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c.get("text", "") for c in chosen])
    if history:
        memory = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        context = memory + "\n\n---\n\n" + context

    # final LLM
    try:
        answer = call_llm_answer(question=query, context=context, mood=mood)
        final = answer.strip() if answer else (chosen[0]["text"] if chosen else "")
    except Exception as e:
        logger.exception("run_rag: final LLM answer failed: %s", e)
        final = chosen[0]["text"] if chosen else ""

    return _log_and_response(query, final, chosen, top_score=(top.get("score") or 0.0), validated=True)
