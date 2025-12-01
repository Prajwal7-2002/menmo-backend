# rag/pipeline.py
from typing import Optional, Dict, Any, List
from .retrieval import retrieve_with_conf, is_confident_enough, retrieve
from .llm import call_llm_answer
from api_app.models import QueryLog

def _log_and_response(query: str, answer: str, chunks: List[Dict[str,Any]], top_score: float, validated: bool=False):
    log = QueryLog.objects.create(
        query=query,
        answer=answer,
        top_score=top_score,
        chunks=chunks
    )
    return {
        "answer": answer,
        "validated": validated,
        "confidence": float(top_score or 0.0),
        "chunks": chunks,
        "query_id": str(log.id)
    }


def run_rag(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    mood: str = "neutral",
    history: Optional[List[Dict[str, str]]] = None,
    max_chunks: int = 4,
) -> Dict[str, Any]:

    chunks, scores, conf = retrieve_with_conf(query, user_id, domain, document_id, top_k=max_chunks*2)
    if not chunks:
        return _log_and_response(query, "", [], 0.0, validated=False)

    top = chunks[0]
    text = top.get("text", "").strip()
    text_len = len(text)
    bm25_norm = top.get("bm25_score_norm", 0.0)

    # --------- LAYER 1: Hybrid Confidence ---------
    if not is_confident_enough(conf):
        return _log_and_response(query, "", chunks[:max_chunks], conf, validated=False)

    # --------- LAYER 2: Meaningful Content ---------
    if text_len < 30 and bm25_norm < 0.20:
        return _log_and_response(query, "", chunks[:max_chunks], conf, validated=False)

    # --------- LAYER 3: LLM Semantic Validation ---------
    try:
        valid = call_llm_answer(
            question=f"""
Does the following chunk meaningfully answer the user's query?

USER:
{query}

CHUNK:
{text}

Answer only "yes" or "no".
""",
            context="",
            max_tokens=4
        )
        if not valid.lower().strip().startswith("y"):
            return _log_and_response(query, "", chunks[:max_chunks], conf, validated=False)
    except:
        # fail-safe: reject if validation fails
        return _log_and_response(query, "", chunks[:max_chunks], conf, validated=False)

    # --------- If all validation passed → REAL RAG ---------
    candidates = retrieve(query, user_id, domain, document_id, top_k=max_chunks*2)
    chosen = candidates[:max_chunks]

    context = "\n\n---\n\n".join([c["text"] for c in chosen])
    if history:
        memory = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        context = memory + "\n\n---\n\n" + context

    answer = call_llm_answer(question=query, context=context, mood=mood)
    final = answer.strip() if answer else text

    return _log_and_response(query, final, chosen, top_score=top["score"], validated=True)
