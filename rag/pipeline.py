# rag/pipeline.py
from typing import Optional, Dict, Any, List

from .retrieval import retrieve, MIN_SCORE_THRESHOLD
from .llm import call_llm_answer


REPHRASE_MSG = (
    "I don't know based on the available documentation. "
    "Please rephrase your question or ask something more specific."
)


def run_rag(
    query: str,
    user_id: Optional[int] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    mood: str = "neutral",
    history: Optional[List[Dict[str, str]]] = None,
    max_chunks: int = 4,
) -> Dict[str, Any]:

    """
    run_rag:
      - query: user query string
      - user_id/domain/document_id: filters
      - mood: passed to LLM for style
      - history: optional list of {"role": "user"|"assistant", "content": "..."} for multi-turn
    """

    candidates = retrieve(
        query=query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max_chunks * 2,
    )

    if not candidates:
        return {
            "answer": REPHRASE_MSG,
            "validated": False,
            "chunks": [],
            "reason": "no_relevant_documents",
        }

    # 2) Score validation
    top = candidates[0]
    if top["score"] < MIN_SCORE_THRESHOLD:
        return {
            "answer": REPHRASE_MSG,
            "validated": False,
            "chunks": candidates,
            "reason": "low_relevance",
        }

    # 3) Pick chunks
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # 3.1 Combine history (if provided) with retrieved context to give the LLM conversational context
    if history and isinstance(history, list):
        try:
            history_text = "\n".join([f"{h.get('role','user')}: {h.get('content','')}" for h in history if h.get("content")])
            if history_text:
                context = history_text + "\n\n---\n\n" + context
        except Exception:
            # keep existing context on error
            pass

    # 4) LLM grounding
    try:
        llm_answer = call_llm_answer(query, context, mood=mood)
        if not llm_answer or llm_answer.strip() == "":
            fallback = chosen[0]["text"]
            return {
                "answer": fallback,
                "validated": True,
                "chunks": chosen,
                "note": "fallback_used_empty_or_none_llm",
            }

        return {
            "answer": llm_answer.strip(),
            "validated": True,
            "chunks": chosen,
        }

    except Exception as e:
        fallback = chosen[0]["text"]
        return {
            "answer": fallback,
            "validated": True,
            "chunks": chosen,
            "note": f"fallback_used_due_to_llm_error: {e}",
        }
