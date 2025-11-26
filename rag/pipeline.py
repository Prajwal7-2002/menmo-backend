# rag/pipeline.py
from typing import Optional, Dict, Any, List

from .retrieval import retrieve, MIN_SCORE_THRESHOLD
from .llm import call_llm_answer

REPHRASE_MSG = (
    "I don’t know based on the available documentation. "
    "Please rephrase your question or ask something more specific."
)


def _fail(reason: str, chunks=None, confidence: float = 0.0) -> Dict[str, Any]:
    fallback = chunks[0]["text"] if chunks else REPHRASE_MSG
    return {
        "answer": fallback,
        "validated": False,
        "confidence": confidence,
        "reason": reason,
        "chunks": chunks or [],
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
    """
    High-level RAG pipeline:
      - retrieve candidates
      - apply confidence gate
      - fuse chunks into context
      - call LLM with hallucination guard
    """

    # 1) Retrieve candidate chunks
    candidates = retrieve(
        query=query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max_chunks * 2,
    )

    if not candidates:
        return _fail("no_relevant_documents", [], 0.0)

    # 2) Confidence gate
    top = candidates[0]
    if top["score"] < MIN_SCORE_THRESHOLD:
        return _fail("low_confidence_match", candidates, top["score"])

    # 3) Choose chunks for context
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # 3.1 Add conversational history if provided
    if history and isinstance(history, list):
        try:
            history_text = "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '')}"
                for h in history
                if h.get("content")
            )
            if history_text:
                context = history_text + "\n\n---\n\n" + context
        except Exception:
            pass

    # 4) LLM call with grounding
    try:
        llm_answer = call_llm_answer(query, context, mood=mood)
        if not llm_answer or not llm_answer.strip():
            # fallback to best chunk
            return {
                "answer": chosen[0]["text"],
                "validated": True,
                "confidence": top["score"],
                "chunks": chosen,
                "note": "fallback_used_empty_llm",
            }

        answer = llm_answer.strip()
        return {
            "answer": answer,
            "validated": True,
            "confidence": top["score"],
            "chunks": chosen,
        }

    except Exception as e:
        return {
            "answer": chosen[0]["text"],
            "validated": True,
            "confidence": top["score"],
            "chunks": chosen,
            "note": f"fallback_used_due_to_llm_error: {e}",
        }
