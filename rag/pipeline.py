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
    max_chunks: int = 4,
) -> Dict[str, Any]:
    """
    Main RAG pipeline.

    Returns:
    {
        "answer": str,
        "validated": bool,
        "chunks": [ {text, score, meta, ...}, ... ],
        "reason": str (optional),
        "note": str (optional)
    }
    """

    # 1) Retrieval (hybrid)
    candidates: List[Dict[str, Any]] = retrieve(
        query=query,
        user_id=user_id,
        domain=domain,
        top_k=max_chunks * 2,
    )

    if not candidates:
        return {
            "answer": REPHRASE_MSG,
            "validated": False,
            "chunks": [],
            "reason": "no_relevant_documents",
        }

    # 2) Validation: score threshold
    top = candidates[0]
    if top["score"] < MIN_SCORE_THRESHOLD:
        # Low-context / irrelevant → block hallucination
        return {
            "answer": REPHRASE_MSG,
            "validated": False,
            "chunks": candidates,
            "reason": "low_relevance",
        }

    # 3) Select chunks for context
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # 4) Call LLM for grounded answer
    try:
        llm_answer = call_llm_answer(query, context)

        if not llm_answer or llm_answer.strip() == "":
            # fallback to top chunk text
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