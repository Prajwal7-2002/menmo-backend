# rag/pipeline.py
from typing import Optional, Dict, Any, List

from .retrieval import retrieve  # MIN_SCORE_THRESHOLD removed — not needed anymore
from .llm import call_llm_answer


# Minimum match confidence before rejecting completely
LOW_CONF_THRESHOLD = 0.35   # was too strict before → now more LLM answers


REPHRASE_MSG = (
    "I don’t know based on the available documentation. "
    "Please rephrase your question or ask something more specific."
)


def _fail(reason: str, chunks=None, confidence: float = 0.0) -> Dict[str, Any]:
    """
    Returned only if retrieval confidence is extremely low.
    Still shows chunks so user can refine their query or give feedback.
    """
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

    # ------------------ 1) Retrieve relevant chunks --------------------
    candidates = retrieve(
        query=query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max_chunks * 2, # retrieve more → filter later
    )

    if not candidates:
        return _fail("no_relevant_documents", [], 0.0)

    top = candidates[0]

    # ------------------ 2) Confidence Gate (relaxed 🔥) --------------------
    if top["score"] < LOW_CONF_THRESHOLD:
        # Still return chunks so LLM can reference if user rewrites question
        return _fail("low_confidence_match", candidates, top["score"])

    # ------------------ 3) Build grounding context --------------------
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # attach previous conversation memory if present
    if history:
        try:
            memory = "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '')}"
                for h in history if h.get("content")
            )
            if memory:
                context = memory + "\n\n---\n\n" + context
        except:
            pass

    # ------------------ 4) Final LLM Generation --------------------
    try:
        response = call_llm_answer(query, context, mood=mood)

        # If LLM gives empty → fallback to most relevant chunk
        if not response or not response.strip():
            return {
                "answer": chosen[0]["text"],
                "validated": True,
                "confidence": top["score"],
                "chunks": chosen,
                "note": "fallback_used_empty_llm",
            }

        return {
            "answer": response.strip(),
            "validated": True,
            "confidence": top["score"],
            "chunks": chosen,
        }

    except Exception as e:
        # Retrieval succeeded but LLM failed → safe fallback
        return {
            "answer": chosen[0]["text"],
            "validated": True,
            "confidence": top["score"],
            "chunks": chosen,
            "note": f"fallback_used_due_to_llm_error: {e}",
        }
