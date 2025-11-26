from typing import Optional, Dict, Any, List
from .retrieval import retrieve
from .llm import call_llm_answer
from .models import QueryLog   # <-- required for feedback logging

LOW_CONF_THRESHOLD = 0.35

REPHRASE_MSG = (
    "I don’t know based on the available documentation. "
    "Please rephrase your question or ask something more specific."
)


def _fail(reason: str, chunks=None, confidence: float = 0.0) -> Dict[str, Any]:
    """Fallback when retrieval is too weak — still logs QueryLog so feedback works."""
    
    fallback = chunks[0]["text"] if chunks else REPHRASE_MSG

    log = QueryLog.objects.create(
        query="(no context match)",
        answer=fallback,
        top_score=confidence,
        chunks=chunks or [],
    )

    return {
        "answer": fallback,
        "validated": False,
        "confidence": confidence,
        "reason": reason,
        "chunks": chunks or [],
        "query_id": str(log.id),  # <-- Needed for feedback
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

    # 1) Retrieve chunks
    candidates = retrieve(
        query=query,
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        top_k=max_chunks * 2,
    )

    if not candidates:
        return _fail("no_relevant_documents", [], 0.0)

    top = candidates[0]

    # 2) Confidence control
    if top["score"] < LOW_CONF_THRESHOLD:
        return _fail("low_confidence_match", candidates, top["score"])

    # 3) Build RAG context
    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # Conversation history support
    if history:
        try:
            memory = "\n".join(
                f"{h.get('role','user')}: {h.get('content','')}" 
                for h in history if h.get("content")
            )
            if memory:
                context = memory + "\n\n---\n\n" + context
        except:
            pass

    # 4) LLM Answering Phase
    try:
        response = call_llm_answer(query, context, mood=mood)
        final_answer = response.strip() if response else chosen[0]["text"]
    except Exception:
        final_answer = chosen[0]["text"]

    # 5) Save + Return Query ID (important!)
    log = QueryLog.objects.create(
        query=query,
        answer=final_answer,
        top_score=top["score"],
        chunks=chosen,
    )

    return {
        "answer": final_answer,
        "validated": True,
        "confidence": top["score"],
        "chunks": chosen,
        "query_id": str(log.id),  # <-- now frontend can send feedback → FIXED
    }
