from typing import Optional, Dict, Any, List
from .retrieval import retrieve
from .llm import call_llm_answer

# 🔥 REAL LOCATION → must import here
from api_app.models import QueryLog  

LOW_CONF_THRESHOLD = 0.35

REPHRASE_MSG = (
    "I don’t know based on the available documentation. "
    "Please rephrase your question or ask something more specific."
)


def _fail(reason: str, chunks=None, confidence: float = 0.0) -> Dict[str, Any]:
    fallback = chunks[0]["text"] if chunks else REPHRASE_MSG

    # Log query even if fail — so feedback still works
    log = QueryLog.objects.create(
        query="(weak/no match)",
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
        "query_id": str(log.id),
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

    # --- 1) Retrieve ---
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

    # --- 2) Low confidence Fallback ---
    if top["score"] < LOW_CONF_THRESHOLD:
        return _fail("low_confidence_match", candidates, top["score"])

    chosen = candidates[:max_chunks]
    context = "\n\n---\n\n".join([c["text"] for c in chosen])

    # --- History Merge ---
    if history:
        memory = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        context = memory + "\n\n---\n\n" + context

    # --- 3) LLM Generation ---
    try:
        answer = call_llm_answer(query, context, mood=mood)
        final = answer.strip() if answer else chosen[0]["text"]
    except:
        final = chosen[0]["text"]

    # --- 4) Save Query Log ---
    log = QueryLog.objects.create(
        query=query,
        answer=final,
        top_score=top["score"],
        chunks=chosen,
    )

    return {
    "answer": final,
    "validated": top["score"] >= LOW_CONF_THRESHOLD,   # 🔥 only valid when strong relevance
    "confidence": top["score"],
    "chunks": chosen,
    "query_id": str(log.id)
    }

