# rag/tools/fallback_llm.py
from rag.llm import call_llm_answer


def fallback_llm_tool(query: str, context: str = "", domain=None, document_id=None) -> str:
    """
    Fallback LLM that prefers retrieved context but can use general knowledge
    when the context is weak or incomplete.
    """

    SYSTEM = (
        "You are a RAG assistant.\n"
        "- Prefer to answer using the provided context.\n"
        "- If the context is clearly unrelated to the question or missing key information,\n"
        "  you may use your general knowledge to answer, but do not contradict anything\n"
        "  that appears in the context.\n"
        "- If you truly cannot answer even with general knowledge, say exactly:\n"
        "  \"The context does not contain enough information to answer this question.\""
    )

    # Build prompt with system instructions plus raw context
    final_context = f"{SYSTEM}\n\n---BEGIN-CONTEXT---\n{context}\n---END-CONTEXT---"

    try:
        answer = call_llm_answer(
            question=query,
            context=final_context,
            max_tokens=200,
            mood="neutral",
        )
    except Exception:
        return "The context does not contain enough information to answer this question."

    if not answer or not answer.strip():
        return "The context does not contain enough information to answer this question."

    return answer.strip()

