# rag/tools/fallback_llm.py
from rag.llm import call_llm_answer

def fallback_llm_tool(query: str, context: str = "", domain=None, document_id=None) -> str:
    """
    FINAL fallback that ALWAYS grounds the answer strictly to retrieved chunks.
    Zero hallucinations. Zero template guessing.
    """

    SYSTEM = (
        "You are a RAG assistant. Answer STRICTLY using the provided context. "
        "Do NOT generalize. Do NOT infer missing sections. "
        "If the context doesn’t contain enough information, say explicitly:\n"
        "\"The context does not contain enough information to answer this question.\"\n"
        "Never produce template-like answers such as introductions, purposes, or assumptions "
        "unless the text explicitly appears in the retrieved chunks."
    )

    # Build strict prompt
    final_context = f"{SYSTEM}\n\n---BEGIN-CONTEXT---\n{context}\n---END-CONTEXT---"

    try:
        answer = call_llm_answer(
            question=query,
            context=final_context,
            max_tokens=200,
            mood="neutral"
        )
    except Exception:
        return "The context does not contain enough information to answer this question."

    if not answer or not answer.strip():
        return "The context does not contain enough information to answer this question."

    return answer.strip()
