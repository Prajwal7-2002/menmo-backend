# rag/tools/fallback_llm.py  (updated)
from rag.llm import call_llm_answer

def fallback_llm_tool(query: str, context: str = "") -> str:
    """
    General-purpose fallback that can summarize memory or answer generic queries.
    If context contains memory entries, produce a concise factual answer that uses
    the memory as evidence. Otherwise, answer normally.
    """
    # If context appears to be memory entries, instruct the model to summarize them
    mem_hint = ""
    if context and ("user:" in context or "MEMORY" in context or "memory" in context.lower()):
        mem_hint = (
            "Context appears to contain the user's stored memories or past conversation. "
            "When answering, use these memories as evidence. Produce a single concise sentence or two that directly answers "
            "the user's question. If memory contradicts itself, point out the contradiction briefly."
        )
    else:
        mem_hint = "Context does not appear to contain stored memories. Answer the user query directly."

    full_context = (
        f"{mem_hint}\n\nMemory/Context:\n{context}\n\n"
        "Guidelines:\n"
        "- Return a short, clear answer (1-2 sentences) for memory recall.\n"
        "- If the user greets (hi/hello/hey), reply with a natural greeting.\n"
        "- If the question is about factual knowledge and context is empty, answer normally.\n"
        "- If you are unsure, say 'I don't know' only when necessary.\n"
    )

    response = call_llm_answer(
        question=query,
        context=full_context,
        mood="friendly",
        max_tokens=180
    )

    return response.strip() if response else "I'm here!"
