# rag/tools/fallback_llm.py

from rag.llm import call_llm_answer


def fallback_llm_tool(query: str, context: str = "") -> str:
    """
    General-purpose conversational fallback.
    Behaves like ChatGPT: greeting, reasoning, answering general queries.
    NOT tied to documents unless context explicitly contains docs.
    """

    full_context = (
        f"Conversation context:\n{context}\n\n"
        "Guidelines:\n"
        "- If the user greets (hi/hello/hey), reply with a natural greeting.\n"
        "- If user asks for general info (not in docs), answer normally.\n"
        "- If user asks something requiring reasoning, think step-by-step.\n"
        "- If context includes document text, use it, otherwise answer freely.\n"
        "- DO NOT say: 'based on documentation'.\n"
        "- DO NOT say: 'I don’t know based on documentation'.\n"
        "- Only say 'I don't know' if logically required.\n"
    )

    response = call_llm_answer(
        question=query,
        context=full_context,
        mood="friendly",
        max_tokens=200
    )

    return response.strip() if response else "I'm here!"
