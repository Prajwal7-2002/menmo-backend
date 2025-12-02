# rag/tools/fallback_llm.py (SAFE & CLEAN VERSION)
from rag.llm import call_llm_answer

def fallback_llm_tool(query: str, context: str = "") -> str:
    """
    A very safe fallback that ALWAYS returns a clean, short text answer.
    Used when planner cannot pick a valid tool or when memory needs summarization.
    """

    # ---- 1. Detect memory context more reliably ----
    is_memory = False
    if context:
        lowered = context.lower()
        if any(k in lowered for k in ["memory", "fact:", "stored", "recall"]):
            is_memory = True

    # ---- 2. Build minimal prompt (VERY IMPORTANT) ----
    if is_memory:
        system_instruction = (
            "You are summarizing the user's stored memories. "
            "Return ONE short sentence using those memories as evidence. "
            "Do not hallucinate. If unsure, say: 'I don't know based on the memory.'"
        )
    else:
        system_instruction = (
            "Provide a short, helpful answer. "
            "If the user greets you (hi, hello, hey), respond naturally. "
            "Never answer with an empty string. "
            "If unsure, say: 'I'm not sure, but I can help if you clarify.'"
        )

    # ---- 3. Build context ----
    compact_context = f"{system_instruction}\n\nContext:\n{context or 'None'}"

    # ---- 4. Try generating ----
    try:
        result = call_llm_answer(
            question=query,
            context=compact_context,
            mood="friendly",
            max_tokens=100
        )
    except Exception:
        return "I'm here to help — can you clarify that?"

    # ---- 5. Clean & normalize output ----
    if not result or not str(result).strip():
        return "I'm here to help — can you rephrase that?"

    # strip weird characters or whitespace
    text = str(result).strip()

    # remove stray JSON formatting or quotes
    if text.startswith("{") or text.startswith("["):
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = parsed.get("answer") or parsed.get("message") or str(parsed)
            else:
                text = str(parsed)
        except Exception:
            pass  # fallback to text

    return text
