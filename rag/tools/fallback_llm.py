# rag/tools/fallback_llm.py
from rag.llm import call_llm_answer

def fallback_llm_tool(query: str, context: str = "") -> str:
    """
    Safe fallback used when:
    - The agent can't choose a tool
    - A tool fails
    - Memory needs summarization
    Always returns a short, clean sentence.
    """

    # ---- Detect if context is memory-like ----
    is_memory = False
    if context:
        lowered = context.lower()
        if any(k in lowered for k in ["memory", "fact:", "stored", "recall"]):
            is_memory = True

    # ---- System prompt depending on context ----
    if is_memory:
        system_instruction = (
            "You are summarizing the user's stored memories. "
            "Return ONE short sentence using those memories. "
            "Do not hallucinate. If unsure, say: 'I don't know based on the memory.'"
        )
    else:
        system_instruction = (
            "Provide a short and helpful answer. "
            "If the user greets you, respond normally. "
            "If unsure, say: 'I'm not sure, but I can help if you clarify.'"
        )

    # ---- Build compact LLM context ----
    compact_context = f"{system_instruction}\n\nContext:\n{context or 'None'}"

    # ---- Call model ----
    try:
        result = call_llm_answer(
            question=query,
            context=compact_context,
            mood="friendly",
            max_tokens=100
        )
    except Exception:
        return "I'm here to help — can you clarify that?"

    # ---- Clean output ----
    if not result or not str(result).strip():
        return "I'm here to help — can you rephrase that?"

    text = str(result).strip()

    # ---- Strip accidental JSON ----
    if text.startswith("{") or text.startswith("["):
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = parsed.get("answer") or parsed.get("message") or str(parsed)
            elif isinstance(parsed, list):
                text = " ".join([str(x) for x in parsed])
        except Exception:
            pass

    return text
