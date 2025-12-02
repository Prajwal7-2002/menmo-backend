# rag/tools/intent_classifier.py
import json
from typing import Dict, Any
from rag.llm import call_llm_answer

SYSTEM_PROMPT = """
Classify the user's intent into ONE of the following options (choose the single best match):

1. "chat"           → casual conversation or greetings.
2. "doc"            → question explicitly about an uploaded document (e.g. "in this document", "chapter 3").
3. "knowledge"      → general factual or topical knowledge queries (no docs requested).
4. "agent"          → needs agent behaviour (web, multiple tools, step-by-step reasoning).
5. "memory_store"   → the user asks the assistant to remember or store a fact (e.g. "remember that...", "note that...").
6. "memory_recall"  → the user asks the assistant to recall previously stored memories (e.g. "what did I tell you about X?", "what is my project about?").
7. "fallback"       → none of the above, low confidence.

Return STRICT JSON only, with these fields:
{
  "intent": "<chat|doc|knowledge|agent|memory_store|memory_recall|fallback>",
  "confidence": 0.0,
  "note": "<short reason>"
}
"""

FALLBACK_DEFAULT = {"intent": "agent", "confidence": 0.5, "note": "llm failed or unclear"}

def classify_intent_and_route(query: str) -> Dict[str, Any]:
    """
    Classify intent using the LLM. Robust to LLM mis-formatting.
    If LLM fails or returns invalid content, fallback to simple heuristics.
    """
    if not query or not query.strip():
        return {"intent": "chat", "confidence": 0.95, "note": "empty or whitespace"}

    # Quick heuristic overrides for clear memory store phrases (fast path)
    qlow = query.strip().lower()
    if qlow.startswith(("remember ", "remember that ", "note that ", "store " , "please remember ")):
        return {"intent": "memory_store", "confidence": 0.95, "note": "heuristic: explicit remember/store phrase"}
    if any(kw in qlow for kw in ("what did i", "what was i", "what is my project", "do i have", "what did we discuss", "what does my project", "what have i")):
        return {"intent": "memory_recall", "confidence": 0.9, "note": "heuristic: recall question"}

    try:
        raw = call_llm_answer(
            question=f"Classify this query and return JSON: {query}",
            context=SYSTEM_PROMPT,
            mood="neutral",
            max_tokens=120
        )

        if not raw or not isinstance(raw, str):
            return FALLBACK_DEFAULT

        # Attempt to parse JSON from the LLM output robustly
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # Try to locate a JSON substring
            s = raw
            start = s.find("{")
            end = s.rfind("}") + 1
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(s[start:end])
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            intent = parsed.get("intent", "").strip()
            try:
                conf = float(parsed.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            note = parsed.get("note", "") or parsed.get("explanation", "")

            if intent in ("chat", "doc", "knowledge", "agent", "memory_store", "memory_recall", "fallback"):
                return {"intent": intent, "confidence": max(0.0, min(1.0, conf)), "note": note}

    except Exception:
        pass

    # Last resort: safe fallback
    return FALLBACK_DEFAULT
