# rag/tools/intent_classifier.py
from typing import Dict, Any
from rag.llm import call_llm_answer

SYSTEM_PROMPT = """
You are a routing assistant. Given a user's raw query, decide whether the query should be:
- handled_by_rag (answer from uploaded documents)
- handled_by_chat (casual conversation / greeting / smalltalk)
- handled_by_web (needs web / world knowledge)
- handled_by_agent (complex, needs multi-step agent planning)
- handled_by_fallback (unanswerable from docs/web, use LLM)

Return JSON ONLY with these fields:
{
  "intent": "<one of the labels above>",
  "confidence": <float between 0.0 and 1.0>,
  "note": "<optional short reason, 10-25 words>"
}

Be concise. Prefer document-handling when the user mentions document-specific items (file names, page numbers, 'in this document'), otherwise judge conservatively.
"""

def classify_intent_and_route(query: str) -> Dict[str, Any]:
    """
    Uses an LLM to suggest routing. This avoids brittle rule lists in code.
    Returns dict with keys: intent, confidence, note
    """
    if not query or not query.strip():
        return {"intent": "handled_by_chat", "confidence": 0.95, "note": "empty or whitespace"}

    try:
        raw = call_llm_answer(
            question=query,
            context=SYSTEM_PROMPT,
            mood="neutral",
            max_tokens=60
        )
        if not raw:
            return {"intent": "handled_by_agent", "confidence": 0.4, "note": "no model output"}
        raw = raw.strip()
        # try to parse JSON if model returned JSON; otherwise attempt to extract simple tokens
        import json
        try:
            j = json.loads(raw)
            intent = j.get("intent", "").strip()
            confidence = float(j.get("confidence", 0.0))
            note = j.get("note", "")
            if intent:
                return {"intent": intent, "confidence": min(max(confidence, 0.0), 1.0), "note": note}
        except Exception:
            # fallback: very simple heuristic parse
            txt = raw.lower()
            if "rag" in txt or "document" in txt or "file" in txt or "in the doc" in txt:
                return {"intent": "handled_by_rag", "confidence": 0.6, "note": "mentions document-like terms"}
            if "hi" in txt or "hello" in txt or "how are you" in txt:
                return {"intent": "handled_by_chat", "confidence": 0.95, "note": "greeting detected"}
            if "web" in txt or "latest" in txt or "how many people" in txt or "population" in txt:
                return {"intent": "handled_by_web", "confidence": 0.7, "note": "likely needs web"}
            # default
            return {"intent": "handled_by_agent", "confidence": 0.45, "note": "default conservative route"}
    except Exception:
        return {"intent": "handled_by_agent", "confidence": 0.3, "note": "intent classifier failed"}
