# rag/tools/intent_classifier.py
import json
from typing import Dict, Any

from rag.llm import call_llm_answer


SYSTEM_PROMPT = """
You are an intent classifier for a RAG + agent chatbot.
Classify the user's query into ONE of the following options (choose the single best match):

1. "chat"
   - Casual conversation, greetings, thanks, small-talk.
   - Examples: "hi", "hello", "how are you", "thanks", "ok cool".

2. "doc_summary"
   - Question asking for the overall topic, purpose, or context of a specific uploaded document.
   - User usually refers to "this document", "this report", "this PDF", "context of this document",
     "what is this document about", "summarise this document", etc.
   - Answer should come ONLY from the selected document, not general web knowledge.

3. "doc_lookup"
   - Question asking for a specific fact, value, or detail inside the selected document/report.
   - Examples: "what is the HbA1c value in this report", "what is the testosterone level in this document",
     "according to this pdf, what is requirement X", "in this document what does CLIA stand for".
   - The user clearly wants information that must be supported by the document contents.

4. "knowledge"
   - General factual or topical questions that do NOT need to be tied to a particular uploaded document.
   - Examples: "what is NLP", "what is GraphRAG", "who is Alan Turing",
     "explain reinforcement learning", "what is hemoglobin".
   - These may use the web or general knowledge.

5. "agent"
   - Complex tasks that benefit from tools, web search, or multi-step reasoning,
     not just a simple factual answer.
   - Examples: "research the latest RAG papers and summarise them",
     "compare three vector databases and recommend one".

6. "memory_store"
   - The user asks the assistant to remember or store a fact.
   - Examples: "remember that my project is about medical RAG",
     "note that my favourite language is Python".

7. "memory_recall"
   - The user asks the assistant to recall previously stored facts.
   - Examples: "what did I tell you about my project",
     "what is my favourite language".

8. "fallback"
   - None of the above fits or you are very uncertain.

Important rules:
- ONLY choose "doc_summary" or "doc_lookup" if the wording clearly refers to
  the selected document/report/PDF (for example: "this document", "this report",
  "in this pdf", "in this report", "according to this report", etc.).
- If the user just asks a generic definition or concept question such as
  "what is NLP" or "what is hemoglobin" without clearly tying it to the
  current document, prefer "knowledge" instead of any doc_* intent.
- Prefer "chat" for short greetings or acknowledgements even if a document is selected.

Return STRICT JSON only, with these fields:
{
  "intent": "<chat|doc_summary|doc_lookup|knowledge|agent|memory_store|memory_recall|fallback>",
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

    # Quick heuristic overrides for clear memory/memory-like phrases (fast path)
    qlow = query.strip().lower()
    if qlow.startswith(("remember ", "remember that ", "note that ", "store ", "please remember ")):
        return {
            "intent": "memory_store",
            "confidence": 0.95,
            "note": "heuristic: explicit remember/store phrase",
        }
    if any(
        kw in qlow
        for kw in (
            "what did i",
            "what was i",
            "what is my project",
            "do i have",
            "what did we discuss",
            "what does my project",
            "what have i",
        )
    ):
        return {
            "intent": "memory_recall",
            "confidence": 0.9,
            "note": "heuristic: recall question",
        }

    # Heuristic: generic definition questions ("what is X", "who is X")
    # that do NOT explicitly tie themselves to "this document/report"
    # should be treated as general knowledge, not doc_summary/doc_lookup.
    if qlow.startswith(("what is ", "who is ")):
        if not any(
            kw in qlow
            for kw in (
                "this document",
                "this report",
                "in this document",
                "in this report",
                "according to this document",
                "according to this report",
            )
        ):
            return {
                "intent": "knowledge",
                "confidence": 0.9,
                "note": "heuristic: generic definition, not tied to document",
            }

    try:
        raw = call_llm_answer(
            question=f"Classify this query and return JSON: {query}",
            context=SYSTEM_PROMPT,
            mood="neutral",
            max_tokens=120,
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

            # Backwards compatibility: if the LLM still returns legacy "doc",
            # treat it as a document lookup question.
            if intent == "doc":
                intent = "doc_lookup"

            allowed = {
                "chat",
                "doc_summary",
                "doc_lookup",
                "knowledge",
                "agent",
                "memory_store",
                "memory_recall",
                "fallback",
            }
            if intent in allowed:
                return {
                    "intent": intent,
                    "confidence": max(0.0, min(1.0, conf)),
                    "note": note,
                }

    except Exception:
        pass

    # Last resort: safe fallback
    return FALLBACK_DEFAULT
