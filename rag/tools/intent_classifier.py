# rag/tools/intent_classifier.py
"""
Query router: one LLM call decides how to handle a message.

It returns
  intent      chat | doc_summary | doc_lookup | knowledge | agent | memory_store | memory_recall
  tone        how the reply should sound, read from the user's wording and the chat so far
  verbosity   brief | normal | detailed
  standalone  the message rewritten so it makes sense without the chat history
  fact        for memory_store: the thing to remember, phrased about the user

The regex rules below are only a fallback for when the LLM is unavailable.
"""
import json
import re
from typing import Any, Dict, List, Optional

from rag.llm import TONES, safe_completion

INTENTS = {"chat", "doc_summary", "doc_lookup", "knowledge", "agent", "memory_store", "memory_recall"}
VERBOSITY = {"brief", "normal", "detailed"}

ROUTER_PROMPT = """You route messages for Mnemo, an assistant that answers questions about the user's uploaded documents and general topics.

Read the conversation and the latest message, then return STRICT JSON:
{
  "intent": one of
     "chat"          greetings, thanks, small talk, reactions
     "doc_summary"   what the selected document is about / summarise it
     "doc_lookup"    a specific fact, value or detail from the selected document
     "knowledge"     a factual or conceptual question not tied to the selected document
     "agent"         research, comparisons or multi-step tasks that need searching
     "memory_store"  the user wants you to remember something about them
     "memory_recall" the user asks what they told you before,
  "standalone": the latest message rewritten so it is fully understandable without the conversation (resolve "it", "that", "the second one"...); unchanged if already standalone,
  "tone": the best reply tone given how the user writes and what they ask, one of %(tones)s,
  "verbosity": "brief" for quick questions or casual chat, "detailed" when they ask to explain/compare/walk through, else "normal",
  "fact": for memory_store only, the thing to remember phrased about the user (e.g. "The user's project is about medical RAG"), else "",
  "confidence": 0.0-1.0
}

Guidance:
- Only use doc_summary/doc_lookup when a document is selected AND the message is about it. A follow-up about the document ("and the cholesterol?") still counts.
- Match tone to the person: stressed or worried -> empathetic; joking -> playful; terse/technical -> concise; professional request -> formal; otherwise friendly or neutral.
Return only the JSON."""

# --------------------------------------------------------------------------
# Fallback rules (LLM unavailable)
# --------------------------------------------------------------------------
GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|yo|hiya|thanks?|thank you|thx|ty|ok(ay)?|cool|great|nice|bye|goodbye|"
    r"good (morning|afternoon|evening|night)|how are you( doing)?|what'?s up|sup)"
    r"( there| so much| a lot| mnemo)?[\s!.?,:)]*$",
    re.I,
)
REMEMBER_RE = re.compile(r"^(please\s+)?(remember|note|keep in mind|don'?t forget)\b(\s+that)?[:,]?\s*", re.I)
RECALL_RE = re.compile(
    r"\b(what (did|have) i (tell|told|say|said|mention|mentioned)|do you remember|what do you (remember|know) about me|"
    r"what did we (discuss|talk about)|remind me what i)\b",
    re.I,
)
DOC_REF_RE = re.compile(r"\b(this|the|attached|uploaded|selected|my)\s+(document|doc|report|pdf|file|paper|article)\b", re.I)
SUMMARY_RE = re.compile(r"\b(summar|overview|tl;?dr|main (points|idea|topic)|key (points|takeaways))", re.I)


def extract_fact(query: str) -> str:
    """'Remember that my project is X' -> 'my project is X'."""
    return REMEMBER_RE.sub("", query.strip(), count=1).strip()


def _rule_route(q: str, has_document: bool) -> Dict[str, Any]:
    if GREETING_RE.match(q):
        intent = "chat"
    elif REMEMBER_RE.match(q) and extract_fact(q):
        intent = "memory_store"
    elif RECALL_RE.search(q):
        intent = "memory_recall"
    elif has_document and DOC_REF_RE.search(q):
        intent = "doc_summary" if SUMMARY_RE.search(q) else "doc_lookup"
    else:
        intent = "knowledge"
    return {"intent": intent, "confidence": 0.5, "tone": "neutral", "verbosity": "normal",
            "standalone": q, "fact": extract_fact(q) if intent == "memory_store" else "", "note": "rules"}


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def classify_intent_and_route(query: str, has_document: bool = False,
                              history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {**_rule_route("hi", has_document), "standalone": "", "note": "empty"}

    turns = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")][-4:]
    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in turns) or "(none)"
    raw = safe_completion([
        {"role": "system", "content": ROUTER_PROMPT % {"tones": ", ".join(f'"{t}"' for t in TONES)}},
        {"role": "user", "content": (f"Document selected: {'yes' if has_document else 'no'}\n"
                                     f"Conversation:\n{transcript}\n\nLatest message: {q}")},
    ], max_tokens=200)

    data = _parse_json(raw) if raw else None
    if not data or str(data.get("intent", "")).strip().lower() not in INTENTS:
        return _rule_route(q, has_document)

    intent = str(data["intent"]).strip().lower()
    if intent.startswith("doc_") and not has_document:
        intent = "knowledge"

    standalone = str(data.get("standalone") or "").strip()
    if not (1 < len(standalone) < 500):
        standalone = q

    fact = str(data.get("fact") or "").strip()
    if intent == "memory_store" and not fact:
        fact = extract_fact(q)

    tone = str(data.get("tone", "")).strip().lower()
    verbosity = str(data.get("verbosity", "")).strip().lower()
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
    except (TypeError, ValueError):
        confidence = 0.7

    return {
        "intent": intent,
        "confidence": confidence,
        "tone": tone if tone in TONES else "neutral",
        "verbosity": verbosity if verbosity in VERBOSITY else "normal",
        "standalone": standalone,
        "fact": fact,
        "note": "llm",
    }
