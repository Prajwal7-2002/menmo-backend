import os
import time
import requests
from typing import Optional

# KEEP YOUR MODEL / KEYS — user insisted we don't change models
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# Answer prompt (kept minimal and strict)
ANSWER_PROMPT = """
You are a document-grounded assistant.
You MUST answer ONLY using the provided CONTEXT.
If the context does not contain relevant information, reply:
"I don’t know based on the available documentation."
CONTEXT:
{context}

USER QUESTION:
{question}

STYLE INSTRUCTIONS:
Respond in the following style/mood: {mood}.
GROUND-TRUTH ANSWER:
"""

# small helper to build payload (keeps deterministic settings)
def _build_payload(question: str, context: str, mood: str, max_tokens: int):
    mood_style = {
        "neutral": "Clear and factual.",
        "friendly": "Warm, helpful, encouraging tone.",
        "formal": "Structured professional tone.",
        "joke": "Light humorous tone.",
        "emotional": "Expressive, empathetic tone.",
    }.get(mood, "Clear and factual.")

    system = (
        "You are a Retrieval-Augmented assistant.\n\n"
        "- Use ONLY the information inside the provided context\n"
        "- Do NOT invent facts or hallucinate\n"
        '- If answer is not found, respond only with: "I don’t know based on the available documentation."\n\n'
        f"Tone style → {mood_style}\n\n"
        "---------------- CONTEXT ----------------\n"
        f"{context}\n"
        "-----------------------------------------\n"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    return payload

def _post_with_backoff(json_payload, headers, max_attempts=5):
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=json_payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Retry on 429 or 502/503/504
            if status in (429, 502, 503, 504):
                wait = backoff * (1.5 ** (attempt - 1))
                time.sleep(wait)
                continue
            else:
                raise
        except Exception:
            # network / timeout -> retry
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    # after attempts yield None
    return None

def call_llm_answer(
    question: str = None,
    context: str = "",
    mood: str = "neutral",
    max_tokens: int = 256,
    messages: list = None
) -> str:
    """
    Unified LLM wrapper.

    Supports:
    - messages=[...]  (Agent mode)
    - question/context (RAG mode)

    Falls back safely to "" if call fails.
    """

    if not GROQ_API_KEY:
        print("[LLM] Missing GROQ_API_KEY — skipping LLM call")
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # =====================================================
    # 1) AGENT MODE (messages list)
    # =====================================================
    if messages is not None:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            res = _post_with_backoff(payload, headers)
            if not res:
                return ""
            choices = res.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            print("❌ LLM messages error:", e)
            return ""

    # =====================================================
    # 2) RAG MODE (existing behavior preserved)
    # =====================================================
    payload = _build_payload(question, context, mood, max_tokens)

    try:
        res = _post_with_backoff(payload, headers)
        if not res:
            print("❌ LLM: request failed after retries")
            return ""

        choices = res.get("choices") or []
        if not choices:
            return ""

        return choices[0].get("message", {}).get("content", "").strip()

    except Exception as e:
        print("❌ LLM Answer Error:", e)
        return ""

# ---------------------------------------------------------------------
# Domain Detection Helper (used by loader.py)
# ---------------------------------------------------------------------
def detect_domain_llm(text: str) -> str:
    """
    Uses the LLM to infer the domain of a document.
    Fallback = 'general'
    """
    prompt = f"""
Read the DOCUMENT below and respond with ONE short domain/category (1-3 words). 
No punctuation.

DOCUMENT:
{text[:5000]}
"""

    resp = call_llm_answer(
        question="Infer domain",
        context=prompt,
        mood="neutral",
        max_tokens=12,
    )

    if not resp:
        return "general"

    d = (
        resp.strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "_")
    )

    if not d or len(d) < 2:
        return "general"

    return d
