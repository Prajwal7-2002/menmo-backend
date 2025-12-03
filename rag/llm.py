# rag/llm.py
import os
import time
import requests
from typing import Optional, List

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

def _build_payload(question: str, context: str, mood: str, max_tokens: int):
    mood_style = {
        "neutral": "Clear and factual.",
        "friendly": "Warm, helpful, encouraging tone.",
        "formal": "Structured professional tone.",
        "joke": "Light humorous tone.",
        "emotional": "Expressive, empathetic tone.",
    }.get(mood, "Clear and factual.")

    system = (
        "You are a hybrid RAG + general-knowledge assistant.\n\n"
        "RULES:\n"
        "1. If context is provided → Prefer answering from the context.\n"
        "2. If context is empty OR irrelevant → Use normal world knowledge.\n"
        "3. NEVER answer with 'I don’t know based on the available documentation' "
        "unless BOTH are true:\n"
        "   - context exists\n"
        "   - and context is insufficient.\n"
        "4. Do NOT hallucinate when context contradicts facts.\n\n"
        f"Tone → {mood_style}\n\n"
        "--------------- CONTEXT ----------------\n"
        f"{context}\n"
        "-----------------------------------------\n"
    )

    return {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def _post_with_backoff(json_payload, headers, max_attempts=5):
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=json_payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (429, 502, 503, 504):
                wait = backoff * (1.5 ** (attempt - 1))
                time.sleep(wait)
                continue
            else:
                raise
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    return None

def call_llm_answer(
    question: str = None,
    context: str = "",
    mood: str = "neutral",
    max_tokens: int = 256,
    messages: list = None
) -> str:
    """
    Unified LLM wrapper used across the project.
    If messages is provided -> use messages mode (agent).
    Else use question+context (RAG).
    Returns empty string on failures (safe fallback).
    """
    if not GROQ_API_KEY:
        # missing API key -> safe empty response
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Agent/messages mode
    if messages is not None:
        payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens}
        try:
            res = _post_with_backoff(payload, headers)
            if not res:
                return ""
            choices = res.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            return ""

    # RAG/question+context mode
    payload = _build_payload(question or "", context or "", mood or "neutral", max_tokens)
    try:
        res = _post_with_backoff(payload, headers)
        if not res:
            return ""
        choices = res.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""

# convenience small helper used by loader
def detect_domain_llm(text: str) -> str:
    prompt = f"Read the DOCUMENT below and respond with ONE short domain/category (1-3 words). No punctuation.\n\nDOCUMENT:\n{text[:5000]}"
    resp = call_llm_answer(question="Infer domain", context=prompt, mood="neutral", max_tokens=12)
    if not resp:
        return "general"
    d = resp.strip().lower().replace(".", "").replace(",", "").replace(" ", "_")
    return d if d and len(d) >= 2 else "general"
