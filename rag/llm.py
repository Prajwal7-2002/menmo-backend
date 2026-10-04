# rag/llm.py
"""
Groq chat-completions client plus the prompts used to produce answers.

Errors are logged (and raised from `chat_completion`) instead of being turned
into empty strings, so a bad model name or key shows up in the Space logs.
"""
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Tried in order if the main model is retired or unavailable on the account.
GROQ_FALLBACK_MODELS = [m.strip() for m in os.getenv("GROQ_FALLBACK_MODELS", "qwen/qwen3.8-27b").split(",") if m.strip()]
# gpt-oss models "think" before answering; low keeps routing calls fast.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_local = threading.local()
_retired_models: set = set()  # models Groq said don't exist, skipped for this process


class LLMError(RuntimeError):
    pass


class _ModelUnavailable(LLMError):
    pass


def _models() -> List[str]:
    seen, out = set(), []
    for m in [GROQ_MODEL, *GROQ_FALLBACK_MODELS]:
        if m not in seen and m not in _retired_models:
            seen.add(m)
            out.append(m)
    return out


def _is_reasoning(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def _model_params(model: str) -> Dict[str, Any]:
    if _is_reasoning(model):
        return {"reasoning_effort": GROQ_REASONING_EFFORT, "include_reasoning": False}
    return {}


# Reasoning models spend part of max_tokens "thinking" before they write
# anything, so a tiny budget (e.g. 12 tokens for a domain label) comes back
# empty. They still stop as soon as the answer is done.
REASONING_MIN_TOKENS = 400

# gpt-oss cites as 【1】 or 【1†source】; the frontend links [1].
_CITATION_RE = re.compile(r"【(\d{1,2})(?:†[^】]*)?】")


def _normalize(text: str) -> str:
    return _CITATION_RE.sub(r"[\1]", text).strip()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


def chat_completion(messages: List[Dict[str, str]], max_tokens: int = 512,
                    temperature: float = 0.2) -> str:
    """Call Groq and return the assistant text, falling back across models. Raises LLMError."""
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY not set")

    last: Optional[LLMError] = None
    for model in _models():
        try:
            return _complete_with(model, messages, max_tokens, temperature)
        except _ModelUnavailable as e:
            _retired_models.add(model)
            logger.error("Groq model %s unavailable, trying the next one (set GROQ_MODEL to fix): %s", model, e)
            last = e
    raise last or LLMError("no Groq models available; set GROQ_MODEL")


def _complete_with(model: str, messages: List[Dict[str, str]], max_tokens: int,
                   temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max(max_tokens, REASONING_MIN_TOKENS) if _is_reasoning(model) else max_tokens,
        "temperature": temperature,
        **_model_params(model),
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    last_err: Optional[str] = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            resp = _session().post(GROQ_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        except requests.RequestException as e:
            last_err = f"network error: {e}"
            logger.warning("Groq attempt %d/%d: %s", attempt, LLM_MAX_ATTEMPTS, last_err)
            time.sleep(min(2 ** (attempt - 1), 4))
            continue

        if resp.status_code in RETRYABLE_STATUS:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Groq attempt %d/%d: %s", attempt, LLM_MAX_ATTEMPTS, last_err)
            try:
                wait = float(resp.headers.get("retry-after", ""))
            except ValueError:
                wait = 2 ** (attempt - 1)
            time.sleep(min(wait, 8))
            continue

        if resp.status_code >= 400:
            body = resp.text[:300]
            if resp.status_code in (400, 404) and ("model_not_found" in body or "decommissioned" in body):
                raise _ModelUnavailable(f"HTTP {resp.status_code}: {body}")
            # 400/401 etc. won't fix themselves (bad key, bad request…)
            raise LLMError(f"HTTP {resp.status_code}: {body}")

        try:
            data = resp.json()
            return _normalize(data["choices"][0]["message"]["content"] or "")
        except Exception as e:
            raise LLMError(f"unexpected response shape: {e}") from e

    raise LLMError(f"gave up after {LLM_MAX_ATTEMPTS} attempts ({last_err})")


def safe_completion(messages: List[Dict[str, str]], max_tokens: int = 256,
                    temperature: float = 0.0) -> str:
    """For helper calls (classify, rewrite…): log failures and return ''."""
    try:
        return chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
    except LLMError as e:
        logger.error("LLM helper call failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

TONE_STYLE = {
    "neutral": "Clear and factual.",
    "friendly": "Warm, approachable and encouraging, like a helpful colleague.",
    "formal": "Professional and structured; no slang or jokes.",
    "empathetic": "Calm, kind and reassuring; acknowledge how the user feels before the facts.",
    "playful": "Light and a little witty, while keeping the facts accurate.",
    "concise": "Direct and to the point; skip pleasantries.",
}
TONES = tuple(TONE_STYLE)

# Old frontend `mood` values
TONE_ALIASES = {"joke": "playful", "emotional": "empathetic"}

VERBOSITY_STYLE = {
    "brief": ("Keep it short: one to three sentences unless a list is clearly needed.", 300),
    "normal": ("Use a moderate length; use a short list if it helps.", 700),
    "detailed": ("Give a thorough, well-structured answer with headings or steps where useful.", 1200),
}


def resolve_tone(requested: str, inferred: str) -> str:
    """An explicit mood from the client wins; 'auto', 'neutral' or nothing lets the router decide."""
    requested = TONE_ALIASES.get((requested or "").strip().lower(), (requested or "").strip().lower())
    if requested in TONE_STYLE and requested != "neutral":
        return requested
    return inferred if inferred in TONE_STYLE else "neutral"

MODE_INSTRUCTIONS = {
    # The user picked one document and asked about it.
    "document": (
        "Answer ONLY from the numbered document excerpts below. Cite the excerpts you use like [1]. "
        "If the excerpts do not contain the answer, say plainly that the selected document does not "
        "mention it. Never invent values, names or numbers that are not in the excerpts."
    ),
    # Excerpts from the user's own documents matched the question.
    "documents": (
        "Prefer the numbered excerpts from the user's documents below and cite them like [1]. "
        "If they only partly answer the question you may add general knowledge, but say which part "
        "comes from general knowledge, and never contradict the excerpts."
    ),
    "web": (
        "Answer using the numbered web search results below and cite them like [1]. "
        "If the results don't answer the question, say so and give your best general-knowledge answer, "
        "clearly marked as such."
    ),
    "memory": (
        "The notes below are things the user previously told you or discussed with you. "
        "Use them to answer. If they don't contain the answer, say you don't have that noted."
    ),
    "general": (
        "No documents or search results are available for this question. Answer from general knowledge. "
        "If you are not confident, say so rather than guessing."
    ),
    "chat": "This is casual conversation. Reply briefly and naturally.",
}


def format_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Number the chunks so the model can cite them."""
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        meta = c.get("meta") or {}
        label = meta.get("title") or meta.get("source") or ""
        if meta.get("page_num"):
            label = f"{label} p.{meta['page_num']}".strip()
        header = f"[{i}]" + (f" ({label})" if label else "")
        block = f"{header}\n{text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def generate_answer(question: str, mode: str, chunks: Optional[List[Dict[str, Any]]] = None,
                    history: Optional[List[Dict[str, str]]] = None, tone: str = "neutral",
                    verbosity: str = "normal", max_tokens: Optional[int] = None) -> str:
    length_rule, default_tokens = VERBOSITY_STYLE.get(verbosity, VERBOSITY_STYLE["normal"])
    system = (
        "You are Mnemo, an assistant that answers questions about the user's uploaded "
        "documents and general topics.\n"
        f"{MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS['general'])}\n"
        f"Tone: {TONE_STYLE.get(tone, TONE_STYLE['neutral'])}\n"
        f"Length: {length_rule}"
    )
    max_tokens = max_tokens or default_tokens
    context = format_context(chunks or [])
    if context:
        system += f"\n\n----- CONTEXT -----\n{context}\n----- END CONTEXT -----"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for m in (history or [])[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:2000]})
    messages.append({"role": "user", "content": question})

    try:
        return chat_completion(messages, max_tokens=max_tokens, temperature=0.2)
    except LLMError as e:
        logger.error("generate_answer failed (mode=%s): %s", mode, e)
        return "Sorry — the language model is unavailable right now. Please try again in a moment."


def detect_domain_llm(text: str) -> str:
    raw = safe_completion([
        {"role": "system", "content": "Reply with ONE short domain/category for the document (1-3 words, no punctuation)."},
        {"role": "user", "content": text[:5000]},
    ], max_tokens=12)
    d = raw.strip().lower().replace(".", "").replace(",", "").replace(" ", "_")
    return d if len(d) >= 2 else "general"
