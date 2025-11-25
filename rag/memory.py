# rag/memory.py
from typing import List, Dict, Optional
from datetime import timedelta
from django.utils import timezone

from api_app.models import QueryLog, UserPreference, ConversationSummary
from .llm import call_llm_answer

# Config
MAX_BUFFER_MESSAGES = 10          # keep last N turns in buffer
SUMMARY_THRESHOLD_MESSAGES = 20   # when history length exceeds, create/refresh summary
SUMMARY_MIN_CHARS = 50
SUMMARY_TTL_DAYS = 30             # how long a summary is considered fresh


# ---------- Buffer Memory ----------
def load_buffer(user, limit: int = MAX_BUFFER_MESSAGES) -> List[Dict[str, str]]:
    """
    Load recent conversation turns for user (most recent first).
    Returns list: [{"role":"user"|"assistant", "content":"..."}] in chronological order.
    """
    logs = (
        QueryLog.objects
        .filter(user=user)
        .order_by("-created_at")[:limit]
    )

    history = []
    # reverse to chronological order
    for entry in reversed(list(logs)):
        # entry.query is user turn, entry.answer is assistant turn
        if entry.query:
            history.append({"role": "user", "content": entry.query})
        if entry.answer:
            history.append({"role": "assistant", "content": entry.answer})
    return history


def prune_buffer(history: List[Dict[str, str]], max_messages: int = MAX_BUFFER_MESSAGES) -> List[Dict[str, str]]:
    """
    Ensure buffer doesn't exceed max_messages (counting both roles).
    Keeps the most recent messages.
    """
    if not isinstance(history, list):
        return []
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# ---------- Summary Memory ----------
def load_summary(user) -> Optional[str]:
    """
    Return the most recent summary text for a user if it's fresh.
    """
    try:
        s = (
            ConversationSummary.objects
            .filter(user=user)
            .order_by("-updated_at")
            .first()
        )
        if not s:
            return None
        # TTL check
        if s.updated_at + timedelta(days=SUMMARY_TTL_DAYS) < timezone.now():
            return None
        return s.summary
    except Exception:
        return None


def save_summary(user, summary_text: str):
    """
    Upsert user's conversation summary.
    """
    if not summary_text or len(summary_text) < SUMMARY_MIN_CHARS:
        return None
    obj, _ = ConversationSummary.objects.update_or_create(
        user=user,
        defaults={"summary": summary_text, "updated_at": timezone.now()},
    )
    return obj


def summarize_history(history: List[Dict[str, str]]) -> Optional[str]:
    """
    Create a concise summary using the LLM.
    Uses call_llm_answer to generate a short summary.
    """
    if not history or len(history) < 3:
        return None

    # Build a compact text version to feed to the summarizer LLM
    snippet = []
    for h in history[-50:]:
        r = h.get("role", "user")
        c = h.get("content", "")
        snippet.append(f"{r}: {c}")

    prompt_context = "\n".join(snippet)
    prompt = (
        "Summarize the conversation below in 2-3 short sentences capturing "
        "the user's goals, important documents referenced, and any action items:\n\n"
        f"{prompt_context}\n\nSUMMARY:"
    )

    try:
        summary = call_llm_answer(prompt, context="", mood="neutral", max_tokens=200)
        if summary and len(summary) > SUMMARY_MIN_CHARS:
            return summary.strip()
    except Exception:
        return None
    return None


# ---------- Preferences ----------
def load_user_preferences(user) -> Dict[str, str]:
    """
    Return user preferences as a dict: {"mood":"friendly", "style":"detailed", ...}
    """
    prefs = {}
    try:
        rows = UserPreference.objects.filter(user=user)
        for p in rows:
            prefs[p.key] = p.value
    except Exception:
        pass
    return prefs


def get_pref(prefs: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    return prefs.get(key, default)
