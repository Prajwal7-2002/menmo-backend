# rag/llm.py
import os
import requests

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------- Answer prompt ----------------------

ANSWER_PROMPT = """
You are a document-grounded assistant.

You MUST answer ONLY using the provided CONTEXT.
If the context does not contain any relevant information,
reply: "I don’t know based on the available documentation. 
Please rephrase your question or ask something more specific."


CONTEXT:
{context}

USER QUESTION:
{question}

STYLE INSTRUCTIONS:
Respond in the following style/mood: {mood}.
If mood = neutral, respond in a factual tone.

GROUND-TRUTH ANSWER:
"""


def call_llm_answer(
    question: str,
    context: str,
    mood: str = "neutral",
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    """
    Uses Groq to generate an answer grounded in the context.
    mood: "neutral" | "serious" | "joke" | "emotional" | "friendly"
    Returns answer string or None if any error.
    """
    if not GROQ_API_KEY:
        return None

    # Map mood to temperature (override incoming temperature for consistent styles)
    mood_map = {
        "neutral": 0.3,
        "serious": 0.1,
        "joke": 0.9,
        "emotional": 0.7,
        "friendly": 0.6,
    }
    # If user passes an unknown mood, fallback to neutral
    temperature = mood_map.get(mood, mood_map["neutral"])

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You MUST answer only using the given context."},
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(context=context, question=question, mood=mood),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[LLM ANSWER ERROR]", e)
        return None


# Backwards-compat for old imports: call_llm(...)
def call_llm(question: str, context: str, max_tokens: int = 256):
    # keep behaviour stable: neutral mood, deterministic temperature 0.0 for compatibility
    return call_llm_answer(question, context, mood="neutral", max_tokens=max_tokens, temperature=0.0)


# ---------------------- Domain detection prompt ----------------------

DOMAIN_PROMPT = """
You are a classifier.

Read the DOCUMENT below and respond with a very short domain/category
that best describes the document content.

Rules:
- Respond with ONLY 1-3 words.
- No sentences. No explanation.
- Examples: "cardiology", "sports_news", "legal_contract", "api_documentation"

DOCUMENT:
{document}

DOMAIN:
"""


def detect_domain_llm(text: str) -> str:
    """
    Use LLM to automatically infer a broad domain/category
    without any predefined label list.
    Ensures stable, reusable domains.
    """

    prompt = f"""
    You are a classification agent. Your job is to assign a short, high-level domain to a document.

    The domain:
    - MUST be 1 to 3 words
    - MUST be a general category (e.g., programming, medical, legal, finance, science, education, business, engineering, literature, history, research)
    - MUST NOT be overly specific (NO: 'Python Crash Course Chapter 9', YES: 'programming')
    - MUST represent the MAIN subject of the entire document
    - NO predefined labels required — infer the best possible category

    Document content:
    {text[:5000]}  # only first ~5k chars for speed.

    Respond with ONLY the domain name. No explanations.
    """

    domain = call_llm_answer("infer_domain", prompt).strip().lower()

    # final cleanup
    domain = domain.replace(".", "").replace(",", "")
    domain = domain.replace(" ", "_")

    return domain
