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
):
    if not GROQ_API_KEY:
        print("[LLM] Missing API KEY")
        return ""

    mood_style = {
        "neutral": "Clear and factual.",
        "friendly": "Warm, helpful, encouraging tone.",
        "formal": "Structured professional tone.",
        "joke": "Light humorous tone.",
        "emotional": "Expressive, empathetic tone.",
    }.get(mood, "neutral")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": 
                f"""
                You are a Retrieval-Augmented assistant.

                RULES FOR ANSWERING:
                - Use ONLY the information inside the provided context
                - Do NOT invent facts or hallucinate
                - If answer is not found, respond only with:
                  "I don’t know based on the available documentation."

                Tone style → {mood_style}

                ---------------- CONTEXT ----------------
                {context}
                -----------------------------------------
                """,
            },
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,   # deterministic, factual
        "top_p": 0.9
    }

    try:
        resp = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("\n❌ LLM Answer Error:", e)
        return ""


# Backwards-compat for old imports: call_llm(...).
def call_llm(question: str, context: str, max_tokens: int = 256) -> str:
    # keep behaviour stable: neutral mood
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

    Returns a safe string, defaults to "general" if anything fails.
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
    {text[:5000]}

    Respond with ONLY the domain name. No explanations.
    """

    raw = call_llm_answer(
        question="Infer a domain for this document.",
        context=prompt,
        mood="neutral",
        max_tokens=16,
    )

    if not raw:
        return "general"

    domain = raw.strip().lower()
    domain = domain.replace(".", "").replace(",", "")
    domain = domain.replace(" ", "_")

    if not domain:
        return "general"

    return domain
