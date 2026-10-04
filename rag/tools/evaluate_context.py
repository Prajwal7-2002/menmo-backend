# rag/tools/evaluate_context.py
"""
Decide whether retrieved chunks are relevant enough to answer from.

Based on the raw cosine similarity of the best chunk (see retrieval.py).
Defaults are tuned for all-MiniLM-L6-v2, where a relevant passage usually
scores 0.45-0.75 against a question and unrelated text below 0.25.
"""
import os
from typing import Any, Dict, List

MIN_GOOD_SIM = float(os.getenv("RAG_MIN_GOOD_SIM", "0.45"))
MIN_WEAK_SIM = float(os.getenv("RAG_MIN_WEAK_SIM", "0.30"))


def evaluate_context_tool(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        return {"quality": "empty", "score": 0.0, "reason": "no_chunks"}
    top = max(float(c.get("similarity", c.get("score", 0.0)) or 0.0) for c in chunks)
    if top >= MIN_GOOD_SIM:
        return {"quality": "good", "score": top, "reason": "top chunk clearly relevant"}
    if top >= MIN_WEAK_SIM:
        return {"quality": "weak", "score": top, "reason": "top chunk loosely related"}
    return {"quality": "empty", "score": top, "reason": "nothing relevant"}
