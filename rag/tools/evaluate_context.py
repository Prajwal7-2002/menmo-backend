import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

MIN_GOOD_SCORE = float(__import__("os").environ.get("MIN_GOOD_SCORE", 0.60))
MIN_WEAK_SCORE = float(__import__("os").environ.get("MIN_WEAK_SCORE", 0.25))

def evaluate_context_tool(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        if not chunks:
            return {"quality": "empty", "score": 0.0, "reason": "no_chunks"}

        scores = [float(c.get("score") or 0.0) for c in chunks]
        avg_score = sum(scores) / len(scores)

        top_text = (chunks[0].get("text") or "").strip()
        top_len = len(top_text)

        composite = (avg_score * 0.6) + (min(top_len, 1000) / 1000.0) * 0.4

        if composite >= MIN_GOOD_SCORE:
            return {"quality": "good", "score": float(composite), "reason": "acceptable"}
        elif composite >= MIN_WEAK_SCORE:
            return {"quality": "weak", "score": float(composite), "reason": "weak_but_present"}
        else:
            return {"quality": "empty", "score": float(composite), "reason": "too_low"}

    except Exception as e:
        logger.exception("evaluate_context_tool error: %s", e)
        return {"quality": "empty", "score": 0.0, "reason": f"error:{e}"}
