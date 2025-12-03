# rag/tools/evaluate_context.py
from typing import List, Dict, Any

# Lightweight evaluator for retrieved chunks.
# Returns { "quality": "good"|"weak"|"empty", "score": float, "reason": str }

MIN_GOOD_SCORE = 0.45   # tune as needed
MIN_WEAK_SCORE = 0.20   # between weak and empty

def evaluate_context_tool(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple deterministic evaluator:
    - if no chunks -> empty
    - compute avg hybrid score (field 'score' expected), and look at top chunk length
    - return quality and numeric score (0..1)
    """
    try:
        if not chunks:
            return {"quality": "empty", "score": 0.0, "reason": "no_chunks"}

        scores = [float(c.get("score") or 0.0) for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        top = chunks[0]
        top_text = (top.get("text") or "").strip()
        top_len = len(top_text)
        # simple heuristics combining avg_score and top length
        composite = avg_score * 0.7 + (min(top_len, 500) / 500.0) * 0.3

        if composite >= MIN_GOOD_SCORE:
            return {"quality": "good", "score": float(composite), "reason": "high_composite"}
        elif composite >= MIN_WEAK_SCORE:
            return {"quality": "weak", "score": float(composite), "reason": "low_composite"}
        else:
            return {"quality": "empty", "score": float(composite), "reason": "too_low"}
    except Exception as e:
        return {"quality": "empty", "score": 0.0, "reason": f"error:{str(e)}"}
