from typing import List, Dict, Any

# Much more forgiving thresholds
MIN_GOOD_SCORE = 0.25
MIN_WEAK_SCORE = 0.10

def evaluate_context_tool(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates RAG chunks and returns:
      {quality: good|weak|empty, score: float}
    Agent expects 'good' to decide final answer.
    """

    try:
        if not chunks:
            return {"quality": "empty", "score": 0.0, "reason": "no_chunks"}

        # Hybrid score from retrieval
        scores = [float(c.get("score") or 0.0) for c in chunks]
        avg_score = sum(scores) / len(scores)

        # top chunk length (text richness indicator)
        top_text = (chunks[0].get("text") or "").strip()
        top_len = len(top_text)

        # More forgiving composite formula
        composite = (avg_score * 0.6) + (min(top_len, 500) / 500.0) * 0.4

        if composite >= MIN_GOOD_SCORE:
            return {"quality": "good", "score": float(composite), "reason": "acceptable"}
        elif composite >= MIN_WEAK_SCORE:
            return {"quality": "weak", "score": float(composite), "reason": "weak_but_present"}
        else:
            return {"quality": "empty", "score": float(composite), "reason": "too_low"}

    except Exception as e:
        return {"quality": "empty", "score": 0.0, "reason": f"error:{e}"}
