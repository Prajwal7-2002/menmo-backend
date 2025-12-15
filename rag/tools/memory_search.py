import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def memory_search_tool(user, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    try:
        from rag import memory
        results = []
        if hasattr(memory, "relevant_chunks"):
            raw = memory.relevant_chunks(user, query, top_k=top_k)
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        score = float(item.get("score", 0.0))
                        meta = item.get("meta", {}) or {}
                        results.append({"text": text, "score": score, "meta": meta})
                    else:
                        results.append({"text": str(item), "score": 0.0, "meta": {}})
        elif hasattr(memory, "load_vector_memory"):
            msgs = memory.load_vector_memory(user, query, top_k=top_k)
            for m in msgs:
                text = m.get("content") if isinstance(m, dict) else str(m)
                results.append({"text": text, "score": 0.0, "meta": {}})
        else:
            return []

        seen = {}
        for r in results:
            key = " ".join(r["text"].split()).strip().lower()[:300]
            if not key:
                continue
            if key not in seen or r["score"] > seen[key]["score"]:
                seen[key] = r

        final = list(seen.values())
        final.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return final[:top_k]
    except Exception as e:
        logger.exception("memory_search_tool failed: %s", e)
        return []
