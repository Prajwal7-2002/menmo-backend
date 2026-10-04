# rag/agent.py
"""
Answer orchestration.

Decision flow (each step is recorded in `trace` for the UI):

  memory_recall            -> stored facts + past conversation -> answer
  doc_* with a document    -> that document only -> answer (never other docs or the web)
  everything else          -> user's documents
                                good match  -> answer from documents
                                weak match  -> one query rewrite, retry
                              -> web search  -> answer from web results
                              -> general LLM knowledge
"""
import logging
from typing import Any, Dict, List, Optional

from rag import memory
from rag.llm import generate_answer
from rag.retrieval import retrieve
from rag.tools.evaluate_context import evaluate_context_tool
from rag.tools.query_rewrite import rewrite_query_tool
from rag.tools.web_search import web_search_chunks

logger = logging.getLogger(__name__)

DOC_INTENTS = ("doc_summary", "doc_lookup")
NO_DOC_TEXT = (
    "I couldn't find any indexed text for the selected document. It may have failed to "
    "index — try uploading it again."
)
NO_MEMORY_TEXT = (
    "I don't have anything noted about that yet. You can ask me to remember something by "
    "starting a message with \"Remember that …\"."
)


def _public_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the duplicated full text from metadata before returning/logging chunks."""
    out = []
    for c in chunks:
        meta = {k: v for k, v in (c.get("meta") or {}).items() if k not in ("chunk_text", "full_text")}
        out.append({**c, "meta": meta})
    return out


class AgenticRAG:
    def __init__(self, user, domain: Optional[str] = None, document_id: Optional[str] = None,
                 question_intent: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None,
                 tone: str = "neutral", verbosity: str = "normal", allow_web: bool = True):
        self.user = user
        self.user_id = getattr(user, "id", None)
        self.domain = domain
        self.document_id = str(document_id) if document_id else None
        self.intent = (question_intent or "").lower()
        self.history = history or []
        self.tone = tone
        self.verbosity = verbosity
        self.allow_web = allow_web
        self.trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ utils
    def _result(self, answer: str, source: str, chunks: List[Dict[str, Any]], confidence: float) -> Dict[str, Any]:
        return {
            "answer": answer,
            "mode": "agent",
            "source": source,
            "answer_source": source,
            "confidence": round(float(confidence or 0.0), 4),
            "chunks": _public_chunks(chunks),
            "trace": self.trace,
            "steps": len(self.trace),
        }

    def _retrieve(self, query: str, document_id: Optional[str] = None, top_k: int = 6):
        chunks = retrieve(query, user_id=self.user_id, domain=None if document_id else self.domain,
                          document_id=document_id, top_k=top_k)
        ev = evaluate_context_tool(chunks)
        self.trace.append({"step": "rag", "query": query, "chunks_count": len(chunks),
                           "quality": ev["quality"], "top_similarity": round(ev["score"], 3)})
        return chunks, ev

    def _retrieve_with_retry(self, query: str, document_id: Optional[str] = None, top_k: int = 6):
        chunks, ev = self._retrieve(query, document_id, top_k)
        # A rewrite only helps when there is something loosely related to find.
        if chunks and ev["quality"] != "good":
            rewritten = rewrite_query_tool(query)
            if rewritten.lower() != query.lower():
                self.trace.append({"step": "rewrite", "result": rewritten})
                chunks2, ev2 = self._retrieve(rewritten, document_id, top_k)
                if ev2["score"] > ev["score"]:
                    return chunks2, ev2
        return chunks, ev

    # ------------------------------------------------------------- strategies
    def _answer_from_document(self, query: str) -> Dict[str, Any]:
        summary = self.intent == "doc_summary"
        if summary:
            # Summaries need broad coverage, not the single best-matching passage.
            chunks, ev = self._retrieve(f"{query}\nmain topic, purpose, overview and conclusions",
                                        self.document_id, top_k=10)
            chunks.sort(key=lambda c: (c.get("meta") or {}).get("chunk_idx", 0))
        else:
            chunks, ev = self._retrieve_with_retry(query, self.document_id, top_k=6)

        if not chunks:
            self.trace.append({"step": "document_empty"})
            return self._result(NO_DOC_TEXT, "document", [], 0.0)

        answer = generate_answer(query, "document", chunks, self.history, self.tone, self.verbosity)
        self.trace.append({"step": "answer", "mode": "document"})
        return self._result(answer, "document", chunks, ev["score"])

    def _answer_from_memory(self, query: str) -> Dict[str, Any]:
        items = memory.search_memory_facts(self.user, query) + memory.search_conversation_memory(self.user, query)
        self.trace.append({"step": "memory_search", "result_count": len(items)})
        if not items:
            return self._result(NO_MEMORY_TEXT, "memory", [], 0.0)
        answer = generate_answer(query, "memory", items, self.history, self.tone, self.verbosity)
        self.trace.append({"step": "answer", "mode": "memory"})
        return self._result(answer, "memory", items, max(i["score"] for i in items))

    def _answer_open(self, query: str) -> Dict[str, Any]:
        chunks, ev = self._retrieve_with_retry(query, None, top_k=6)
        if ev["quality"] == "good":
            answer = generate_answer(query, "documents", chunks, self.history, self.tone, self.verbosity)
            self.trace.append({"step": "answer", "mode": "documents"})
            return self._result(answer, "document", chunks, ev["score"])

        if self.allow_web:
            web = web_search_chunks(query, max_results=5)
            self.trace.append({"step": "web_search", "result_count": len(web)})
            if web:
                answer = generate_answer(query, "web", web, self.history, self.tone, self.verbosity)
                self.trace.append({"step": "answer", "mode": "web"})
                return self._result(answer, "web", web, 0.5)

        answer = generate_answer(query, "general", [], self.history, self.tone, self.verbosity)
        self.trace.append({"step": "answer", "mode": "general"})
        return self._result(answer, "llm", [], 0.3)

    # ------------------------------------------------------------------- run
    def run(self, query: str) -> Dict[str, Any]:
        if self.intent == "memory_recall":
            return self._answer_from_memory(query)
        if self.document_id and self.intent in DOC_INTENTS:
            return self._answer_from_document(query)
        return self._answer_open(query)


def build_agent(user, domain: Optional[str] = None, document_id: Optional[str] = None,
                question_intent: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None,
                tone: str = "neutral", verbosity: str = "normal", allow_web: bool = True) -> AgenticRAG:
    return AgenticRAG(user=user, domain=domain, document_id=document_id, question_intent=question_intent,
                      history=history, tone=tone, verbosity=verbosity, allow_web=allow_web)
