# api_app/views.py
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.db.models import Avg, Count
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api_app.models import Conversation, Feedback, Message, QueryLog
from rag import memory
from rag.agent import build_agent
from rag.llm import generate_answer, resolve_tone
from rag.loader import SUPPORTED_EXTENSIONS, IndexingError, index_document
from rag.models import ChunkFeedback, DocumentChunk, UploadedDocument
from rag.retrieval import delete_chunks, delete_document_vectors
from rag.tools.intent_classifier import classify_intent_and_route

from .serializers import AskSerializer, FeedbackSerializer

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
HISTORY_MESSAGES = int(os.getenv("CHAT_HISTORY_MESSAGES", "6"))
# Saved with each answer so reopened chats still show source, tone, citations and feedback
ANSWER_META_FIELDS = ("source", "intent", "tone", "verbosity", "confidence", "chunks", "trace", "query_id")


# ---------------------------------------------------------------------------
# Shared answering logic
# ---------------------------------------------------------------------------

class RequestError(Exception):
    def __init__(self, detail: str, status: int = 400):
        super().__init__(detail)
        self.detail, self.status = detail, status


def _resolve_document(user, document_id: Any) -> Optional[UploadedDocument]:
    """The selected document must exist and belong to the caller."""
    if not document_id:
        return None
    try:
        doc_uuid = uuid.UUID(str(document_id))
    except ValueError:
        raise RequestError("invalid document_id")
    doc = UploadedDocument.objects.filter(id=doc_uuid, user=user).first()
    if doc is None:
        raise RequestError("document not found", status=404)
    return doc


def _recent_history(conv: Conversation) -> List[Dict[str, str]]:
    # "role" breaks timestamp ties so a question always precedes its answer
    msgs = list(conv.messages.order_by("-timestamp", "role")[:HISTORY_MESSAGES])
    return [{"role": m.role, "content": m.content} for m in reversed(msgs)]


def answer_query(user, query: str, *, domain: Optional[str], document: Optional[UploadedDocument],
                 mood: Optional[str], history: List[Dict[str, str]], force_agent: bool = False,
                 conversation_id: Optional[str] = None, chat_mode_label: str = "chat") -> Dict[str, Any]:
    # One LLM call decides intent, tone, length, and rewrites follow-ups
    # ("and its side effects?") into a standalone question.
    route = classify_intent_and_route(query, has_document=document is not None, history=history)
    intent = route["intent"]
    standalone = route["standalone"] or query
    tone = resolve_tone(mood, route["tone"])
    verbosity = route["verbosity"]

    trace: List[Dict[str, Any]] = [{
        "step": "route", "intent": intent, "confidence": route["confidence"], "tone": tone,
        "verbosity": verbosity, "decided_by": route["note"],
    }]
    if standalone != query:
        trace.append({"step": "condense", "result": standalone})

    if intent == "chat" and not force_agent:
        answer = generate_answer(query, "chat", history=history, tone=tone, verbosity="brief")
        result = {"answer": answer, "mode": chat_mode_label, "source": "chat",
                  "confidence": route["confidence"], "chunks": [], "trace": trace}
    elif intent == "memory_store":
        status = memory.add_memory_fact(user, route["fact"])
        answer = {
            "stored": "Got it — I'll remember that.",
            "duplicate": "I already have that noted.",
            "empty": "What would you like me to remember?",
        }.get(status, "Sorry, I couldn't save that right now. Please try again.")
        trace.append({"step": "memory_store", "status": status})
        result = {"answer": answer, "mode": "agent", "source": "memory", "confidence": 1.0,
                  "chunks": [], "trace": trace}
    else:
        agent = build_agent(
            user,
            domain=domain,
            document_id=document.id if document is not None else None,
            question_intent=intent,
            history=history,
            tone=tone,
            verbosity=verbosity,
        )
        res = agent.run(standalone)
        result = {
            "answer": res["answer"],
            "mode": "agent",
            "source": res["answer_source"],
            "confidence": res["confidence"],
            "chunks": res["chunks"],
            "trace": trace + res["trace"],
        }

    memory.store_conversation_turn(
        user, query, result["answer"], conversation_id=conversation_id,
        document_id=str(document.id) if document is not None and result["source"] == "document" else None,
    )
    qlog = QueryLog.objects.create(user=user, query=query, answer=result["answer"],
                                   top_score=result["confidence"], chunks=result["chunks"])
    result.update({"intent": intent, "tone": tone, "verbosity": verbosity,
                   "query_id": str(qlog.id), "query_log_id": str(qlog.id)})
    return result


class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_scope = "ask"

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            document = _resolve_document(request.user, request.data.get("document_id"))
        except RequestError as e:
            return Response({"detail": e.detail}, status=e.status)

        result = answer_query(
            request.user,
            serializer.validated_data["query"].strip(),
            domain=request.data.get("domain") or request.session.get("active_domain"),
            document=document,
            mood=request.data.get("mood"),
            history=[],
            chat_mode_label="agent",
        )
        return Response(result)


# ---------------------------------------------------------------------------
# Feedback & analytics
# ---------------------------------------------------------------------------

class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        fb = serializer.save()
        for ch in (fb.query_log.chunks or []):
            if isinstance(ch, dict) and ch.get("id") and (ch.get("meta") or {}).get("type") != "web":
                ChunkFeedback.apply_feedback(ch["id"], fb.value)
        return Response({"detail": "feedback saved"}, status=201)


class AnalyticsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        logs = QueryLog.objects.filter(user=user)
        return Response({
            "total_queries": logs.count(),
            "queries_today": logs.filter(created_at__date=timezone.now().date()).count(),
            "avg_score": float(logs.aggregate(avg=Avg("top_score"))["avg"] or 0.0),
            "positive_feedback": Feedback.objects.filter(user=user, value="up").count(),
            "negative_feedback": Feedback.objects.filter(user=user, value="down").count(),
            "top_queries": list(logs.values("query").annotate(count=Count("id")).order_by("-count")[:10]),
        })


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class UploadDocumentAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_scope = "upload"

    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response({"detail": "No file"}, status=400)
        ext = Path(file.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return Response({"detail": f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
                            status=400)
        if file.size > MAX_UPLOAD_MB * 1024 * 1024:
            return Response({"detail": f"File too large (max {MAX_UPLOAD_MB} MB)"}, status=413)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for c in file.chunks():
                tmp.write(c)
            path = tmp.name
        try:
            result = index_document(path, request.user.id, file.name)
        except IndexingError as e:
            return Response({"detail": str(e)}, status=422)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return Response(result, status=201)


class DocumentListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        data: Dict[str, list] = {}
        for d in UploadedDocument.objects.filter(user=request.user).order_by("domain", "-created_at"):
            dom = (d.domain or "general").lower()
            data.setdefault(dom, []).append({
                "id": str(d.id),
                "title": d.title or d.original_filename,
                "filename": d.original_filename,
                "domain": dom,
                "created_at": d.created_at,
            })
        return Response(data)


class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        doc = UploadedDocument.objects.filter(id=doc_id, user=request.user).first()
        if doc is None:
            return Response({"detail": "Not found"}, status=404)

        ids = list(DocumentChunk.objects.filter(document=doc).values_list("pinecone_id", flat=True))
        try:
            if ids:
                delete_chunks(ids)
            else:
                delete_document_vectors(request.user.id, str(doc.id))
        except Exception as e:
            # Keep the DB row so the user can retry; deleting it would orphan
            # vectors that would keep showing up in answers.
            logger.error("Pinecone deletion failed for document %s: %s", doc.id, e)
            return Response({"detail": "Could not remove the document from the vector index; please retry."},
                            status=502)

        memory.clear_memory_for_document(request.user.id, str(doc.id))
        doc.delete()  # cascades to DocumentChunk
        return Response({"detail": "Deleted"}, status=200)


class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        domain = request.data.get("domain")
        if not domain:
            return Response({"detail": "domain required"}, status=400)
        request.session["active_domain"] = domain
        request.session.save()
        return Response({"detail": f"Domain -> {domain}"})


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class CreateConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        conv = Conversation.objects.create(user=request.user)
        return Response({"id": str(conv.id)}, status=201)


class ConversationListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        chats = Conversation.objects.filter(user=request.user).order_by("-updated_at")
        return Response([{"id": str(c.id), "title": c.title, "updated_at": c.updated_at} for c in chats])


class ConversationChatAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_scope = "ask"

    def post(self, request, cid):
        conv = Conversation.objects.filter(id=cid, user=request.user).first()
        if conv is None:
            return Response({"detail": "conversation missing"}, status=404)

        query = (request.data.get("query") or "").strip()
        if not query:
            return Response({"detail": "empty prompt"}, status=400)
        try:
            document = _resolve_document(request.user, request.data.get("document_id"))
        except RequestError as e:
            return Response({"detail": e.detail}, status=e.status)

        result = answer_query(
            request.user,
            query,
            domain=request.data.get("domain") or request.session.get("active_domain"),
            document=document,
            mood=request.data.get("mood"),
            history=_recent_history(conv),
            force_agent=bool(request.data.get("agent_mode")),
            conversation_id=str(conv.id),
        )

        Message.objects.create(conversation=conv, role="user", content=query)
        Message.objects.create(conversation=conv, role="assistant", content=result["answer"],
                               meta={k: result.get(k) for k in ANSWER_META_FIELDS})
        if conv.title == "New Chat":
            conv.title = query[:60] + ("…" if len(query) > 60 else "")
        conv.save()  # bumps updated_at so the chat list stays in recency order
        return Response(result)


class ConversationHistoryAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, cid):
        conv = Conversation.objects.filter(id=cid, user=request.user).first()
        if conv is None:
            return Response({"detail": "not found"}, status=404)
        messages = list(conv.messages.order_by("timestamp", "-role"))

        # Attach the user's latest rating to each answer so the UI shows it
        query_ids = [m.meta["query_id"] for m in messages if m.meta and m.meta.get("query_id")]
        ratings = {
            str(qid): value
            for qid, value in Feedback.objects.filter(query_log_id__in=query_ids, user=request.user)
            .order_by("created_at")
            .values_list("query_log_id", "value")
        }

        out = []
        for m in messages:
            item: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "assistant" and m.meta:
                item["meta"] = {**m.meta, "feedback": ratings.get(str(m.meta.get("query_id")))}
            out.append(item)
        return Response(out)


class RenameConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, cid):
        conv = Conversation.objects.filter(id=cid, user=request.user).first()
        if conv is None:
            return Response({"detail": "not found"}, status=404)
        name = (request.data.get("title") or "").strip()
        if not name:
            return Response({"detail": "required"}, status=400)
        conv.title = name[:200]
        conv.save()
        return Response({"detail": "renamed", "title": conv.title})


class DeleteConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, cid):
        conv = Conversation.objects.filter(id=cid, user=request.user).first()
        if conv is None:
            return Response({"detail": "not found"}, status=404)
        memory.clear_memory_for_conversation(request.user.id, str(conv.id))
        conv.delete()
        return Response({"detail": "deleted"}, status=200)
