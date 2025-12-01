# api_app/views.py
import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Avg, Count
from django.utils import timezone

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback
from rag.models import UploadedDocument, ChunkFeedback, DocumentChunk
from rag.loader import index_document
from rag.pipeline import run_rag
from rag.controller import agentic_answer
from rag.memory import store_conversation_turn, load_vector_memory

from api_app.models import Conversation, Message
from rag.llm import call_llm_answer
from rag.tools.intent_classifier import classify_intent_and_route


logger = logging.getLogger(__name__)


# -------------------------------------------------------
#  ASK — Document-first → Agentic fallback (no hard rules)
# -------------------------------------------------------
class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = request.user
        query = serializer.validated_data["query"].strip()
        if not query:
            return Response({"detail": "empty query"}, status=400)

        mood = request.data.get("mood", "neutral")
        domain = request.session.get("active_domain")
        document_id = request.data.get("document_id")
        memory = load_vector_memory(user, query)

        # 🔥 NEW: INTENT CLASSIFICATION FIRST
        route = classify_intent_and_route(query)
        intent = route["intent"]
        route_conf = route["confidence"]

        # -------------------------------------------------------
        # 1. CHAT / GREETING → never call RAG
        # -------------------------------------------------------
        if intent == "handled_by_chat":
            answer = call_llm_answer(
                question=query,
                context="User is chatting casually.",
                mood="friendly",
                max_tokens=40
            )
            return Response({
                "answer": answer,
                "mode": "chat",
                "confidence": route_conf,
                "chunks": []
            })

        # -------------------------------------------------------
        # 2. WEB SEARCH → run agent (has web search tool)
        # -------------------------------------------------------
        if intent == "handled_by_web":
            agent_res = agentic_answer(
                query=query,
                user_id=user.id,
                domain=None,
                document_id=None,
                user=user,
                max_steps=6
            )
            return Response({
                "answer": agent_res["answer"],
                "mode": "agent",
                "confidence": route_conf,
                "chunks": []
            })

        # -------------------------------------------------------
        # 3. DOCUMENT RAG → FIRST TRY RAG
        # -------------------------------------------------------
        if intent == "handled_by_rag":
            rag_res = run_rag(query, user.id, domain, document_id, mood, memory)
            if rag_res.get("validated", False):
                return Response({
                    "answer": rag_res["answer"],
                    "mode": "rag",
                    "confidence": rag_res["confidence"],
                    "chunks": rag_res["chunks"]
                })
            # fallback to agent
            agent_res = agentic_answer(
                query=query,
                user_id=user.id,
                domain=domain,
                document_id=document_id,
                user=user,
                max_steps=6
            )
            return Response({
                "answer": agent_res["answer"],
                "mode": "agent",
                "confidence": rag_res["confidence"],
                "chunks": rag_res.get("chunks", [])
            })

        # -------------------------------------------------------
        # 4. COMPLEX / UNKNOWN → Agent
        # -------------------------------------------------------
        if intent == "handled_by_agent":
            agent_res = agentic_answer(
                query=query,
                user_id=user.id,
                domain=domain,
                document_id=document_id,
                user=user,
                max_steps=6
            )
            return Response({
                "answer": agent_res["answer"],
                "mode": "agent",
                "confidence": route_conf,
                "chunks": []
            })

        # -------------------------------------------------------
        # 5. FALLBACK → simple LLM
        # -------------------------------------------------------
        answer = call_llm_answer(question=query, context="", mood="neutral")
        return Response({
            "answer": answer,
            "mode": "fallback",
            "confidence": route_conf,
            "chunks": []
        })

# -------------------------------------------------------
# FEEDBACK
# -------------------------------------------------------
class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)

        fb = serializer.save()
        qlog = fb.query_log

        for ch in (qlog.chunks or []):
            if ch.get("id"):
                ChunkFeedback.apply_feedback(ch["id"], fb.value)

        return Response({"detail": "feedback saved"}, status=201)


# -------------------------------------------------------
# ANALYTICS
# -------------------------------------------------------
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
            "top_queries": list(logs.values("query")
                               .annotate(count=Count("id"))
                               .order_by("-count")[:10]),
        })


# -------------------------------------------------------
# UPLOAD DOCUMENT
# -------------------------------------------------------
class UploadDocumentAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response({"detail": "No file"}, status=400)

        ext = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in file.chunks():
                tmp.write(chunk)
            file_path = tmp.name

        try:
            res = index_document(file_path, request.user.id, file.name)
            return Response(res, status=201)
        except Exception as e:
            logger.exception("index_document failed")
            return Response({"detail": "index failed", "error": str(e)}, status=500)


# -------------------------------------------------------
# DOCUMENT LIST / DELETE / DOMAIN
# -------------------------------------------------------
class DocumentListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        docs = UploadedDocument.objects.filter(user=request.user).order_by("domain", "-created_at")
        data = {}
        for d in docs:
            dom = (d.domain or "general").lower()
            data.setdefault(dom, []).append({
                "id": str(d.id),
                "title": d.title,
                "filename": d.original_filename,
                "domain": dom,
                "created_at": d.created_at
            })
        return Response(data)


class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
        except:
            return Response({"detail": "Not found"}, status=404)

        ids = list(DocumentChunk.objects.filter(document=doc)
                   .values_list("pinecone_id", flat=True))

        if ids:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                pc.Index(os.getenv("PINECONE_INDEX_NAME")).delete(ids=ids)
            except Exception:
                logger.exception("vector delete failed")

        doc.delete()
        return Response({"detail": "Deleted"}, status=200)


class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        dom = request.data.get("domain")
        if not dom:
            return Response({"detail": "domain required"}, status=400)

        request.session["active_domain"] = dom
        request.session.save()

        return Response({"detail": f"Domain -> {dom}"})


# -------------------------------------------------------
# Conversations (create/list/chat/history/rename/delete)
# -------------------------------------------------------
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


class ConversationHistoryAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail": "not found"}, status=404)
        return Response([{"role": m.role, "content": m.content} for m in conv.messages.order_by("timestamp")])


class RenameConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail": "not found"}, status=404)
        title = request.data.get("title", "").strip()
        if not title:
            return Response({"detail": "required"}, status=400)
        conv.title = title
        conv.save()
        return Response({"detail": "renamed", "title": title})


class DeleteConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail": "not found"}, status=404)
        conv.delete()
        return Response({"detail": "deleted"}, status=200)
    
# -------------------------------------------------------
# MAIN CHAT — RAG + Agent fallback
# -------------------------------------------------------
class ConversationChatAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "conversation missing"}, 404)

        query = (request.data.get("query") or "").strip()
        if not query:
            return Response({"detail": "empty prompt"}, 400)

        mood = request.data.get("mood", "neutral")
        domain = request.session.get("active_domain")
        document_id = request.data.get("document_id")
        memory = load_vector_memory(request.user, query)

        # 🔥 Intent Routing First
        route = classify_intent_and_route(query)
        intent = route["intent"]
        route_conf = route["confidence"]

        if document_id:
            intent = "handled_by_rag"

        # 🔥 If user selected a specific document → FORCE RAG mode
        if document_id:
            intent = "handled_by_rag"


        # === 1. CHAT / GREETING ===
        if intent == "handled_by_chat":
            answer = call_llm_answer(
                question=query,
                context="User is chatting casually.",
                mood="friendly",
                max_tokens=40
            )
            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            return Response({
                "answer": answer,
                "mode": "chat",
                "confidence": route_conf,
                "chunks": []
            })

        # === 2. WEB ===
        if intent == "handled_by_web":
            agent_res = agentic_answer(
                query=query,
                user_id=request.user.id,
                domain=None,
                document_id=None,
                user=request.user,
                max_steps=6
            )
            answer = agent_res["answer"]
            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            return Response({
                "answer": answer,
                "mode": "agent",
                "confidence": route_conf,
                "chunks": []
            })

        # === 3. DOCUMENT RAG ===
        if intent == "handled_by_rag":
            rag_res = run_rag(query, request.user.id, domain, document_id, mood, memory)
            if rag_res.get("validated", False):
                answer = rag_res["answer"]
                chunks = rag_res["chunks"]
                conf = rag_res["confidence"]
                mode = "rag"
            else:
                agent_res = agentic_answer(
                    query=query,
                    user_id=request.user.id,
                    domain=domain,
                    document_id=document_id,
                    user=request.user,
                    max_steps=6
                )
                answer = agent_res["answer"]
                chunks = rag_res.get("chunks", [])
                conf = rag_res.get("confidence", 0.0)
                mode = "agent"

            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            return Response({
                "answer": answer,
                "mode": mode,
                "confidence": conf,
                "chunks": chunks
            })

        # === 4. AGENT ===
        if intent == "handled_by_agent":
            agent_res = agentic_answer(
                query=query,
                user_id=request.user.id,
                domain=domain,
                document_id=document_id,
                user=request.user,
                max_steps=6
            )
            answer = agent_res["answer"]
            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            return Response({
                "answer": answer,
                "mode": "agent",
                "confidence": route_conf,
                "chunks": []
            })

        # === 5. FALLBACK ===
        answer = call_llm_answer(question=query, context="", mood="neutral")
        Message.objects.create(conversation=conv, role="user", content=query)
        Message.objects.create(conversation=conv, role="assistant", content=answer)
        return Response({
            "answer": answer,
            "mode": "fallback",
            "confidence": route_conf,
            "chunks": []
        })