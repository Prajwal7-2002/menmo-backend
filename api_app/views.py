# views.py — Fully Patched & Crash-Proof Version

import os
import json
import uuid
import tempfile
from pathlib import Path

from django.db.models import Avg, Count
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback

from rag.models import UploadedDocument, ChunkFeedback, DocumentChunk
from rag.pipeline import run_rag
from rag.loader import index_document
from rag.agent import build_agent

from rag.memory import (
    store_conversation_turn,
    load_vector_memory,
    add_memory_fact
)

from api_app.models import Conversation, Message
from rag.tools.intent_classifier import classify_intent_and_route
from rag.llm import call_llm_answer
from rag.controller import agentic_answer


# ================================================================
# SAFETY WRAPPER — PREVENT FRONTEND CRASH BY ENSURING STRING OUTPUT
# ================================================================
def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# =====================================================================
# ASK API — RAG + AGENT + CHAT + MEMORY STORE + MEMORY RECALL
# =====================================================================

class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = request.user
        query = serializer.validated_data["query"].strip()

        mood = request.data.get("mood", "neutral")
        domain = request.session.get("active_domain")
        document_id = request.data.get("document_id")
        memory = load_vector_memory(user, query)

        route = classify_intent_and_route(query)
        intent = route["intent"]
        intent_conf = route["confidence"]

        force_doc_flag = bool(request.data.get("force_doc"))
        document_mention = any(
            key in query.lower() for key in
            ("in the document", "in this document", "in the pdf", "page", "section", "chapter")
        )

        if document_id and (force_doc_flag or document_mention):
            intent = "doc"

        # --------------------------- MEMORY STORE ---------------------------
        if intent == "memory_store":
            content = query
            low = content.lower()

            for pref in ("remember that", "remember", "note that", "please remember", "store"):
                if low.startswith(pref):
                    content = content[len(pref):].strip()
                    break

            content = content or query

            stored = add_memory_fact(user=user, text=content)

            QueryLog.objects.create(user=user, query=query, top_score=0.0, chunks=[])

            if stored:
                return Response({"answer": "Okay — I will remember that.", "mode": "memory_store", "confidence": intent_conf})
            else:
                return Response({"answer": "I tried to save that memory but ran into an issue.", "mode": "memory_store"}, status=500)

        # --------------------------- MEMORY RECALL ---------------------------
        if intent == "memory_recall":
            agent_res = agentic_answer(query, user.id, domain, document_id, user=user)
            final_text = safe_text(agent_res.get("answer"))

            store_conversation_turn(user, query, final_text)

            return Response({
                "answer": final_text,
                "mode": "agent_memory",
                "confidence": intent_conf,
                "chunks": [],
                "trace": agent_res.get("trace", [])
            })

        # ------------------------------- CHAT MODE ---------------------------
        if intent == "chat":
            answer = call_llm_answer(question=query, context="Casual conversation.", mood=mood, max_tokens=60)
            answer = safe_text(answer)

            store_conversation_turn(user, query, answer)
            return Response({"answer": answer, "mode": "chat", "confidence": intent_conf, "chunks": []})

        # ----------------------------- DOCUMENT MODE ------------------------
        if intent == "doc":
            rag_res = run_rag(query, user.id, domain, document_id, mood, memory)
            if rag_res.get("validated"):
                store_conversation_turn(user, query, safe_text(rag_res["answer"]))
                return Response({**rag_res, "mode": "rag"})

            agent_res = agentic_answer(query, user.id, domain, document_id, user=user)
            final_text = safe_text(agent_res.get("answer"))

            store_conversation_turn(user, query, final_text)

            return Response({
                "answer": final_text,
                "mode": "agent_fallback",
                "confidence": float(rag_res.get("confidence", 0)),
                "chunks": rag_res.get("chunks", []),
                "trace": agent_res.get("trace", [])
            })

        # --------------------------- KNOWLEDGE MODE -------------------------
        if intent == "knowledge":
            agent_res = agentic_answer(query, user.id, domain, document_id, user=user)
            final_text = safe_text(agent_res.get("answer"))

            store_conversation_turn(user, query, final_text)

            return Response({
                "answer": final_text,
                "mode": "agent",
                "confidence": intent_conf,
                "chunks": [],
                "trace": agent_res.get("trace", [])
            })

        # ---------------------------- DEFAULT MODE --------------------------
        rag_res = run_rag(query, user.id, domain, document_id, mood, memory)
        if rag_res.get("validated"):
            store_conversation_turn(user, query, safe_text(rag_res["answer"]))
            return Response({**rag_res, "mode": "rag"})

        agent_res = agentic_answer(query, user.id, domain, document_id, user=user)
        final_text = safe_text(agent_res.get("answer"))

        store_conversation_turn(user, query, final_text)

        return Response({
            "answer": final_text,
            "mode": "agent_fallback",
            "confidence": float(rag_res.get("confidence", 0)),
            "chunks": rag_res.get("chunks", []),
            "trace": agent_res.get("trace", [])
        })


# =====================================================================
# FEEDBACK
# =====================================================================

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


# =====================================================================
# ANALYTICS
# =====================================================================

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


# =====================================================================
# DOCUMENT UPLOAD
# =====================================================================

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
            tmp_path = tmp.name

        result = index_document(tmp_path, request.user.id, file.name)
        return Response(result, status=201)


# =====================================================================
# DOCUMENT LIST
# =====================================================================

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


# =====================================================================
# DOCUMENT DELETE
# =====================================================================

class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
        except UploadedDocument.DoesNotExist:
            return Response({"detail": "Not found"}, status=404)

        chunk_ids = list(DocumentChunk.objects.filter(document=doc).values_list("pinecone_id", flat=True))

        if chunk_ids:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            index.delete(ids=chunk_ids)

        doc.delete()

        return Response({"detail": "Deleted"}, status=200)


# =====================================================================
# DOMAIN SWITCH
# =====================================================================

class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        domain = request.data.get("domain")
        if not domain:
            return Response({"detail": "domain required"}, status=400)

        request.session["active_domain"] = domain
        request.session.save()

        return Response({"detail": f"Domain -> {domain}"})


# =====================================================================
# CONVERSATION SYSTEM
# =====================================================================

class CreateConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        conv = Conversation.objects.create(user=request.user)
        return Response({"id": str(conv.id)}, status=201)


class ConversationListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        chats = Conversation.objects.filter(user=request.user).order_by("-updated_at")
        return Response([
            {"id": str(c.id), "title": c.title, "updated_at": c.updated_at}
            for c in chats
        ])


class ConversationChatAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "conversation missing"}, status=404)

        query = safe_text(request.data.get("query") or "").strip()
        if not query:
            return Response({"detail": "empty prompt"}, status=400)

        agent_mode = bool(request.data.get("agent_mode"))
        domain = request.session.get("active_domain")
        mood = request.data.get("mood", "neutral")
        doc_id = request.data.get("document_id")
        memory = load_vector_memory(request.user, query)

        # ----------------------------- SMART CHAT DETECTION -----------------------------
        route = classify_intent_and_route(query)
        if route.get("intent") == "chat":
            answer = call_llm_answer(question=query, context="Casual conversation.", mood=mood, max_tokens=80)
            answer = safe_text(answer)

            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            store_conversation_turn(request.user, query, answer)

            QueryLog.objects.create(user=request.user, query=query, top_score=0, chunks=[])

            return Response({"answer": answer, "mode": "chat", "confidence": float(route.get("confidence", 0.9)), "chunks": []})

        # ------------------------------------- AGENT MODE -------------------------------------
        if agent_mode:
            agent = build_agent(
                user_id=request.user.id,
                domain=domain,
                document_id=doc_id,
                mood=mood,
                history=memory,
                agent_enabled=True
            )

            agent_output = agent.run(query)
            final_text = safe_text(agent_output.get("answer"))

            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=final_text)

            store_conversation_turn(request.user, query, final_text)

            QueryLog.objects.create(user=request.user, query=query, top_score=0, chunks=[])

            return Response(agent_output)

        # --------------------------------------- RAG MODE --------------------------------------
        rag_res = run_rag(query, request.user.id, domain, doc_id, mood, memory)
        final_answer = safe_text(rag_res.get("answer"))

        Message.objects.create(conversation=conv, role="user", content=query)
        Message.objects.create(conversation=conv, role="assistant", content=final_answer)

        store_conversation_turn(request.user, query, final_answer)

        QueryLog.objects.create(
            user=request.user,
            query=query,
            top_score=rag_res.get("top_score", 0),
            chunks=rag_res.get("chunks", [])
        )

        return Response(rag_res)


class ConversationHistoryAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "not found"}, status=404)

        return Response([
            {"role": m.role, "content": safe_text(m.content)}
            for m in conv.messages.order_by("timestamp")
        ])


class RenameConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "not found"}, status=404)

        new_name = safe_text(request.data.get("title", "")).strip()
        if not new_name:
            return Response({"detail": "required"}, status=400)

        conv.title = new_name
        conv.save()

        return Response({"detail": "renamed", "title": new_name})


class DeleteConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "not found"}, status=404)

        conv.delete()

        return Response({"detail": "deleted"}, status=200)
