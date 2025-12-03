# api_app/views.py (FINAL patched — Agent-driven RAG orchestration)

import os
import json
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict

from django.db.models import Avg, Count
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback

from rag.models import UploadedDocument, ChunkFeedback, DocumentChunk
from rag.loader import index_document
from rag.agent import build_agent

from rag.memory import (
    store_conversation_turn,
    load_vector_memory,
    add_memory_fact,
    relevant_chunks
)

from api_app.models import Conversation, Message
from rag.tools.intent_classifier import classify_intent_and_route
from rag.llm import call_llm_answer


# ------------------------
# Small helpers
# ------------------------
def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)

def normalize_agent_output(agent_output: Any) -> Dict[str, Any]:
    if agent_output is None:
        return {"answer": "", "mode": "agent", "confidence": 0.0, "chunks": [], "trace": [], "steps": 0}
    if isinstance(agent_output, str):
        return {"answer": agent_output, "mode": "agent", "confidence": 1.0, "chunks": [], "trace": [], "steps": 1}
    if isinstance(agent_output, dict):
        return {
            "answer": safe_text(agent_output.get("answer") or agent_output.get("output") or agent_output.get("text") or ""),
            "mode": agent_output.get("mode", "agent"),
            "confidence": float(agent_output.get("confidence", 1.0)),
            "chunks": agent_output.get("chunks", []) or [],
            "trace": agent_output.get("trace", []) or [],
            "steps": int(agent_output.get("steps", len(agent_output.get("trace", []) or [])))
        }
    return {"answer": str(agent_output), "mode": "agent", "confidence": 1.0, "chunks": [], "trace": [], "steps": 1}


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
        intent = route.get("intent")
        intent_conf = route.get("confidence", 1.0)

        # force doc intent if explicitly passing document_id
        if document_id:
            intent = "doc"

        # MEMORY STORE
        if intent == "memory_store":
            content = query
            low = content.strip().lower()
            for pref in ("remember that", "remember", "note that", "please remember", "store"):
                if low.startswith(pref):
                    content = content[len(pref):].strip()
                    break
            if not content:
                content = query
            stored = False
            try:
                stored = add_memory_fact(user=user, text=content)
            except Exception:
                try:
                    store_conversation_turn(user, f"[MEMORY] {content}", "")
                    stored = True
                except Exception:
                    stored = False

            QueryLog.objects.create(user=user, query=query, top_score=0.0, chunks=[])
            if stored:
                return Response({"answer": "Okay — I will remember that.", "mode": "memory_store", "confidence": intent_conf})
            return Response({"answer": "I tried to save that memory but ran into an issue.", "mode": "memory_store", "confidence": 0.0}, status=500)

        # MEMORY RECALL via agent
        if intent == "memory_recall":
            agent = build_agent(user_id=user.id, domain=domain, document_id=document_id, mood=mood, history=memory, agent_enabled=True)
            agent_raw = agent.run(query)
            agent_res = normalize_agent_output(agent_raw)
            final_text = safe_text(agent_res["answer"])
            store_conversation_turn(user, query, final_text)
            return Response({"answer": final_text, "mode": "agent_memory", "confidence": intent_conf, "chunks": agent_res.get("chunks", []), "trace": agent_res.get("trace", [])})

        # CHAT-only intent -> direct LLM
        if intent == "chat":
            answer = call_llm_answer(question=query, context="Casual conversation.", mood=mood, max_tokens=60)
            store_conversation_turn(user, query, answer)
            QueryLog.objects.create(user=user, query=query, top_score=0.0, chunks=[])
            return Response({"answer": answer, "mode": "chat", "confidence": intent_conf, "chunks": []})

        # DOCUMENT question -> AGENT-driven RAG
        if intent == "doc":
            # Let the agent orchestrate retrieval/rewrite/retry
            agent = build_agent(user_id=user.id, domain=domain, document_id=document_id, mood=mood, history=memory, agent_enabled=True)
            agent_res = agent.run(query)
            final_ans = safe_text(agent_res.get("answer", ""))
            store_conversation_turn(user, query, final_ans)
            QueryLog.objects.create(user=user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
            return Response({"answer": final_ans, "mode": "agent", "confidence": float(agent_res.get("confidence", 0.0)), "chunks": agent_res.get("chunks", []), "trace": agent_res.get("trace", [])})

        # Default: let the Agent orchestrate RAG + retries
        agent = build_agent(user_id=user.id, domain=domain, document_id=document_id, mood=mood, history=memory, agent_enabled=True)
        agent_res = agent.run(query)
        final_ans = safe_text(agent_res.get("answer", ""))
        store_conversation_turn(user, query, final_ans)
        QueryLog.objects.create(user=user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
        return Response({"answer": final_ans, "mode": "agent", "confidence": float(agent_res.get("confidence", 0.0)), "chunks": agent_res.get("chunks", []), "trace": agent_res.get("trace", [])})


# -------------------------------------------------------
# FEEDBACK SYSTEM
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
# 📊 ANALYTICS
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
            "top_queries": list(logs.values("query").annotate(count=Count("id")).order_by("-count")[:10]),
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
            for c in file.chunks():
                tmp.write(c)
            path = tmp.name

        return Response(index_document(path, request.user.id, file.name), status=201)


# -------------------------------------------------------
# LIST DOCUMENTS BY DOMAIN
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


# -------------------------------------------------------
# DELETE DOCUMENT + Remove Vectors
# -------------------------------------------------------
class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
        except:
            return Response({"detail": "Not found"}, status=404)

        ids = list(DocumentChunk.objects.filter(document=doc).values_list("pinecone_id", flat=True))
        if ids:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            pc.Index(os.getenv("PINECONE_INDEX_NAME")).delete(ids=ids)

        doc.delete()
        return Response({"detail": "Deleted"}, status=200)


# -------------------------------------------------------
# DOMAIN SWITCH
# -------------------------------------------------------
class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        domain = request.data.get("domain")
        if not domain:
            return Response({"detail": "domain required"}, status=400)
        request.session["active_domain"] = domain
        request.session.save()
        return Response({"detail": f"Domain -> {domain}"})


# -------------------------------------------------------
# CREATE NEW CHAT SESSION
# -------------------------------------------------------
class CreateConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        conv = Conversation.objects.create(user=request.user)
        return Response({"id": str(conv.id)}, status=201)


# -------------------------------------------------------
# LIST ALL CONVERSATIONS (REQUIRED for URL import)
# -------------------------------------------------------
class ConversationListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        chats = Conversation.objects.filter(user=request.user).order_by("-updated_at")
        return Response([{
            "id": str(c.id),
            "title": c.title,
            "updated_at": c.updated_at
        } for c in chats])


# -------------------------------------------------------
# 🔥 MAIN CHAT VIEW — AGENTIC RAG (Agent orchestrates)
# -------------------------------------------------------
class ConversationChatAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail": "conversation missing"}, status=404)

        query = (request.data.get("query") or "").strip()
        if not query:
            return Response({"detail": "empty prompt"}, status=400)

        agent_mode = bool(request.data.get("agent_mode"))
        domain = request.session.get("active_domain")
        mood = request.data.get("mood", "neutral")
        doc_id = request.data.get("document_id")
        memory = load_vector_memory(request.user, query)

        # Check chat intent before agent/rag
        route = classify_intent_and_route(query)
        if route.get("intent") == "chat" and not agent_mode:
            # lightweight chat bypass: use simple LLM for small talk
            answer = call_llm_answer(question=query, context="Casual conversation.", mood=mood, max_tokens=80)
            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            store_conversation_turn(request.user, query, answer)
            QueryLog.objects.create(user=request.user, query=query, top_score=0, chunks=[])
            return Response({"answer": answer, "mode": "chat", "confidence": float(route.get("confidence", 0.9)), "chunks": []})

        # AGENT MODE (explicit toggle) — respect agent_mode if provided
        if agent_mode:
            agent = build_agent(user_id=request.user.id, domain=domain, document_id=doc_id, mood=mood, history=memory, agent_enabled=True)
            agent_raw = agent.run(query)
            agent_res = normalize_agent_output(agent_raw)
            final_text = safe_text(agent_res["answer"])

            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=final_text)
            store_conversation_turn(request.user, query, final_text)
            QueryLog.objects.create(user=request.user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
            return Response(agent_res)

        # Default: Agent orchestrates for document/knowledge queries (agent decides retrieval/web/rewrite)
        agent = build_agent(user_id=request.user.id, domain=domain, document_id=doc_id, mood=mood, history=memory, agent_enabled=True)
        agent_raw = agent.run(query)
        agent_res = normalize_agent_output(agent_raw)
        final_text = safe_text(agent_res["answer"])

        Message.objects.create(conversation=conv, role="user", content=query)
        Message.objects.create(conversation=conv, role="assistant", content=final_text)
        store_conversation_turn(request.user, query, final_text)
        QueryLog.objects.create(user=request.user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
        return Response(agent_res)


# ========= Chat History / Rename / Delete ========= #
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
        name = request.data.get("title", "").strip()
        if not name:
            return Response({"detail": "required"}, status=400)
        conv.title = name
        conv.save()
        return Response({"detail": "renamed", "title": name})


class DeleteConversationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail": "not found"}, status=404)
        conv.delete()
        return Response({"detail": "deleted"}, status=200)
