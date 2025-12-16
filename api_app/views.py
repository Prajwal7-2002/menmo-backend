# api_app/views.py
"""
Final patched API views.

Key changes:
- When creating agents or calling agent.run, always pass domain & document_id.
- Document deletion clears Pinecone vectors and calls memory.clear_memory_for_document.
- Conversation endpoints pass domain/document_id into agent build.
- Defensive: all external calls wrapped with try/except and warnings.
"""

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
    relevant_chunks,
    clear_memory_for_document,
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

        route = classify_intent_and_route(query)
        intent = route.get("intent")

        # If intent is pure chat/small-talk, answer directly without RAG or web.
        if intent == "chat":
            try:
                ans = call_llm_answer(question=query, context="", mood=mood, max_tokens=128) or "Hi, how can I help you today?"
            except Exception:
                ans = "Hi, how can I help you today?"
            store_conversation_turn(user, query, ans)
            QueryLog.objects.create(user=user, query=query, top_score=1.0, chunks=[])
            return Response({
                "answer": ans,
                "mode": "agent",
                "source": "chat",
                "confidence": 1.0,
                "chunks": [],
                "trace": [{"step": "intent", "intent": "chat"}],
            })

        # Strict document questions (doc_summary/doc_lookup) answer ONLY from the selected document
        # and never broaden to web. Non-doc intents (knowledge/agent/memory) can use web.
        if intent in ("doc", "doc_summary", "doc_lookup") and document_id:
            agent = build_agent(
                user_id=user.id,
                domain=domain,
                document_id=document_id,
                allow_web=False,
                agent_enabled=True,
                question_intent=intent,
            )
            agent_res = agent.run(query)
            final_ans = safe_text(agent_res.get("answer", ""))
            source = agent_res.get("answer_source", "document")
            store_conversation_turn(user, query, final_ans)
            QueryLog.objects.create(user=user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
            return Response({
                "answer": final_ans,
                "mode": "agent",
                "source": source,
                "confidence": float(agent_res.get("confidence", 0.0)),
                "chunks": agent_res.get("chunks", []),
                "trace": agent_res.get("trace", []),
            })

        # All other intents (knowledge/agent/memory) use open agent with web allowed.
        agent = build_agent(user_id=user.id, domain=domain, document_id=None, allow_web=True, agent_enabled=True)
        agent_res = agent.run(query)
        final_ans = safe_text(agent_res.get("answer", ""))
        source = agent_res.get("answer_source", "agent")
        store_conversation_turn(user, query, final_ans)
        QueryLog.objects.create(user=user, query=query, top_score=agent_res.get("confidence", 0.0), chunks=agent_res.get("chunks", []))
        return Response({
            "answer": final_ans,
            "mode": "agent",
            "source": source,
            "confidence": float(agent_res.get("confidence", 0.0)),
            "chunks": agent_res.get("chunks", []),
            "trace": agent_res.get("trace", []),
        })
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
        except UploadedDocument.DoesNotExist:
            return Response({"detail": "Not found"}, status=404)

        # ---- 1. Collect chunk IDs ----
        chunk_ids = list(
            DocumentChunk.objects.filter(document_id=doc.id)
            .values_list("pinecone_id", flat=True)
        )

        # ---- 2. Delete vectors from Pinecone in safe batches ----
        if chunk_ids:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

                BATCH = 100
                for i in range(0, len(chunk_ids), BATCH):
                    batch = chunk_ids[i:i+BATCH]
                    index.delete(ids=batch)

            except Exception as e:
                # Still proceed but warn user
                print(f"⚠ Pinecone deletion error: {e}")

        # ---- 3. Delete DB chunks ----
        DocumentChunk.objects.filter(document_id=doc.id).delete()

        # ---- 3.5 Clear any vector-memory related to this document
        try:
            clear_memory_for_document(user_id=request.user.id, document_id=str(doc.id))
        except Exception as e:
            print(f"⚠ clear_memory_for_document failed: {e}")

        # ---- 4. Delete the document ----
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

        route = classify_intent_and_route(query)
        if route.get("intent") == "chat" and not agent_mode:
            answer = call_llm_answer(question=query, context="Casual conversation.", mood=mood, max_tokens=80)
            Message.objects.create(conversation=conv, role="user", content=query)
            Message.objects.create(conversation=conv, role="assistant", content=answer)
            store_conversation_turn(request.user, query, answer)
            QueryLog.objects.create(user=request.user, query=query, top_score=0, chunks=[])
            return Response({"answer": answer, "mode": "chat", "confidence": float(route.get("confidence", 0.9)), "chunks": []})

        # Agentic RAG path
        agent = build_agent(
            user_id=request.user.id,
            domain=domain,
            document_id=doc_id,
            allow_web=True,
            agent_enabled=True,
        )
        agent_res = agent.run(query)
        final_ans = safe_text(agent_res.get("answer", ""))
        source = agent_res.get("answer_source", "agent")

        Message.objects.create(conversation=conv, role="user", content=query)
        Message.objects.create(conversation=conv, role="assistant", content=final_ans)
        store_conversation_turn(request.user, query, final_ans)
        QueryLog.objects.create(
            user=request.user,
            query=query,
            top_score=agent_res.get("confidence", 0.0),
            chunks=agent_res.get("chunks", []),
        )

        return Response({
            "answer": final_ans,
            "mode": "agent",
            "source": source,
            "confidence": float(agent_res.get("confidence", 0.0)),
            "chunks": agent_res.get("chunks", []),
            "trace": agent_res.get("trace", []),
        })


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
