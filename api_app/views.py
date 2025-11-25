# api/views.py
import os
import uuid
import tempfile
from pathlib import Path

from rag.memory import (
    load_buffer, prune_buffer, load_summary,
    save_summary, summarize_history, load_user_preferences
)
from api_app.models import UserPreference, ConversationSummary

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.db.models import Avg, Count
from django.utils import timezone

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback
from rag.models import UploadedDocument, ChunkFeedback, DocumentChunk

from rag.pipeline import run_rag
from rag.loader import index_document
from rag.agent import build_agent


# ---------------------------------------------------------
# ASK API
# ---------------------------------------------------------
class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data["query"]

        req_mood = request.data.get("mood")
        provided_history = request.data.get("history", []) or []

        prefs = load_user_preferences(request.user)
        default_mood = prefs.get("default_mood", "neutral")
        mood = req_mood or default_mood

        buffer = load_buffer(request.user)
        summary = load_summary(request.user)

        # Build final memory
        final_history = []
        if summary:
            final_history.append({"role": "system", "content": f"Summary: {summary}"})
        if buffer:
            final_history.extend(buffer)
        if provided_history:
            final_history.extend(provided_history)
        final_history = prune_buffer(final_history)

        active_domain = request.session.get("active_domain")
        document_id = request.data.get("document_id")

        result = run_rag(
            query=query,
            user_id=request.user.id,
            domain=active_domain,
            document_id=document_id,
            mood=mood,
            history=final_history,
        )

        top_score = result["chunks"][0]["score"] if result.get("chunks") else 0.0

        QueryLog.objects.create(
            id=uuid.uuid4(),
            user=request.user,
            query=query,
            answer=result.get("answer", "")[:4000],
            top_score=top_score,
            chunks=result.get("chunks", []),
        )

        return Response({"query_id": str(uuid.uuid4()), **result})


# ---------------------------------------------------------
# FEEDBACK API
# ---------------------------------------------------------
class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)

        fb = serializer.save()
        qlog = fb.query_log
        chunks = qlog.chunks or []

        for ch in chunks:
            pine_id = ch.get("id")
            if pine_id:
                ChunkFeedback.apply_feedback(pine_id, fb.value)

        return Response({"detail": "feedback saved", "id": str(fb.id)}, status=201)


# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------
class AnalyticsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        logs = QueryLog.objects.filter(user=user)

        total = logs.count()
        today = logs.filter(created_at__date=timezone.now().date()).count()
        avg_score = logs.aggregate(avg=Avg("top_score"))["avg"] or 0.0

        fb = Feedback.objects.filter(user=user)
        up = fb.filter(value="up").count()
        down = fb.filter(value="down").count()

        top_queries = (
            logs.values("query")
            .annotate(count=Count("id"))
            .order_by("-count")[:10]
        )

        return Response({
            "total_queries": total,
            "queries_today": today,
            "avg_score": float(avg_score),
            "positive_feedback": up,
            "negative_feedback": down,
            "top_queries": list(top_queries),
        })


# ---------------------------------------------------------
# UPLOAD DOCUMENT
# ---------------------------------------------------------
class UploadDocumentAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"detail": "No file uploaded"}, status=400)

        # Save file to temp path
        suffix = Path(file_obj.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in file_obj.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        print("📁 Uploaded:", file_obj.name)
        print("📁 Saved to temp:", tmp_path)

        info = index_document(
            file_path=tmp_path,
            user_id=request.user.id,
            original_name=file_obj.name,
        )

        print("📝 index_document() returned:", info)

        return Response(info, status=201)


# ---------------------------------------------------------
# LIST DOCUMENTS
# ---------------------------------------------------------
class DocumentListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        docs = UploadedDocument.objects.filter(user=request.user).order_by("domain", "-created_at")

        grouped = {}
        for doc in docs:
            dom = (doc.domain or "general").lower()
            grouped.setdefault(dom, []).append({
                "id": str(doc.id),
                "title": doc.title,
                "filename": doc.original_filename,
                "domain": dom,
                "created_at": doc.created_at,
            })

        return Response(grouped, status=200)


# ---------------------------------------------------------
# DELETE DOCUMENT
# ---------------------------------------------------------
class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
        except UploadedDocument.DoesNotExist:
            return Response({"detail": "Document not found"}, status=404)

        chunk_ids = list(
            DocumentChunk.objects.filter(document=doc)
            .values_list("pinecone_id", flat=True)
        )

        if chunk_ids:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            index.delete(ids=chunk_ids)
            print("🗑 Deleted Pinecone vectors:", len(chunk_ids))

        doc.delete()
        return Response({"detail": "Document deleted"}, status=200)


# ---------------------------------------------------------
# SWITCH DOMAIN
# ---------------------------------------------------------
class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        domain = request.data.get("domain")
        if not domain:
            return Response({"detail": "Missing domain"}, status=400)

        request.session["active_domain"] = domain
        request.session.save()
        return Response({"detail": f"Active domain switched to {domain}"})


# ---------------------------------------------------------
# AGENT ASK
# ---------------------------------------------------------
class AgentAskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        query = request.data.get("query")
        req_mood = request.data.get("mood")
        provided_history = request.data.get("history", []) or []

        prefs = load_user_preferences(request.user)
        mood = req_mood or prefs.get("default_mood", "neutral")

        buffer = load_buffer(request.user)
        summary = load_summary(request.user)

        final_history = []
        if summary:
            final_history.append({"role": "system", "content": f"Summary: {summary}"})
        if buffer:
            final_history.extend(buffer)
        if provided_history:
            final_history.extend(provided_history)
        final_history = prune_buffer(final_history)

        agent = build_agent(
            user_id=request.user.id,
            domain=request.session.get("active_domain"),
            document_id=request.data.get("document_id"),
            mood=mood,
            history=final_history,
        )

        try:
            answer = agent.run(query)
        except Exception as e:
            answer = f"Agent error: {e}"

        QueryLog.objects.create(
            id=uuid.uuid4(),
            user=request.user,
            query=query,
            answer=answer[:4000],
            top_score=1.0,
            chunks=[],
        )

        return Response({"answer": answer, "validated": True, "chunks": [], "agent_mode": True})
