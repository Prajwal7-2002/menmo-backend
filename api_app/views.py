import os
import uuid
import tempfile

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from django.db.models import Avg, Count
from django.utils import timezone

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback
from rag.models import UploadedDocument, ChunkFeedback, DocumentChunk

from rag.pipeline import run_rag
from rag.loader import index_document


# ---------------------------------------
# 🟦  ASK API
# ---------------------------------------
# ---------------------------------------
# 🟦  ASK API (supports optional document)
# ---------------------------------------
class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data["query"]

        active_domain = request.session.get("active_domain", None)
        document_id = request.data.get("document_id")  # optional

        result = run_rag(
            query,
            user_id=request.user.id,
            domain=active_domain,
            document_id=document_id   # <<< NEW
        )

        top_score = result["chunks"][0]["score"] if result.get("chunks") else 0.0

        qlog = QueryLog.objects.create(
            id=uuid.uuid4(),
            user=request.user,
            query=query,
            answer=result.get("answer", "")[:4000],
            top_score=top_score,
            chunks=result.get("chunks", []),
        )

        return Response({"query_id": str(qlog.id), **result})



# ---------------------------------------
# 🟩  FEEDBACK API
# ---------------------------------------
class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)

        fb = serializer.save()
        qlog = fb.query_log
        chunks = qlog.chunks or []

        # Apply feedback to each Pinecone chunk
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            pinecone_id = ch.get("id")
            if pinecone_id:
                ChunkFeedback.apply_feedback(pinecone_id, fb.value)

        return Response({"detail": "feedback saved", "id": str(fb.id)}, status=201)


# ---------------------------------------
# 🟧  ANALYTICS
# ---------------------------------------
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


# ---------------------------------------
# 🟪  UPLOAD DOCUMENT
# ---------------------------------------
class UploadDocumentAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"detail": "No file uploaded"}, status=400)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in file_obj.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        info = index_document(
            file_path=tmp_path,
            user_id=request.user.id,
            original_name=file_obj.name,
        )

        return Response(info, status=201)


# ---------------------------------------
# 🟨  LIST DOCUMENTS (grouped by domain)
# ---------------------------------------
class DocumentListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        docs = UploadedDocument.objects.filter(user=request.user).order_by("domain", "-created_at")

        result = {}
        for doc in docs:
            domain = (doc.domain or "general").lower()

            if domain not in result:
                result[domain] = []

            result[domain].append({
                "id": str(doc.id),
                "title": doc.title,
                "filename": doc.original_filename,
                "domain": domain,
                "created_at": doc.created_at,
            })

        return Response(result, status=200)


# ---------------------------------------
# 🟥  DELETE DOCUMENT
# ---------------------------------------
class DocumentDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, doc_id):
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
        except UploadedDocument.DoesNotExist:
            return Response({"detail": "Document not found"}, status=404)

        # delete Pinecone vectors
        chunk_ids = list(
            DocumentChunk.objects.filter(document=doc)
            .values_list("pinecone_id", flat=True)
        )

        if chunk_ids:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            index.delete(ids=chunk_ids)

        doc.delete()
        return Response({"detail": "Document deleted"}, status=200)


# ---------------------------------------
# 🟦  SWITCH ACTIVE DOMAIN
# ---------------------------------------
class SwitchDomainAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        domain = request.data.get("domain")
        if not domain:
            return Response({"detail": "Missing domain"}, status=400)

        request.session["active_domain"] = domain
        request.session.save()

        return Response({"detail": f"Active domain switched to {domain}"})
