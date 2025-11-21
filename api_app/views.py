from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.db.models import Avg, Count
from django.utils import timezone
import uuid

from .serializers import AskSerializer, FeedbackSerializer
from .models import QueryLog, Feedback
from rag.pipeline import run_rag


class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data['query']
        result = run_rag(query)

        # Save query log
        top_score = result["chunks"][0]["score"] if result.get("chunks") else 0.0
        qlog = QueryLog.objects.create(
            id=uuid.uuid4(),
            user=request.user,
            query=query,
            answer=result.get("answer", "")[:4000],
            top_score=top_score,
            chunks=result.get("chunks", [])
        )

        return Response({"query_id": str(qlog.id), **result})
        

class FeedbackAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeedbackSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        fb = serializer.save()
        return Response({"detail": "feedback saved", "id": str(fb.id)}, status=201)


class AnalyticsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        
        # Only this user's logs
        user_logs = QueryLog.objects.filter(user=user)

        total = user_logs.count()

        today = user_logs.filter(
            created_at__date=timezone.now().date()
        ).count()

        avg_score = user_logs.aggregate(avg=Avg("top_score"))["avg"] or 0.0

        # Only this user's feedback
        fb = Feedback.objects.filter(user=user)
        up = fb.filter(value="up").count()
        down = fb.filter(value="down").count()

        top_queries = (
            user_logs.values("query")
            .annotate(count=Count("id"))
            .order_by("-count")[:10]
        )

        return Response({
            "total_queries": total,
            "queries_today": today,
            "avg_score": float(avg_score),
            "positive_feedback": up,
            "negative_feedback": down,
            "top_queries": list(top_queries)
        })
