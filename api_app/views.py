# api/views.py
import os
import uuid
import tempfile
from pathlib import Path

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
from rag.memory import store_conversation_turn, load_vector_memory
from api_app.models import Conversation, Message


# ---------------------------------------------------------
# ASK API
# ---------------------------------------------------------
class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user  = request.user
        query = serializer.validated_data["query"]
        mood  = request.data.get("mood", "neutral")

        # optional context
        domain      = request.session.get("active_domain")
        document_id = request.data.get("document_id")

        # 🧠 retrieve semantic memory from Pinecone
        memory = load_vector_memory(user, query)

        result = run_rag(
            query       = query,
            user_id     = user.id,
            domain      = domain,
            document_id = document_id,
            mood        = mood,
            history     = memory,
        )

        # 🔥 store new message + answer in vector DB
        store_conversation_turn(user, query, result["answer"])

        return Response(result)



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
        user   = request.user
        query  = request.data.get("query", "")
        mood   = request.data.get("mood", "neutral")
        domain = request.session.get("active_domain")
        document_id = request.data.get("document_id")

        # 🧠 Get memory relevant to the query
        memory = load_vector_memory(user, query)

        agent = build_agent(
            user_id     = user.id,
            domain      = domain,
            document_id = document_id,
            mood        = mood,
            history     = memory,   # ✅ build_agent expects history
        )

        try:
            answer = agent.run(query)
        except Exception as e:
            answer = f"Agent error: {e}"

        # 🔥 Save conversation memory
        store_conversation_turn(user, query, answer)

        return Response({"answer": answer, "agent_mode": True})





# CREATE CHAT SESSION
class CreateConversationAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def post(self, request):
        conv = Conversation.objects.create(user=request.user)
        return Response({"id": str(conv.id)}, status=201)


# LIST ALL CONVERSATIONS
class ConversationListAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def get(self, request):
        chats = Conversation.objects.filter(user=request.user).order_by("-updated_at")
        return Response([
            {"id": str(c.id), "title": c.title, "updated_at": c.updated_at}
            for c in chats
        ])


# CHAT WITH VECTOR MEMORY (PER SESSION)
class ConversationChatAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail": "Conversation not found"}, status=404)

        query = (request.data.get("query") or "").strip()
        if not query:
            return Response({"detail": "Query is required"}, status=400)

        # 🔥 Extra controls from frontend
        agent_mode  = bool(request.data.get("agent_mode", False))
        mood        = request.data.get("mood", "neutral")
        document_id = request.data.get("document_id")
        domain      = request.session.get("active_domain")

        # 🧠 vector memory (Pinecone)
        memory = load_vector_memory(request.user, query)

        # ================== AGENT MODE ================== #
# ================== AGENT MODE ================== #
        if agent_mode:
            agent = build_agent(
                user_id     = request.user.id,
                domain      = domain,
                document_id = document_id,
                mood        = mood,
                history     = memory,
                agent_enabled = True      # 🔥 THIS IS THE KEY
)


            agent.agent_enabled = True      #  <<< REQUIRED 🔥🔥

            try:
                answer = agent.run(query)
            except Exception as e:
                answer = f"Agent error: {e}"

            # Save messages + semantic memory
            Message.objects.create(conversation=conv, role="user",       content=query)
            Message.objects.create(conversation=conv, role="assistant",  content=answer)
            store_conversation_turn(request.user, query, answer)

            return Response({"answer": answer, "mode": "agent"}, status=200)


        # ================== RAG MODE (DEFAULT) ================== #
        response = run_rag(
            query       = query,
            user_id     = request.user.id,
            domain      = domain,
            document_id = document_id,
            mood        = mood,
            history     = memory,
        )

        # save message + vector memory
        Message.objects.create(conversation=conv, role="user",      content=query)
        Message.objects.create(conversation=conv, role="assistant", content=response.get("answer", ""))

        store_conversation_turn(request.user, query, response.get("answer", ""))
        print("🔥 agent_mode =", request.data.get("agent_mode"))


        return Response(response, status=200)



# GET CHAT HISTORY (NO BUFFER ANYMORE)
class ConversationHistoryAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def get(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except:
            return Response({"detail":"Conversation not found"}, status=404)

        messages = conv.messages.order_by("timestamp")

        return Response([
            {"role":m.role,"content":m.content} for m in messages
        ])
    
# RENAME CHAT TITLE
class RenameConversationAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def patch(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail":"Not found"}, status=404)

        new_title = request.data.get("title", "").strip()
        if not new_title:
            return Response({"detail":"Title required"}, status=400)

        conv.title = new_title
        conv.save()
        return Response({"detail":"Renamed successfully", "title":new_title})


# DELETE A CHAT
class DeleteConversationAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def delete(self, request, cid):
        try:
            conv = Conversation.objects.get(id=cid, user=request.user)
        except Conversation.DoesNotExist:
            return Response({"detail":"Not found"}, status=404)

        conv.delete()
        return Response({"detail":"Conversation deleted"}, status=200)

