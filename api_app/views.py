# =======================================================
# api/views.py  — FINAL WORKING VERSION
# Analytics + Agent Mode Fixed
# =======================================================

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


# -------------------------------------------------------
# ❗ ASK (not conversation aware) — ANALYTICS ENABLED
# -------------------------------------------------------
class AskAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user  = request.user
        query = serializer.validated_data["query"]

        mood        = request.data.get("mood", "neutral")
        domain      = request.session.get("active_domain")
        document_id = request.data.get("document_id")
        memory      = load_vector_memory(user, query)

        result = run_rag(query, user.id, domain, document_id, mood, memory)

        store_conversation_turn(user, query, result["answer"])

        # 📌 REQUIRED FOR ANALYTICS
        QueryLog.objects.create(
            user=user,
            query=query,
            top_score=result.get("top_score",0),
            chunks=result.get("chunks",[])
        )

        return Response(result)


# -------------------------------------------------------
# FEEDBACK SYSTEM
# -------------------------------------------------------
class FeedbackAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def post(self,request):
        serializer=FeedbackSerializer(data=request.data,context={"request":request})
        serializer.is_valid(raise_exception=True)

        fb=serializer.save()
        qlog=fb.query_log
        for ch in (qlog.chunks or []):
            if ch.get("id"): ChunkFeedback.apply_feedback(ch["id"],fb.value)

        return Response({"detail":"feedback saved"},201)


# -------------------------------------------------------
# 📊 ANALYTICS (Now Works)
# -------------------------------------------------------
class AnalyticsAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def get(self, request):
        user=request.user
        logs=QueryLog.objects.filter(user=user)

        return Response({
            "total_queries":      logs.count(),
            "queries_today":      logs.filter(created_at__date=timezone.now().date()).count(),
            "avg_score":          float(logs.aggregate(avg=Avg("top_score"))["avg"] or 0.0),
            "positive_feedback":  Feedback.objects.filter(user=user,value="up").count(),
            "negative_feedback":  Feedback.objects.filter(user=user,value="down").count(),
            "top_queries":        list(logs.values("query").annotate(count=Count("id")).order_by("-count")[:10]),
        })


# -------------------------------------------------------
# UPLOAD DOCUMENT
# -------------------------------------------------------
class UploadDocumentAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def post(self,request):
        file=request.FILES.get("file")
        if not file:return Response({"detail":"No file"},400)

        ext=Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False,suffix=ext) as tmp:
            for c in file.chunks(): tmp.write(c)
            path=tmp.name

        return Response(index_document(path,request.user.id,file.name),201)


# -------------------------------------------------------
# LIST DOCUMENTS BY DOMAIN
# -------------------------------------------------------
class DocumentListAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def get(self,request):
        docs=UploadedDocument.objects.filter(user=request.user).order_by("domain","-created_at")
        data={}
        for d in docs:
            dom=(d.domain or "general").lower()
            data.setdefault(dom,[]).append({
                "id":str(d.id),"title":d.title,"filename":d.original_filename,
                "domain":dom,"created_at":d.created_at
            })
        return Response(data)


# -------------------------------------------------------
# DELETE DOCUMENT + Remove Vectors
# -------------------------------------------------------
class DocumentDeleteAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def delete(self,request,doc_id):
        try: doc=UploadedDocument.objects.get(id=doc_id,user=request.user)
        except: return Response({"detail":"Not found"},404)

        ids=list(DocumentChunk.objects.filter(document=doc).values_list("pinecone_id",flat=True))
        if ids:
            from pinecone import Pinecone
            pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            pc.Index(os.getenv("PINECONE_INDEX_NAME")).delete(ids=ids)

        doc.delete()
        return Response({"detail":"Deleted"},200)


# -------------------------------------------------------
# DOMAIN SWITCH
# -------------------------------------------------------
class SwitchDomainAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def post(self,request):
        domain=request.data.get("domain")
        if not domain:return Response({"detail":"domain required"},400)
        request.session["active_domain"]=domain;request.session.save()
        return Response({"detail":f"Domain -> {domain}"})
    

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
        return Response([
            {
                "id": str(c.id),
                "title": c.title,
                "updated_at": c.updated_at
            }
            for c in chats
        ])




# -------------------------------------------------------
# 🔥 MAIN CHAT VIEW — RAG + AGENT MODE
# -------------------------------------------------------
class ConversationChatAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def post(self,request,cid):
        try: conv=Conversation.objects.get(id=cid,user=request.user)
        except:return Response({"detail":"conversation missing"},404)

        query=(request.data.get("query") or "").strip()
        if not query:return Response({"detail":"empty prompt"},400)

        agent_mode=bool(request.data.get("agent_mode"))
        domain=request.session.get("active_domain")
        mood=request.data.get("mood","neutral")
        doc_id=request.data.get("document_id")
        memory=load_vector_memory(request.user,query)


        # ============ AGENT MODE 🔥 ============
        if agent_mode:
            agent=build_agent(
                user_id=request.user.id,domain=domain,
                document_id=doc_id,mood=mood,
                history=memory,agent_enabled=True
            )
            answer=agent.run(query)

            Message.objects.create(conversation=conv,role="user",content=query)
            Message.objects.create(conversation=conv,role="assistant",content=answer)
            store_conversation_turn(request.user,query,answer)

            # 📌 log in analytics
            QueryLog.objects.create(user=request.user,query=query,top_score=0,chunks=[])

            return Response({"answer":answer,"mode":"agent"})


        # ============ RAG MODE =============
        res=run_rag(query,request.user.id,domain,doc_id,mood,memory)

        Message.objects.create(conversation=conv,role="user",content=query)
        Message.objects.create(conversation=conv,role="assistant",content=res.get("answer",""))
        store_conversation_turn(request.user,query,res.get("answer",""))

        # 📌 log for analytics
        QueryLog.objects.create(
            user=request.user,query=query,
            top_score=res.get("top_score",0),
            chunks=res.get("chunks",[])
        )

        return Response(res)


# ========= Chat History / Rename / Delete ========= #

class ConversationHistoryAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def get(self,request,cid):
        try: conv=Conversation.objects.get(id=cid,user=request.user)
        except:return Response({"detail":"not found"},404)
        return Response([{"role":m.role,"content":m.content}for m in conv.messages.order_by("timestamp")])


class RenameConversationAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def patch(self,request,cid):
        try:conv=Conversation.objects.get(id=cid,user=request.user)
        except:return Response({"detail":"not found"},404)
        name=request.data.get("title","").strip()
        if not name:return Response({"detail":"required"},400)
        conv.title=name;conv.save()
        return Response({"detail":"renamed","title":name})


class DeleteConversationAPIView(APIView):
    permission_classes=[IsAuthenticated]

    def delete(self,request,cid):
        try:conv=Conversation.objects.get(id=cid,user=request.user)
        except:return Response({"detail":"not found"},404)
        conv.delete()
        return Response({"detail":"deleted"},200)