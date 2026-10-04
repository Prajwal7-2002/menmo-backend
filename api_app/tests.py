"""
API tests. External services (embeddings, Pinecone, Groq, web) are faked;
see rag/tests.py for the pipeline-level tests.
"""
import uuid
from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase

from api_app.models import Conversation, Message, QueryLog
from rag.models import DocumentChunk, UploadedDocument
from rag.tests import FakeLLM


class BaseAPITest(APITestCase):
    def setUp(self):
        cache.clear()  # throttle counters
        self.user = User.objects.create_user("alice", password="x")
        self.other = User.objects.create_user("bob", password="x")
        self.client.force_authenticate(self.user)

        self.llm = FakeLLM()
        self.memory = mock.MagicMock()
        self.memory.add_memory_fact.return_value = "stored"
        self.agent_result = {"answer": "agent answer", "answer_source": "document", "confidence": 0.7,
                             "chunks": [{"id": "chunk-1", "text": "t", "meta": {}}], "trace": []}
        for p in (
            mock.patch("rag.llm.chat_completion", side_effect=self.llm),
            mock.patch("rag.llm.GROQ_API_KEY", "x"),
            mock.patch("api_app.views.memory", self.memory),
            mock.patch("api_app.views.build_agent"),
        ):
            started = p.start()
            self.addCleanup(p.stop)
        self.build_agent = started
        self.build_agent.return_value.run.return_value = self.agent_result

    def doc_for(self, user):
        return UploadedDocument.objects.create(user=user, original_filename="r.pdf", domain="medical")


class ConversationChatTests(BaseAPITest):
    def setUp(self):
        super().setUp()
        self.conv = Conversation.objects.create(user=self.user)
        self.url = f"/api/conversations/{self.conv.id}/chat/"

    def test_greeting_works(self):
        # Regression: `agent_mode` was undefined here, so every greeting returned 500.
        res = self.client.post(self.url, {"query": "hi"}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(res.data["source"], "chat")
        self.build_agent.assert_not_called()
        self.assertEqual(Message.objects.filter(conversation=self.conv).count(), 2)
        self.conv.refresh_from_db()
        self.assertEqual(self.conv.title, "hi")

    def test_agent_mode_skips_chat_shortcut(self):
        res = self.client.post(self.url, {"query": "hi", "agent_mode": True}, format="json")
        self.assertEqual(res.status_code, 200)
        self.build_agent.assert_called_once()

    def test_question_goes_to_agent_with_history(self):
        Message.objects.create(conversation=self.conv, role="user", content="Tell me about metformin")
        Message.objects.create(conversation=self.conv, role="assistant", content="Metformin is ...")
        res = self.client.post(self.url, {"query": "what are its side effects?"}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        kwargs = self.build_agent.call_args.kwargs
        self.assertEqual([m["content"] for m in kwargs["history"]],
                         ["Tell me about metformin", "Metformin is ..."])
        # follow-up was condensed into a standalone question before running
        self.build_agent.return_value.run.assert_called_once_with("STANDALONE what are its side effects?")
        self.assertEqual(res.data["answer"], "agent answer")
        self.assertTrue(QueryLog.objects.filter(id=res.data["query_id"], user=self.user).exists())

    def test_selected_document_is_passed(self):
        doc = self.doc_for(self.user)
        res = self.client.post(self.url, {"query": "summarize this report", "document_id": str(doc.id)},
                               format="json")
        self.assertEqual(res.status_code, 200, res.content)
        kwargs = self.build_agent.call_args.kwargs
        self.assertEqual(kwargs["document_id"], doc.id)
        self.assertEqual(kwargs["question_intent"], "doc_summary")

    def test_other_users_document_rejected(self):
        doc = self.doc_for(self.other)
        res = self.client.post(self.url, {"query": "summarize this report", "document_id": str(doc.id)},
                               format="json")
        self.assertEqual(res.status_code, 404)
        self.build_agent.assert_not_called()

    def test_remember(self):
        res = self.client.post(self.url, {"query": "Remember that my project is about medical RAG"},
                               format="json")
        self.assertEqual(res.status_code, 200)
        # the router's rephrased fact is stored, not the raw message
        self.memory.add_memory_fact.assert_called_once_with(self.user, "FACT about the user")
        self.assertEqual(res.data["source"], "memory")

    def test_tone_is_inferred_unless_client_sets_mood(self):
        self.llm.tone, self.llm.verbosity = "empathetic", "detailed"
        res = self.client.post(self.url, {"query": "explain my options", "mood": "neutral"}, format="json")
        self.assertEqual((res.data["tone"], res.data["verbosity"]), ("empathetic", "detailed"))
        kwargs = self.build_agent.call_args.kwargs
        self.assertEqual((kwargs["tone"], kwargs["verbosity"]), ("empathetic", "detailed"))

        res = self.client.post(self.url, {"query": "explain my options", "mood": "formal"}, format="json")
        self.assertEqual(res.data["tone"], "formal")

    def test_other_users_conversation(self):
        conv = Conversation.objects.create(user=self.other)
        res = self.client.post(f"/api/conversations/{conv.id}/chat/", {"query": "hi"}, format="json")
        self.assertEqual(res.status_code, 404)

    def test_delete_conversation_clears_memory(self):
        res = self.client.delete(f"/api/conversations/{self.conv.id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.memory.clear_memory_for_conversation.assert_called_once_with(self.user.id, str(self.conv.id))


class AskTests(BaseAPITest):
    def test_ask(self):
        res = self.client.post("/api/ask/", {"query": "what is GraphRAG"}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(res.data["source"], "document")

    def test_requires_auth(self):
        self.client.force_authenticate(None)
        self.assertEqual(self.client.post("/api/ask/", {"query": "x"}, format="json").status_code, 401)


class FeedbackTests(BaseAPITest):
    def test_cannot_rate_other_users_answer(self):
        qlog = QueryLog.objects.create(user=self.other, query="q", answer="a", chunks=[])
        res = self.client.post("/api/feedback/", {"query_id": str(qlog.id), "value": "down"}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_rating_updates_chunk_feedback(self):
        qlog = QueryLog.objects.create(user=self.user, query="q", answer="a",
                                       chunks=[{"id": "c1"}, {"id": None, "meta": {"type": "web"}}])
        with mock.patch("api_app.views.ChunkFeedback.apply_feedback") as fb:
            res = self.client.post("/api/feedback/", {"query_id": str(qlog.id), "value": "up"}, format="json")
        self.assertEqual(res.status_code, 201)
        fb.assert_called_once_with("c1", "up")


class DocumentTests(BaseAPITest):
    def test_upload_rejects_unsupported_type(self):
        f = SimpleUploadedFile("evil.exe", b"MZ")
        res = self.client.post("/api/upload-document/", {"file": f}, format="multipart")
        self.assertEqual(res.status_code, 400)

    def test_upload_success_and_temp_cleanup(self):
        seen = {}

        def fake_index(path, user_id, name):
            seen["path"] = path
            return {"document_id": str(uuid.uuid4()), "chunks_indexed": 3, "domain": "general", "title": "n"}

        with mock.patch("api_app.views.index_document", side_effect=fake_index):
            res = self.client.post("/api/upload-document/",
                                   {"file": SimpleUploadedFile("n.txt", b"hello")}, format="multipart")
        self.assertEqual(res.status_code, 201)
        import os
        self.assertFalse(os.path.exists(seen["path"]))

    def test_upload_indexing_error(self):
        from rag.loader import IndexingError
        with mock.patch("api_app.views.index_document", side_effect=IndexingError("No text")):
            res = self.client.post("/api/upload-document/",
                                   {"file": SimpleUploadedFile("n.pdf", b"%PDF")}, format="multipart")
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.data["detail"], "No text")

    def test_delete_keeps_document_if_vector_delete_fails(self):
        doc = self.doc_for(self.user)
        DocumentChunk.objects.create(document=doc, chunk_idx=0, pinecone_id="p1")
        with mock.patch("api_app.views.delete_chunks", side_effect=RuntimeError("pinecone down")):
            res = self.client.delete(f"/api/delete-document/{doc.id}/")
        self.assertEqual(res.status_code, 502)
        self.assertTrue(UploadedDocument.objects.filter(id=doc.id).exists())

    def test_delete_success(self):
        doc = self.doc_for(self.user)
        DocumentChunk.objects.create(document=doc, chunk_idx=0, pinecone_id="p1")
        with mock.patch("api_app.views.delete_chunks") as dc:
            res = self.client.delete(f"/api/delete-document/{doc.id}/")
        self.assertEqual(res.status_code, 200)
        dc.assert_called_once_with(["p1"])
        self.assertFalse(UploadedDocument.objects.filter(id=doc.id).exists())
        self.memory.clear_memory_for_document.assert_called_once_with(self.user.id, str(doc.id))

    def test_cannot_delete_other_users_document(self):
        doc = self.doc_for(self.other)
        self.assertEqual(self.client.delete(f"/api/delete-document/{doc.id}/").status_code, 404)


class AuthTests(APITestCase):
    def setUp(self):
        cache.clear()

    def test_signup_rejects_weak_password(self):
        res = self.client.post("/auth/signup/", {"username": "carol", "password": "123"}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_signup_and_login(self):
        res = self.client.post("/auth/signup/", {"username": "carol", "password": "a-Strong-pass-42"},
                               format="json")
        self.assertEqual(res.status_code, 201, res.content)
        res = self.client.post("/auth/login/", {"username": "carol", "password": "a-Strong-pass-42"},
                               format="json")
        self.assertIn("access", res.data)
