"""
Offline tests for the RAG pipeline. The embedding model, Pinecone, Groq and
web search are all replaced by fakes, so these run without network or keys.
"""
import json
from unittest import mock

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from rag import agent as agent_mod
from rag import loader, retrieval, vectorstore
from rag.memory import clear_memory_for_document
from rag.models import DocumentChunk, UploadedDocument
from rag.tools import intent_classifier
from rag.tools.evaluate_context import evaluate_context_tool


def fake_vec(_text=None):
    return [0.1] * 384


class FakeLLM:
    """
    Answers each kind of prompt the code sends, recording calls.
    The router reply is a crude stand-in for the real model's judgement;
    pass `route` to force a specific routing decision.
    """

    def __init__(self, intent="knowledge", tone="friendly", verbosity="normal", route=None):
        self.intent, self.tone, self.verbosity, self.route = intent, tone, verbosity, route
        self.calls, self.user_messages = [], []

    def _route(self, user_content):
        latest = user_content.split("Latest message: ", 1)[-1]
        has_history = "Conversation:\n(none)" not in user_content
        low = latest.lower()
        fact = ""
        if low.startswith(("hi", "hello", "thanks")):
            intent = "chat"
        elif low.startswith("remember"):
            intent, fact = "memory_store", "FACT about the user"
        elif "this report" in low:
            intent = "doc_summary" if "summar" in low else "doc_lookup"
        else:
            intent = self.intent
        return {"intent": intent, "standalone": ("STANDALONE " + latest) if has_history else latest,
                "tone": self.tone, "verbosity": self.verbosity, "fact": fact, "confidence": 0.8}

    def __call__(self, messages, **kwargs):
        system = messages[0]["content"]
        self.calls.append(system)
        self.user_messages.append(messages[-1]["content"])
        if system.startswith("You route messages"):
            return json.dumps(self.route or self._route(messages[-1]["content"]))
        if system.startswith("Rewrite the search query"):
            return messages[-1]["content"] + " expanded"
        if "domain/category" in system:
            return "Medical Reports"
        return "ANSWER"


def match(id_, score, text, **meta):
    return {"id": id_, "score": score, "metadata": {"chunk_text": text, **meta}}


LONG = "Hemoglobin A1c reflects average blood glucose over roughly three months of time. "


class ChunkTextTests(SimpleTestCase):
    def test_short_text_is_single_chunk(self):
        self.assertEqual(loader.chunk_text("hello   world"), ["hello world"])

    def test_long_text_covers_everything_with_bounded_chunks(self):
        text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(400))
        chunks = loader.chunk_text(text, size=300, overlap=60)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        joined = " ".join(chunks)
        for i in (0, 199, 399):
            self.assertIn(f"Sentence number {i} ", joined)
        # cuts land on word boundaries
        self.assertTrue(all(not c.startswith(("umber", "entence")) for c in chunks))

    def test_no_whitespace_text_terminates(self):
        chunks = loader.chunk_text("x" * 2500, size=900, overlap=150)
        self.assertGreaterEqual(len(chunks), 3)


class FilterTests(SimpleTestCase):
    def test_user_always_in_filter(self):
        self.assertEqual(retrieval.build_filter(7), {"user_id": "7"})
        self.assertEqual(retrieval.build_filter(7, domain="Med"), {"user_id": "7", "domain": "med"})
        self.assertEqual(retrieval.build_filter(7, domain="Med", document_id="d1"),
                         {"user_id": "7", "document_id": "d1"})

    def test_shared_mode_drops_user(self):
        with mock.patch.object(retrieval, "SHARED_DOCUMENTS", True):
            self.assertEqual(retrieval.build_filter(7, document_id="d1"), {"document_id": "d1"})


class RetrieveTests(TestCase):
    def setUp(self):
        p = mock.patch.object(retrieval, "embed_query", side_effect=fake_vec)
        p.start()
        self.addCleanup(p.stop)

    def test_requires_user(self):
        with mock.patch.object(vectorstore, "query") as q:
            self.assertEqual(retrieval.retrieve("anything", user_id=None), [])
            q.assert_not_called()

    def test_ranks_and_keeps_raw_similarity(self):
        hits = [match("a", 0.20, "Unrelated text about the weather and nothing else at all here."),
                match("b", 0.62, LONG)]
        with mock.patch.object(vectorstore, "query", return_value=hits) as q:
            res = retrieval.retrieve("what does hemoglobin a1c reflect", user_id=3)
        self.assertEqual(q.call_args.kwargs["flt"], {"user_id": "3"})
        self.assertEqual(res[0]["id"], "b")
        self.assertEqual(res[0]["similarity"], 0.62)   # not min-max normalised
        self.assertEqual(res[1]["similarity"], 0.20)

    def test_domain_relaxation_keeps_user_scope(self):
        with mock.patch.object(vectorstore, "query", side_effect=[[], [match("b", 0.5, LONG)]]) as q:
            res = retrieval.retrieve("a1c", user_id=3, domain="medical")
        self.assertEqual([c.kwargs["flt"] for c in q.call_args_list],
                         [{"user_id": "3", "domain": "medical"}, {"user_id": "3"}])
        self.assertEqual(len(res), 1)

    def test_document_scope_never_widens(self):
        with mock.patch.object(vectorstore, "query", return_value=[]) as q:
            self.assertEqual(retrieval.retrieve("a1c", user_id=3, domain="medical", document_id="d9"), [])
        self.assertEqual(q.call_count, 1)

    def test_vectorstore_error_returns_empty(self):
        with mock.patch.object(vectorstore, "query", side_effect=vectorstore.VectorStoreError("down")):
            self.assertEqual(retrieval.retrieve("a1c", user_id=3), [])


class EvaluateTests(SimpleTestCase):
    def test_thresholds(self):
        self.assertEqual(evaluate_context_tool([])["quality"], "empty")
        self.assertEqual(evaluate_context_tool([{"similarity": 0.6}])["quality"], "good")
        self.assertEqual(evaluate_context_tool([{"similarity": 0.35}])["quality"], "weak")
        self.assertEqual(evaluate_context_tool([{"similarity": 0.1}])["quality"], "empty")


class RouterTests(SimpleTestCase):
    def route(self, q, has_document=False, history=None, llm=None):
        llm = llm or FakeLLM()
        with mock.patch("rag.llm.chat_completion", side_effect=llm),              mock.patch("rag.llm.GROQ_API_KEY", "x"):
            return intent_classifier.classify_intent_and_route(q, has_document=has_document, history=history)

    def test_llm_decides_intent_tone_and_length(self):
        llm = FakeLLM(route={"intent": "knowledge", "standalone": "What are metformin's side effects?",
                             "tone": "empathetic", "verbosity": "detailed", "confidence": 0.9})
        r = self.route("im scared, what are its side effects??",
                       history=[{"role": "user", "content": "I was prescribed metformin"}], llm=llm)
        self.assertEqual((r["intent"], r["tone"], r["verbosity"], r["note"]),
                         ("knowledge", "empathetic", "detailed", "llm"))
        self.assertEqual(r["standalone"], "What are metformin's side effects?")
        self.assertIn("I was prescribed metformin", llm.user_messages[0])  # history reached the router

    def test_greetings_also_go_through_the_llm(self):
        llm = FakeLLM()
        self.assertEqual(self.route("hi!", llm=llm)["intent"], "chat")
        self.assertEqual(len(llm.calls), 1)

    def test_invalid_fields_are_sanitised(self):
        llm = FakeLLM(route={"intent": "doc_lookup", "tone": "sarcastic", "verbosity": "huge",
                             "standalone": "", "confidence": "high"})
        r = self.route("what is the HbA1c in this report", has_document=False, llm=llm)
        self.assertEqual(r["intent"], "knowledge")          # no document selected
        self.assertEqual((r["tone"], r["verbosity"]), ("neutral", "normal"))
        self.assertEqual(r["standalone"], "what is the HbA1c in this report")
        self.assertEqual(r["confidence"], 0.7)

    def test_memory_fact_comes_from_llm_with_fallback(self):
        llm = FakeLLM(route={"intent": "memory_store", "fact": "The user's favourite language is Python"})
        self.assertEqual(self.route("remember I like python best", llm=llm)["fact"],
                         "The user's favourite language is Python")
        llm = FakeLLM(route={"intent": "memory_store"})
        self.assertEqual(self.route("Remember that I like Python", llm=llm)["fact"], "I like Python")

    def test_rules_used_when_llm_unavailable(self):
        from rag.llm import LLMError
        cases = [("hi!", False, "chat"), ("Remember that my project is about medical RAG", False, "memory_store"),
                 ("what did I tell you about my project?", False, "memory_recall"),
                 ("summarize this report", True, "doc_summary"),
                 ("what is the HbA1c in this report", True, "doc_lookup"),
                 ("explain transformers", False, "knowledge")]
        for q, has_doc, expected in cases:
            r = self.route(q, has_document=has_doc, llm=mock.Mock(side_effect=LLMError("down")))
            self.assertEqual((r["intent"], r["note"]), (expected, "rules"), q)


class ModelFallbackTests(SimpleTestCase):
    def test_retired_model_falls_back_and_is_skipped_afterwards(self):
        from rag import llm

        class Resp:
            def __init__(self, status, body):
                self.status_code, self.text, self._body, self.headers = status, json.dumps(body), body, {}

            def json(self):
                return self._body

        gone = Resp(404, {"error": {"code": "model_not_found"}})
        ok = Resp(200, {"choices": [{"message": {"content": "hi"}}]})
        session = mock.Mock()
        session.post.side_effect = [gone, ok, ok]
        with mock.patch.object(llm, "_session", return_value=session),              mock.patch.object(llm, "GROQ_API_KEY", "x"),              mock.patch.object(llm, "GROQ_MODEL", "old/model"),              mock.patch.object(llm, "GROQ_FALLBACK_MODELS", ["new/model"]),              mock.patch.object(llm, "_retired_models", set()):
            self.assertEqual(llm.chat_completion([{"role": "user", "content": "x"}]), "hi")
            self.assertEqual(llm.chat_completion([{"role": "user", "content": "x"}]), "hi")
        models = [c.kwargs["json"]["model"] for c in session.post.call_args_list]
        self.assertEqual(models, ["old/model", "new/model", "new/model"])  # retired model not retried

    def test_reasoning_model_gets_room_to_think_and_citations_normalised(self):
        from rag import llm
        resp = mock.Mock(status_code=200, headers={})
        resp.json.return_value = {"choices": [{"message": {"content": "HbA1c is 5.4%【1】 and normal【2†source】."}}]}
        session = mock.Mock()
        session.post.return_value = resp
        with mock.patch.object(llm, "_session", return_value=session), mock.patch.object(llm, "GROQ_API_KEY", "x"):
            out = llm._complete_with("openai/gpt-oss-120b", [{"role": "user", "content": "x"}], 12, 0.0)
            self.assertEqual(session.post.call_args.kwargs["json"]["max_tokens"], llm.REASONING_MIN_TOKENS)
            llm._complete_with("qwen/qwen3.8-27b", [{"role": "user", "content": "x"}], 12, 0.0)
            self.assertEqual(session.post.call_args.kwargs["json"]["max_tokens"], 12)
        self.assertEqual(out, "HbA1c is 5.4%[1] and normal[2].")

    def test_reasoning_params_only_for_gpt_oss(self):
        from rag.llm import _model_params
        self.assertEqual(_model_params("openai/gpt-oss-120b")["include_reasoning"], False)
        self.assertEqual(_model_params("qwen/qwen3.8-27b"), {})


class ToneTests(SimpleTestCase):
    def test_explicit_mood_wins_auto_otherwise(self):
        from rag.llm import resolve_tone
        self.assertEqual(resolve_tone("formal", "playful"), "formal")
        self.assertEqual(resolve_tone("joke", "formal"), "playful")      # legacy alias
        for auto in (None, "", "auto", "neutral"):
            self.assertEqual(resolve_tone(auto, "empathetic"), "empathetic")
        self.assertEqual(resolve_tone(None, "bogus"), "neutral")

    def test_tone_and_length_reach_the_prompt(self):
        from rag.llm import generate_answer
        with mock.patch("rag.llm.chat_completion", return_value="ok") as cc:
            generate_answer("q", "general", tone="empathetic", verbosity="brief")
        system = cc.call_args.args[0][0]["content"]
        self.assertIn("reassuring", system)
        self.assertIn("one to three sentences", system)
        self.assertEqual(cc.call_args.kwargs["max_tokens"], 300)


class AgentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("u", password="x")
        self.llm = FakeLLM()
        for p in (mock.patch("rag.llm.chat_completion", side_effect=self.llm),
                  mock.patch("rag.llm.GROQ_API_KEY", "x")):
            p.start()
            self.addCleanup(p.stop)

    def run_agent(self, chunks, web=(), **kw):
        with mock.patch.object(agent_mod, "retrieve", return_value=list(chunks)) as r, \
             mock.patch.object(agent_mod, "web_search_chunks", return_value=list(web)) as w:
            res = agent_mod.build_agent(self.user, **kw).run("question")
        return res, r, w

    def test_document_mode_stays_in_document(self):
        res, r, w = self.run_agent([{"id": "c", "text": LONG, "similarity": 0.2, "meta": {"chunk_text": LONG}}],
                                   web=[{"text": "web"}], document_id="d1", question_intent="doc_lookup")
        self.assertEqual(res["source"], "document")
        w.assert_not_called()
        self.assertTrue(all(c.kwargs["document_id"] == "d1" for c in r.call_args_list))
        self.assertNotIn("chunk_text", res["chunks"][0]["meta"])  # trimmed from response

    def test_document_mode_without_chunks(self):
        res, _, w = self.run_agent([], document_id="d1", question_intent="doc_summary")
        self.assertIn("couldn't find any indexed text", res["answer"])
        w.assert_not_called()

    def test_open_mode_good_docs(self):
        res, _, w = self.run_agent([{"id": "c", "text": LONG, "similarity": 0.7}], question_intent="knowledge")
        self.assertEqual(res["source"], "document")
        w.assert_not_called()

    def test_open_mode_falls_back_to_web_then_llm(self):
        res, _, _ = self.run_agent([], web=[{"text": "web result", "meta": {"type": "web"}}])
        self.assertEqual(res["source"], "web")
        res, _, _ = self.run_agent([], web=[])
        self.assertEqual(res["source"], "llm")

    def test_weak_match_triggers_one_rewrite(self):
        weak = [{"id": "c", "text": LONG, "similarity": 0.35}]
        _, r, _ = self.run_agent(weak, web=[{"text": "w"}])
        self.assertEqual(r.call_count, 2)
        self.assertEqual(r.call_args_list[1].args[0], "question expanded")


class LoaderTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("u", password="x")
        self.enterContext(mock.patch.object(loader, "embed_texts",
                                                        side_effect=lambda ts: [fake_vec() for _ in ts]))
        self.enterContext(mock.patch("rag.llm.chat_completion", side_effect=FakeLLM()))
        self.enterContext(mock.patch("rag.llm.GROQ_API_KEY", "x"))

    def write(self, text, name="notes.txt"):
        import os
        import tempfile
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(name)[1])
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_index_txt(self):
        with mock.patch.object(loader, "upsert_chunks") as up:
            res = loader.index_document(self.write(LONG * 40), self.user.id, "lab notes.txt")
        doc = UploadedDocument.objects.get(id=res["document_id"])
        self.assertEqual(doc.domain, "medical_reports")
        self.assertEqual(doc.title, "lab notes")
        self.assertEqual(DocumentChunk.objects.filter(document=doc).count(), res["chunks_indexed"])
        items = up.call_args.args[0]
        self.assertEqual(len(items), res["chunks_indexed"])
        self.assertEqual(items[0]["metadata"]["user_id"], str(self.user.id))
        self.assertEqual(items[0]["metadata"]["document_id"], res["document_id"])

    def test_upsert_failure_rolls_back(self):
        with mock.patch.object(loader, "upsert_chunks", side_effect=vectorstore.VectorStoreError("x")), \
             mock.patch.object(loader, "delete_chunks") as cleanup:
            with self.assertRaises(loader.IndexingError):
                loader.index_document(self.write(LONG), self.user.id, "a.txt")
        self.assertFalse(UploadedDocument.objects.exists())
        self.assertFalse(DocumentChunk.objects.exists())
        cleanup.assert_called_once()

    def test_empty_and_unsupported(self):
        with self.assertRaises(loader.IndexingError):
            loader.index_document(self.write("   "), self.user.id, "a.txt")
        with self.assertRaises(loader.IndexingError):
            loader.index_document(self.write("x", name="a.xyz"), self.user.id, "a.xyz")


class VectorstoreTests(SimpleTestCase):
    def test_missing_index_raises_vectorstore_error(self):
        client = mock.Mock()
        client.Index.side_effect = RuntimeError("(404) Resource neurostack-rag-dev not found")
        with mock.patch.object(vectorstore, "PINECONE_API_KEY", "x"),              mock.patch.object(vectorstore, "_client", client),              mock.patch.object(vectorstore, "_indexes", {}):
            with self.assertRaises(vectorstore.VectorStoreError):
                vectorstore.get_index("neurostack-rag-dev")
            self.assertEqual(vectorstore._indexes, {})  # failure not cached; retried next time

    def test_delete_by_filter_stops_when_nothing_new(self):
        page = [{"id": "m1"}, {"id": "m2"}]
        with mock.patch.object(vectorstore, "query", return_value=page) as q, \
             mock.patch.object(vectorstore, "delete_ids") as d:
            n = clear_memory_for_document(5, "doc")
        self.assertEqual(n, 2)
        self.assertEqual(q.call_args.kwargs["flt"], {"user": "5", "document_id": "doc"})
        d.assert_called_once_with(vectorstore.MEMORY_INDEX_NAME, ["m1", "m2"])
