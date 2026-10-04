---
title: "Menmo Backend"
emoji: "⚙️"
colorFrom: "indigo"
colorTo: "blue"
sdk: "docker"
pinned: false
---

# Menmo Backend

Django REST backend for Menmo: upload documents, then chat with them. Answers come from your
documents when they're relevant, otherwise from a web search, otherwise from the LLM's general
knowledge — and every response says which (`source`).

## How a question is answered

```
query ──► router (one LLM call, sees the recent chat) decides:
            intent · tone · answer length · standalone rewrite of follow-ups
            (regex rules only if the LLM is unreachable)
           chat           → short LLM reply
           memory_store   → save fact to memory index
           memory_recall  → search saved facts + past turns → answer
           doc_summary /  → selected document ONLY → answer
           doc_lookup       (never other documents or the web)
           anything else  → user's documents
                              similarity ≥ RAG_MIN_GOOD_SIM → answer from documents
                              weaker → one query rewrite, retry
                            → web search (DuckDuckGo) → answer with citations
                            → general LLM knowledge
```

Tone (`neutral`, `friendly`, `formal`, `empathetic`, `playful`, `concise`) and length
(`brief`, `normal`, `detailed`) are inferred from how the user writes. A client can still force a
tone by sending `mood`; `auto`, `neutral` or no `mood` lets the router choose.

Retrieval is hybrid (dense cosine + BM25 + optional cross-encoder + thumbs up/down feedback), but
relevance decisions use the raw cosine similarity, which is comparable across queries.
Each user only ever searches their own documents.

| Piece | Where |
|---|---|
| API views | `api_app/views.py` |
| Orchestration | `rag/agent.py` |
| Retrieval / Pinecone | `rag/retrieval.py`, `rag/vectorstore.py` |
| Ingestion (PDF/OCR/DOCX/TXT/MD) | `rag/loader.py` |
| Memory | `rag/memory.py` |
| LLM (Groq) + prompts | `rag/llm.py` |
| Router (intent/tone), rewrite, web search, relevance | `rag/tools/` |

## Configuration (Space secrets / env vars)

Required in production:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Django/JWT signing key. **The app refuses to start without it** outside local dev. |
| `GROQ_API_KEY` | LLM access |
| `PINECONE_API_KEY` | Vector store |
| `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT` | Postgres (Supabase). Without `DB_HOST`, SQLite is used. |

Optional:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model id |
| `GROQ_FALLBACK_MODELS` | `qwen/qwen3.8-27b` | Comma-separated models tried if the main one is retired |
| `PINECONE_INDEX_NAME` / `PINECONE_MEMORY_INDEX` | `neurostack-rag` / `neurostack-memory` | Index names (384-dim; kept from the project's earlier name so existing data stays reachable) |
| `CORS_ALLOWED_ORIGINS` | *(all)* | Comma-separated frontend origins |
| `DJANGO_ALLOWED_HOSTS` | Space host, `.hf.space`, localhost | Comma-separated |
| `RAG_MIN_GOOD_SIM` / `RAG_MIN_WEAK_SIM` | `0.45` / `0.30` | Similarity needed to answer from documents |
| `RAG_SHARED_DOCUMENTS` | `false` | `true` lets every user search every document (old behaviour) |
| `ALLOW_RELAXED_FALLBACK` | `true` | If a domain has no hits, search all of the user's documents |
| `USE_RERANKER` | `false` | Cross-encoder reranking (slower, more RAM) |
| `WEB_SEARCH_ENABLED` | `true` | DuckDuckGo fallback |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters per chunk (new uploads only) |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `THROTTLE_ASK` / `THROTTLE_UPLOAD` / `THROTTLE_AUTH` | `30/min` / `20/hour` / `10/min` | Rate limits |
| `GUNICORN_WORKERS` / `GUNICORN_THREADS` / `GUNICORN_TIMEOUT` | `2` / `4` / `180` | Server tuning |
| `LOG_LEVEL` | `INFO` | |

## API

All `/api/*` endpoints need `Authorization: Bearer <access token>`.

| Method | Path | Body / notes |
|---|---|---|
| POST | `/auth/signup/` | `username`, `email`, `password` |
| POST | `/auth/login/` | `username`, `password` → `access`, `refresh` |
| POST | `/auth/refresh/` | `refresh` |
| POST | `/auth/reset-password/` | `old_password`, `new_password` |
| POST | `/api/upload-document/` | multipart `file` (.pdf .docx .txt .md) |
| GET | `/api/my-documents/` | documents grouped by domain |
| DELETE | `/api/delete-document/<id>/` | removes vectors, chunks and related memory |
| POST | `/api/conversations/` | create chat |
| GET | `/api/conversations/list/` | |
| POST | `/api/conversations/<id>/chat/` | `query`, optional `document_id`, `domain`, `mood`, `agent_mode` |
| GET | `/api/conversations/<id>/history/` | |
| PATCH | `/api/conversations/<id>/rename/` | `title` |
| DELETE | `/api/conversations/<id>/delete/` | |
| POST | `/api/ask/` | one-off question, same fields as chat |
| POST | `/api/feedback/` | `query_id`, `value` (`up`/`down`), `reason` |
| GET | `/api/analytics/` | |
| GET | `/health/` | liveness |

Chat/ask responses: `answer`, `source` (`document` · `web` · `llm` · `memory` · `chat`), `intent`, `tone`, `verbosity`,
`confidence`, `chunks` (with `meta.page_num`, `meta.title`, or `meta.source` URL for web), `trace`, `query_id`.

## Local development

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install -r requirements.txt python-dotenv
cp .env.example .env    # or create .env with the variables above
python manage.py migrate
python manage.py runserver
```

### Tests

The tests fake the embedding model, Pinecone, Groq and web search, so they need no keys or network
and only the light dependencies:

```bash
pip install Django djangorestframework djangorestframework_simplejwt django-cors-headers rank-bm25 numpy requests
python manage.py test
```

### Changing dependencies

`requirements.txt` lists direct dependencies; `constraints.txt` pins the full resolved set so Space
rebuilds are reproducible. After editing `requirements.txt`, regenerate it for the container's
platform:

```bash
pip install --dry-run --ignore-installed --only-binary=:all: \
  --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --platform linux_x86_64 \
  --python-version 3.11 --implementation cp --target /tmp/resolve \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html "torch==2.2.1+cpu" \
  -r requirements.txt --report report.json
# then write every package from report.json except torch as name==version into constraints.txt
```

## Deployment

The git remote is the Hugging Face Space, so **every push to `main` rebuilds and redeploys**.
Run the tests before pushing.
