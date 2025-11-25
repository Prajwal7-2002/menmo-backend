# Dockerfile
FROM python:3.11-slim

# Force IPv4 for DNS (helps some cloud environments)
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# --- HuggingFace persistent cache inside the image (not external storage) ---
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface

RUN mkdir -p /app/.cache/huggingface

# Install build deps
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python deps (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# === Pre-download models at build time to avoid runtime download ===
# This will bake the sentence-transformers model into the image.
RUN python3 - <<'PY'
from sentence_transformers import SentenceTransformer
# Ensure this model matches EMBED_MODEL_NAME in retrieval.py
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# If you want to pre-download reranker (large), uncomment:
# from sentence_transformers import CrossEncoder
# CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
print("✅ Pre-download complete")
PY

# Copy source after installing deps + caches populated
COPY . .

# Remove any .env inside container for safety
RUN find /app -name ".env" -delete || true

# Expose application port (match runserver/gunicorn)
EXPOSE 7860

# Entrypoint script will start the app with gunicorn
ENTRYPOINT ["sh", "entrypoint.sh"]
