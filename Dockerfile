# Dockerfile
FROM python:3.11-slim

# Prefer IPv4 for DNS
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# -----------------------------
# HuggingFace offline + cache
# -----------------------------
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
ENV HF_HUB_CACHE=/app/hf-cache

# FORCE OFFLINE (prevents ANY downloads)
ENV HF_HUB_OFFLINE=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf-cache

RUN mkdir -p /app/hf-cache
RUN mkdir -p /app/models

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Pre-download MiniLM model and bake into image
# -----------------------------
RUN python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("/app/models/minilm")
print("✅ MiniLM model saved to /app/models/minilm")
EOF

# If you ever want to bake reranker:
# RUN python3 - <<'EOF'
# from sentence_transformers import CrossEncoder
# m = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
# m.save_pretrained("/app/models/reranker")
# print("Reranker saved!")
# EOF

# -----------------------------
# Copy project source
# -----------------------------
COPY . .

# Force retrieval.py to use local model folder
ENV HF_EMBED_MODEL=/app/models/minilm

# Remove .env to avoid leaking local credentials
RUN find /app -name ".env" -delete || true

EXPOSE 7860

ENTRYPOINT ["sh", "entrypoint.sh"]
