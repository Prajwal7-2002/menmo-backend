# Dockerfile
FROM python:3.11-slim

RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# Local persistent cache inside image
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
ENV HF_HUB_CACHE=/app/hf-cache

RUN mkdir -p /app/hf-cache
RUN mkdir -p /app/models

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === PRE-DOWNLOAD AND SAVE MODEL INTO THE IMAGE ===
RUN python3 - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("/app/models/minilm")
print("✅ MiniLM saved to /app/models/minilm")
EOF

# If you want reranker baked in (optional, heavy):
# RUN python3 - <<'EOF'
# from sentence_transformers import CrossEncoder
# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
# model.save_pretrained("/app/models/reranker")
# print("✅ Reranker saved")
# EOF

# Copy remaining source code
COPY . .

# FORCE HF_EMBED_MODEL to use local path → prevents all downloads
ENV HF_EMBED_MODEL=/app/models/minilm

# Remove .env for security
RUN find /app -name ".env" -delete || true

EXPOSE 7860

ENTRYPOINT ["sh", "entrypoint.sh"]
