# ----------------------
# Base Image
# ----------------------
FROM python:3.11-slim

# Fix DNS for Hugging Face Spaces
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# ----------------------
# HuggingFace Cache
# ----------------------
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache
ENV HF_HUB_CACHE=/app/hf-cache

RUN mkdir -p /app/hf-cache

# ----------------------
# System Dependencies
# ----------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ----------------------
# Python Dependencies
# ----------------------
COPY requirements.txt .

# Install CPU-only torch FIRST to stop CUDA downloads
RUN pip install --no-cache-dir torch==2.2.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# ----------------------
# Project Files
# ----------------------
COPY . .

# ----------------------
# Runtime model selection
# ----------------------
ENV HF_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Remove any accidental secrets
RUN find /app -name ".env" -delete || true

# ----------------------
# Expose & Start
# ----------------------
EXPOSE 7860
ENTRYPOINT ["sh", "entrypoint.sh"]
