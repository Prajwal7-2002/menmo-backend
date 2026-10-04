FROM python:3.11-slim

# Prefer IPv4 (DNS issues on Hugging Face Spaces)
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

# poppler + tesseract for scanned-PDF OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; give that user ownership of /app so
# the model cache and collected static files are writable.
RUN useradd -m -u 1000 user
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache \
    HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

COPY requirements.txt constraints.txt ./

# CPU-only torch first so nothing pulls the multi-GB CUDA build
RUN pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install -r requirements.txt -c constraints.txt

# Bake the embedding model into the image (no download on the first request)
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['HF_EMBED_MODEL'])"

COPY . .
RUN chown -R user:user /app

USER user
ENV HOME=/home/user

EXPOSE 7860
ENTRYPOINT ["sh", "entrypoint.sh"]
