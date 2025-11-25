FROM python:3.11-slim

RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# --- HuggingFace persistent cache ---
ENV HF_HOME=/data/huggingface
ENV TRANSFORMERS_CACHE=/data/huggingface
ENV HF_HUB_CACHE=/data/huggingface

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN find /app -name ".env" -delete || true

EXPOSE 7860
ENTRYPOINT ["sh", "entrypoint.sh"]
