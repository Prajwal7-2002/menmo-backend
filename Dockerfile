
FROM python:3.11-slim
RUN echo "BUILD-TIME DB_USER=$DB_USER"

WORKDIR /app

# Install PostgreSQL client + required build deps
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Delete ANY .env file inside project
RUN find /app -name ".env" -delete || true

EXPOSE 7860

CMD ["sh", "entrypoint.sh"]
