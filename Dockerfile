FROM python:3.11-slim

WORKDIR /app

# Install PostgreSQL client + required build deps
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose Django port
EXPOSE 7860

RUN rm -f .env || true


# Use entrypoint to migrate + run server
CMD ["sh", "entrypoint.sh"]
