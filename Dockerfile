FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for numpy, faiss, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Expose HuggingFace port
EXPOSE 7860

# Start server
CMD ["bash", "/app/entrypoint.sh"]
