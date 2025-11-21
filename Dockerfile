FROM python:3.11-slim

# Force IPv4 for DNS
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

WORKDIR /app

# Install build deps + postgres client
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Remove .env inside container
RUN find /app -name ".env" -delete || true

EXPOSE 7860

# ENTRYPOINT runs the script that starts Django
ENTRYPOINT ["sh", "entrypoint.sh"]

# CMD is NOT needed (entrypoint already starts Django)
