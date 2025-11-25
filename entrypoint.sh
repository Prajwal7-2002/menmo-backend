#!/bin/bash
set -e

echo "===== DEBUG: PRINTING SELECT ENV VARIABLES BEFORE START ====="
echo "DB_HOST=$DB_HOST"
echo "DB_NAME=$DB_NAME"
echo "PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME"
echo "USE_RERANKER=$USE_RERANKER"
echo "USE_QUERY_REWRITE=$USE_QUERY_REWRITE"
echo "HF_HOME=$HF_HOME"
echo "HF_EMBED_MODEL=$HF_EMBED_MODEL"
echo "==========================================================="

# Test DB connectivity if env provided (best-effort)
if [ -n "$DB_HOST" ] && [ -n "$DB_NAME" ]; then
  echo "Trying DB connection..."
  PGPASSWORD="$DB_PASSWORD" psql \
    --host="$DB_HOST" \
    --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -c "SELECT 1;" || echo "❌ MANUAL DB CONNECT FAILED INSIDE CONTAINER"
fi

# Run migrations and collectstatic
python manage.py migrate --noinput
python manage.py collectstatic --noinput || true

echo "Starting Gunicorn..."
# Use a simple worker count; tune for your Space hardware
gunicorn backend.wsgi:application \
  --bind 0.0.0.0:7860 \
  --workers ${GUNICORN_WORKERS:-2} \
  --threads ${GUNICORN_THREADS:-4} \
  --timeout ${GUNICORN_TIMEOUT:-120}
