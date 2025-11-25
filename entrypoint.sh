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

# Test DB connectivity (optional but useful)
if [ -n "$DB_HOST" ] && [ -n "$DB_NAME" ]; then
  echo "Trying DB connection..."
  if command -v psql >/dev/null; then
    PGPASSWORD="$DB_PASSWORD" psql \
      --host="$DB_HOST" \
      --username="$DB_USER" \
      --dbname="$DB_NAME" \
      -c "SELECT NOW();" || echo "❌ MANUAL DB CONNECT FAILED INSIDE CONTAINER"
  else
    echo "⚠️ psql not installed (skipping manual DB test)"
  fi
fi

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput || true

echo "Starting Gunicorn..."
gunicorn copilot_backend.wsgi:application \
  --chdir /app \
  --bind 0.0.0.0:7860 \
  --workers ${GUNICORN_WORKERS:-2} \
  --threads ${GUNICORN_THREADS:-4} \
  --timeout ${GUNICORN_TIMEOUT:-120}
