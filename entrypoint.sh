#!/bin/bash
set -e

echo "===== DEBUG: PRINTING ALL ENV VARIABLES BEFORE START ====="
echo "DB_USER=$DB_USER"
echo "DB_PASSWORD=$DB_PASSWORD"
echo "DB_HOST=$DB_HOST"
echo "DB_NAME=$DB_NAME"
echo "DB_PORT=$DB_PORT"
echo "SECRET_KEY=$SECRET_KEY"
echo "GROQ_API_KEY=$GROQ_API_KEY"

echo "==========================================================="

echo "====== Testing manual DB login inside container ======"
PGPASSWORD="$DB_PASSWORD" psql \
  --host="$DB_HOST" \
  --username="$DB_USER" \
  --dbname="$DB_NAME" \
  -c "SELECT NOW();" || echo "❌ MANUAL DB CONNECT FAILED INSIDE CONTAINER"
echo "======================================================"



echo "Applying migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput || true

echo "Starting Django..."
python manage.py runserver 0.0.0.0:7860
