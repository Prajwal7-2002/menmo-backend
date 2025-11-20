#!/bin/bash
set -e

# Wait for database to be available (optional robust startup)
# Uncomment the following for a simple wait loop if needed:
# until nc -z $DB_HOST $DB_PORT; do
#   echo "Waiting for Postgres at $DB_HOST:$DB_PORT..."
#   sleep 1
# done

echo "Applying database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django server..."
python manage.py runserver 0.0.0.0:7860
