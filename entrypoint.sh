#!/bin/bash
set -e

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django..."
python manage.py runserver 0.0.0.0:7860
