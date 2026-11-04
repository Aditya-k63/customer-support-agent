#!/bin/bash
set -e

echo "Running database migrations..."
# alembic upgrade head  # Uncomment when using Alembic

echo "Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000