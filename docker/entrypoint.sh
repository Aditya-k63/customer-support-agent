#!/bin/bash
set -e

echo "Applying database migrations..."
python - <<'PY'
from alembic.config import Config

from app.core.config import settings
from app.db.migrations import upgrade_database

config = Config("alembic.ini")
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
upgrade_database(config)
PY

# Migrations already ran above; avoid running them twice on startup.
export RUN_MIGRATIONS_ON_STARTUP=false

echo "Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
