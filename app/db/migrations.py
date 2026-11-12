"""Shared migration helper with safe adoption of legacy schemas."""
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)


def _stamp_legacy_schema(config: Config) -> None:
    """Adopt a database that predates Alembic version tracking.

    Older deployments created tables with ``metadata.create_all`` and have no
    ``alembic_version`` row. Stamping the matching revision lets subsequent
    upgrades apply only the new changes instead of recreating existing tables.
    """
    url = _sync_url(config.get_main_option("sqlalchemy.url") or "")
    engine = create_engine(url, pool_pre_ping=True)
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if "tickets" not in tables or "alembic_version" in tables:
            return
        revision = "001_initial_schema"
        if "document_chunks" in tables:
            columns = {
                column["name"]
                for column in inspector.get_columns("document_chunks")
            }
            if "doc_id" in columns:
                revision = "002_hardening"
        command.stamp(config, revision)
        print(f"[INFO] Stamped legacy schema at revision {revision}")
    finally:
        engine.dispose()


def upgrade_database(config: Config) -> None:
    """Stamp legacy schemas when needed, then run all pending migrations."""
    _stamp_legacy_schema(config)
    command.upgrade(config, "head")
