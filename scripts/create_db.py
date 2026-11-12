"""Create the configured database, install pgvector, and run migrations."""
import asyncio
import sys
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402


async def connect_admin() -> asyncpg.Connection:
    last_error: Exception | None = None
    for attempt in range(1, settings.DB_CONNECT_RETRIES + 1):
        try:
            return await asyncpg.connect(
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_ADMIN_DB,
                timeout=5,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[INFO] Waiting for PostgreSQL "
                f"({attempt}/{settings.DB_CONNECT_RETRIES})"
            )
            await asyncio.sleep(settings.DB_CONNECT_DELAY)
    raise RuntimeError(f"PostgreSQL was not reachable: {last_error}")


async def connect_target() -> asyncpg.Connection:
    return await asyncpg.connect(
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
    )


async def create_database() -> None:
    conn = await connect_admin()
    try:
        exists = await conn.fetch(
            "SELECT 1 FROM pg_database WHERE datname = $1", settings.DB_NAME
        )
        if exists:
            print(f"[INFO] Database '{settings.DB_NAME}' already exists")
        else:
            await conn.execute(f'CREATE DATABASE "{settings.DB_NAME}"')
            print(f"[OK] Database '{settings.DB_NAME}' created")
    finally:
        await conn.close()

    target = await connect_target()
    try:
        await target.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print(f"[OK] pgvector installed in '{settings.DB_NAME}'")
    finally:
        await target.close()

    from alembic.config import Config

    from app.db.migrations import upgrade_database

    alembic_cfg = Config(str(ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
    upgrade_database(alembic_cfg)
    print("[OK] Alembic migrations applied")


if __name__ == "__main__":
    asyncio.run(create_database())
