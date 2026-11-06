"""Create the 'agent' database if it doesn't exist."""
import asyncio
import asyncpg
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def wait_for_postgres(max_retries=30, delay=2):
    """Wait for PostgreSQL to be ready."""
    for i in range(max_retries):
        try:
            conn = await asyncpg.connect(
                user="postgres",
                password="1234509876",
                host="localhost",
                port=5432,
                database="postgres",
                timeout=5,
            )
            await conn.close()
            print("[OK] PostgreSQL is ready")
            return True
        except Exception:
            print(f"[INFO] Waiting for PostgreSQL... ({i+1}/{max_retries})")
            await asyncio.sleep(delay)
    print("[WARN] PostgreSQL not ready, proceeding anyway")
    return False


async def create_database() -> None:
    # Wait for PostgreSQL to be ready
    await wait_for_postgres()

    # Connect to the default 'postgres' database to create our new one
    conn = await asyncpg.connect(
        user="postgres",
        password="1234509876",
        host="localhost",
        port=5432,
        database="postgres",
    )

    try:
        # Check if 'agent' exists
        result = await conn.fetch(
            "SELECT 1 FROM pg_database WHERE datname = $1", "agent"
        )
        if result:
            print("[INFO] Database 'agent' already exists")
        else:
            await conn.execute('CREATE DATABASE agent')
            print("[OK] Database 'agent' created successfully")

        # Install pgvector extension in the 'agent' database
        agent_conn = await asyncpg.connect(
            user="postgres",
            password="1234509876",
            host="localhost",
            port=5432,
            database="agent",
        )
        try:
            await agent_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("[OK] pgvector extension installed in 'agent'")
        finally:
            await agent_conn.close()
    finally:
        await conn.close()

    # Run Alembic migrations to create schema
    print("[SETUP] Running Alembic migrations...")
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str(ROOT / "alembic.ini"))
    alembic_cfg.set_main_option(
        "sqlalchemy.url",
        "postgresql+asyncpg://postgres:1234509876@localhost:5432/agent",
    )

    try:
        command.upgrade(alembic_cfg, "head")
        print("[OK] Migrations applied")
    except Exception as exc:
        print(f"[WARN] Migration warning: {exc}")


if __name__ == "__main__":
    asyncio.run(create_database())