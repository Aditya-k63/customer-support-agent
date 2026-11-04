"""Create the 'agent' database if it doesn't exist."""
import asyncio
import asyncpg
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def create_database() -> None:
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


if __name__ == "__main__":
    asyncio.run(create_database())