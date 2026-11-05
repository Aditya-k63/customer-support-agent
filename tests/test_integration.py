"""Integration tests against real PostgreSQL + pgvector.

These tests require a running PostgreSQL instance with pgvector extension.
They verify actual database persistence, vector search, and end-to-end flow.
"""
import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.base import engine, AsyncSessionLocal  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.agents.tools.ticket_db import TicketDB  # noqa: E402
from app.agents.tools.vector_store import VectorStore  # noqa: E402


async def test_ticket_crud_persistence():
    """Full CRUD cycle against real PostgreSQL."""
    ticket_db = TicketDB()

    # Create
    ticket_id = await ticket_db.create_ticket(
        title="Integration test ticket",
        status="open",
        intent="billing",
        escalated=False,
    )
    assert ticket_id is not None
    assert isinstance(ticket_id, int)

    # Read
    ticket = await ticket_db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["title"] == "Integration test ticket"
    assert ticket["status"] == "open"
    assert ticket["intent"] == "billing"
    assert ticket["escalated"] is False

    # Update status
    updated = await ticket_db.update_ticket_status(ticket_id, "resolved")
    assert updated is True

    ticket = await ticket_db.get_ticket(ticket_id)
    assert ticket["status"] == "resolved"

    # Add conversation
    conv_id = await ticket_db.add_conversation(
        ticket_id=ticket_id, role="user", content="Test message"
    )
    assert conv_id is not None

    conv_id2 = await ticket_db.add_conversation(
        ticket_id=ticket_id, role="assistant", content="Test response"
    )
    assert conv_id2 is not None

    print(f"[OK] Ticket CRUD persistence test passed (ticket_id={ticket_id})")


async def test_vector_store_insert_and_search():
    """Test pgvector embedding insertion and cosine similarity search."""
    vector_store = VectorStore()

    # Insert test documents
    docs = [
        {
            "content": "How to reset password on the website",
            "metadata": {"category": "technical", "source": "test"},
        },
        {
            "content": "Refund policy for defective items",
            "metadata": {"category": "billing", "source": "test"},
        },
        {
            "content": "Business hours are Monday to Friday 9am to 5pm",
            "metadata": {"category": "general", "source": "test"},
        },
    ]
    await vector_store.add_documents(docs)

    # Search for password-related query
    results = await vector_store.search_with_scores(
        "how do I reset my password", top_k=3, threshold=0.1
    )
    assert len(results) > 0, "Should return at least one result"
    print(f"[OK] Vector store search test passed ({len(results)} results)")


async def test_pgvector_extension():
    """Verify pgvector extension is installed in the database."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.fetchone()
        assert row is not None, "pgvector extension not installed"
    print("[OK] pgvector extension test passed")


async def main() -> None:
    print("=" * 50)
    print("INTEGRATION TEST SUITE")
    print("=" * 50)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await test_ticket_crud_persistence()
    await test_pgvector_extension()
    await test_vector_store_insert_and_search()

    print("=" * 50)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())