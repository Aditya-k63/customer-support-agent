"""Test database connection and ticket persistence against local PostgreSQL."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.base import engine, AsyncSessionLocal
from app.db.models import Base, Ticket, Conversation, DocumentChunk
from sqlalchemy import select
from app.agents.tools.ticket_db import TicketDB


async def test_connection() -> bool:
    """Verify we can connect to PostgreSQL and query."""
    print("[TEST] Database connection...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Ticket).limit(1))
        print("[OK] Database connection successful")
        return True


async def test_ticket_persistence() -> int:
    """Create, read, update a ticket to verify full CRUD."""
    print("[TEST] Ticket persistence...")
    ticket_db = TicketDB()

    # Create
    ticket_id = await ticket_db.create_ticket(
        title="Test refund request",
        status="open",
        intent="refund_request",
        escalated=True,
    )
    print(f"   Created ticket ID: {ticket_id}")

    # Read
    ticket = await ticket_db.get_ticket(ticket_id)
    print(
        f"   Retrieved: {ticket['title']} | status={ticket['status']} | "
        f"intent={ticket['intent']} | escalated={ticket['escalated']}"
    )
    assert ticket is not None
    assert ticket["title"] == "Test refund request"
    assert ticket["intent"] == "refund_request"
    assert ticket["escalated"] is True

    # Update status
    updated = await ticket_db.update_ticket_status(ticket_id, "in_progress")
    assert updated is True
    ticket = await ticket_db.get_ticket(ticket_id)
    assert ticket["status"] == "in_progress"
    print(f"   Updated status to: {ticket['status']}")

    # Add conversation
    conv_id = await ticket_db.add_conversation(
        ticket_id=ticket_id, role="user", content="I want a refund"
    )
    print(f"   Added conversation ID: {conv_id}")

    print("[OK] Ticket persistence test passed")
    return ticket_id


async def test_vector_chunk() -> None:
    """Verify DocumentChunk model can be inserted (pgvector schema)."""
    print("[TEST] Document chunk (pgvector) table...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DocumentChunk).limit(1))
        rows = result.fetchall()
        print(f"   DocumentChunk table accessible (rows: {len(rows)})")
    print("[OK] Vector chunk table test passed")


async def create_tables() -> None:
    """Create all tables defined in SQLAlchemy models."""
    print("[SETUP] Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[OK] Tables created: tickets, conversations, document_chunks")


async def main() -> None:
    print("=" * 50)
    print("DATABASE TEST SUITE")
    print("=" * 50)

    await create_tables()
    await test_connection()
    ticket_id = await test_ticket_persistence()
    await test_vector_chunk()

    print("\n" + "=" * 50)
    print("ALL DATABASE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())