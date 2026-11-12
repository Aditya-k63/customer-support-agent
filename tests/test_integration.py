"""Integration tests against real PostgreSQL + pgvector.

Requires a running PostgreSQL instance with the pgvector extension and the
schema migrated. Run directly:

    python scripts/create_db.py
    python tests/test_integration.py
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select, text

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 with PostgreSQL running",
    ),
]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.tools.ticket_db import TicketDB  # noqa: E402
from app.agents.tools.vector_store import VectorStore  # noqa: E402
from app.db.base import AsyncSessionLocal, engine  # noqa: E402
from app.db.models import Base, DocumentChunk  # noqa: E402


async def test_ticket_crud_and_conversation_history():
    """Full ticket lifecycle with session-scoped conversation memory."""
    ticket_db = TicketDB()
    session_id = f"integration-session-{uuid.uuid4().hex[:8]}"

    ticket_id = await ticket_db.get_or_create_ticket(
        session_id=session_id,
        title="Integration test ticket",
        status="in_progress",
        intent="billing",
        escalated=True,
    )
    assert isinstance(ticket_id, int)

    ticket = await ticket_db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["status"] == "in_progress"
    assert ticket["session_id"] == session_id

    # A second turn updates the same ticket with its final status.
    ticket_id_again = await ticket_db.get_or_create_ticket(
        session_id=session_id,
        title="Integration test ticket",
        status="resolved",
        intent="billing",
        escalated=False,
    )
    assert ticket_id_again == ticket_id
    assert (await ticket_db.get_ticket(ticket_id))["status"] == "resolved"

    await ticket_db.add_conversation(ticket_id, "user", "Test message")
    await ticket_db.add_conversation(ticket_id, "assistant", "Test response")

    history = await ticket_db.get_conversation_history(session_id)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "Test message"

    print(f"[OK] Ticket CRUD + conversation memory passed (ticket_id={ticket_id})")


async def test_vector_store_insert_and_search():
    """pgvector insertion, deduplication and cosine similarity search."""
    vector_store = VectorStore()
    source_ids = ["test-password", "test-refund", "test-hours"]
    docs = [
        {
            "doc_id": "test-password#chunk-0",
            "source_doc_id": "test-password",
            "chunk_index": 0,
            "content": "How to reset password on the website",
            "metadata": {"category": "technical", "source": "test"},
        },
        {
            "doc_id": "test-refund#chunk-0",
            "source_doc_id": "test-refund",
            "chunk_index": 0,
            "content": "Refund policy for defective items",
            "metadata": {"category": "billing", "source": "test"},
        },
        {
            "doc_id": "test-hours#chunk-0",
            "source_doc_id": "test-hours",
            "chunk_index": 0,
            "content": "Business hours are Monday to Friday 9am to 5pm",
            "metadata": {"category": "general", "source": "test"},
        },
    ]

    async def remove_test_documents() -> None:
        for source_id in source_ids:
            await vector_store.delete_by_source_doc_id(source_id)

    # Start from a clean slate so reruns are deterministic, and always clean up
    # so evaluation runs never see synthetic test documents.
    await remove_test_documents()
    try:
        inserted = await vector_store.add_documents(docs)
        assert inserted == 3

        # Re-inserting the same content must not create duplicates.
        assert await vector_store.add_documents(docs) == 0

        results = await vector_store.search_documents(
            "how do I reset my password", top_k=3, threshold=0.1
        )
        assert len(results) > 0
        assert all(result.doc_id for result in results)

        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.source_doc_id.like("test-%")
                )
            )
            assert len(rows.scalars().all()) == 3
    finally:
        await remove_test_documents()

    print(f"[OK] Vector store search + dedupe passed ({len(results)} results)")


async def test_pgvector_extension():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        assert result.fetchone() is not None, "pgvector extension not installed"
    print("[OK] pgvector extension test passed")


async def main() -> None:
    print("=" * 50)
    print("INTEGRATION TEST SUITE")
    print("=" * 50)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await test_ticket_crud_and_conversation_history()
    await test_pgvector_extension()
    await test_vector_store_insert_and_search()

    print("=" * 50)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
