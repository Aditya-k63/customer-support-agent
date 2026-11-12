"""Hardening: session-scoped tickets, document IDs, dedupe, and vector index.

Revision ID: 002_hardening
Revises: 001_initial_schema
Create Date: 2026-11-12 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_hardening"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Session-scoped ticket lookup
    op.add_column("tickets", sa.Column("session_id", sa.String(length=255), nullable=True))
    op.create_index(op.f("ix_tickets_session_id"), "tickets", ["session_id"], unique=False)

    # Stable document/chunk identity + embedding model tracking
    op.add_column("document_chunks", sa.Column("doc_id", sa.String(length=255), nullable=True))
    op.add_column(
        "document_chunks", sa.Column("source_doc_id", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "document_chunks",
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "document_chunks", sa.Column("content_hash", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "document_chunks", sa.Column("embedding_model", sa.String(length=255), nullable=True)
    )

    # Backfill pre-existing rows so the unique indexes can be created safely.
    op.execute(
        "UPDATE document_chunks SET "
        "doc_id = 'legacy-' || id, "
        "source_doc_id = 'legacy-' || id, "
        "content_hash = md5(content), "
        "embedding_model = 'unknown' "
        "WHERE doc_id IS NULL"
    )

    op.create_index(
        op.f("ix_document_chunks_doc_id"), "document_chunks", ["doc_id"], unique=True
    )
    op.create_index(
        op.f("ix_document_chunks_source_doc_id"),
        "document_chunks",
        ["source_doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_content_hash"),
        "document_chunks",
        ["content_hash"],
        unique=True,
    )

    # Approximate nearest-neighbour index for cosine similarity search.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.drop_index(op.f("ix_document_chunks_content_hash"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_source_doc_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_doc_id"), table_name="document_chunks")
    op.drop_column("document_chunks", "embedding_model")
    op.drop_column("document_chunks", "content_hash")
    op.drop_column("document_chunks", "chunk_index")
    op.drop_column("document_chunks", "source_doc_id")
    op.drop_column("document_chunks", "doc_id")
    op.drop_index(op.f("ix_tickets_session_id"), table_name="tickets")
    op.drop_column("tickets", "session_id")
