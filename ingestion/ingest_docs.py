"""Ingest the knowledge base into pgvector with chunking and deduplication."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.tools.vector_store import VectorStore  # noqa: E402
from ingestion.chunker import chunk_documents  # noqa: E402
from ingestion.knowledge_base import KNOWLEDGE_BASE  # noqa: E402


async def ingest_knowledge_base() -> int:
    """Re-ingest every knowledge-base document and return chunk count."""
    store = VectorStore()
    total = 0
    for document in KNOWLEDGE_BASE:
        chunks = chunk_documents([document])
        inserted = await store.replace_source_documents(document["id"], chunks)
        print(f"[OK] {document['id']}: {inserted} chunk(s)")
        total += inserted
    return total


def main() -> None:
    total = asyncio.run(ingest_knowledge_base())
    print(f"Indexed {total} chunks into pgvector with semantic embeddings.")


if __name__ == "__main__":
    main()
