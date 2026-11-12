"""Vector store with real pgvector cosine similarity search.

Each chunk carries a stable ``doc_id``, its ``source_doc_id``, chunk index,
metadata and the embedding model that produced the vector. Retrieval returns
these identifiers so evaluation can compare retrieved documents against
labelled ground truth.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sqlalchemy import delete, select

from app.core.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import DocumentChunk
from app.services.embeddings import EmbeddingService


@dataclass
class RetrievedChunk:
    doc_id: str
    content: str
    similarity: float
    metadata: Dict = field(default_factory=dict)


class VectorStore:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()

    async def search_documents(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> List[RetrievedChunk]:
        """Semantic search returning stable document identifiers."""
        top_k = top_k or settings.RETRIEVAL_TOP_K
        threshold = settings.RETRIEVAL_THRESHOLD if threshold is None else threshold

        query_embedding = await self.embedding_service.embed(query)
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(
                DocumentChunk.doc_id,
                DocumentChunk.content,
                DocumentChunk.doc_metadata,
                distance.label("distance"),
            )
            .where(distance <= (1 - threshold))
            .order_by("distance")
            .limit(top_k)
        )
        async with AsyncSessionLocal() as session:
            result = await session.execute(stmt)
            return [
                RetrievedChunk(
                    doc_id=row[0] or "",
                    content=row[1],
                    similarity=round(max(0.0, min(1.0, 1.0 - float(row[3]))), 4),
                    metadata=row[2] or {},
                )
                for row in result.fetchall()
            ]

    async def search_with_scores(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Backward-compatible search returning ``(content, cosine_distance)``."""
        chunks = await self.search_documents(query, top_k=top_k, threshold=threshold)
        return [(chunk.content, round(1.0 - chunk.similarity, 4)) for chunk in chunks]

    async def search(self, query: str, top_k: int = 3) -> List[str]:
        """Backward-compatible search returning just content strings."""
        chunks = await self.search_documents(query, top_k=top_k)
        return [chunk.content for chunk in chunks]

    async def add_documents(self, documents: List[dict]) -> int:
        """Ingest chunked documents, skipping duplicate content.

        Returns the number of newly inserted chunks.
        """
        prepared: List[dict] = []
        for doc in documents:
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            prepared.append(
                {
                    "doc_id": doc.get("doc_id") or f"doc-{content_hash[:16]}",
                    "source_doc_id": doc.get("source_doc_id")
                    or doc.get("doc_id")
                    or f"doc-{content_hash[:16]}",
                    "chunk_index": int(doc.get("chunk_index", 0)),
                    "content": content,
                    "content_hash": content_hash,
                    "metadata": doc.get("metadata", {}),
                }
            )

        if not prepared:
            return 0

        hashes = [doc["content_hash"] for doc in prepared]
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(DocumentChunk.content_hash).where(
                    DocumentChunk.content_hash.in_(hashes)
                )
            )
            existing_hashes = set(existing.scalars().all())
            new_docs = [d for d in prepared if d["content_hash"] not in existing_hashes]

            if not new_docs:
                return 0

            embeddings = await self.embedding_service.embed_batch(
                [d["content"] for d in new_docs]
            )
            for doc, embedding in zip(new_docs, embeddings):
                session.add(
                    DocumentChunk(
                        doc_id=doc["doc_id"],
                        source_doc_id=doc["source_doc_id"],
                        chunk_index=doc["chunk_index"],
                        content=doc["content"],
                        content_hash=doc["content_hash"],
                        doc_metadata=doc["metadata"],
                        embedding_model=self.embedding_service.model_name,
                        embedding=embedding,
                    )
                )
            await session.commit()
            return len(new_docs)

    async def delete_by_source_doc_id(self, source_doc_id: str) -> None:
        """Remove all chunks belonging to a source document."""
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.source_doc_id == source_doc_id
                )
            )
            await session.commit()

    async def replace_source_documents(
        self, source_doc_id: str, documents: List[dict]
    ) -> int:
        """Re-ingest a source document, removing its previous chunks first."""
        await self.delete_by_source_doc_id(source_doc_id)
        return await self.add_documents(documents)
