"""Vector store with real pgvector cosine similarity search."""
from typing import List, Tuple
from app.db.base import AsyncSessionLocal
from app.db.models import DocumentChunk
from sqlalchemy import select
from app.services.embeddings import EmbeddingService


class VectorStore:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()

    async def search_with_scores(
        self, query: str, top_k: int = 5, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Semantic search using cosine distance via pgvector.
        Returns [(content, cosine_distance)] sorted by relevance.
        """
        query_embedding = await self.embedding_service.embed(query)
        async with AsyncSessionLocal() as session:
            stmt = (
                select(
                    DocumentChunk.content,
                    DocumentChunk.embedding.cosine_distance(query_embedding).label(
                        "distance"
                    ),
                )
                .where(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                    < (1 - threshold)
                )
                .order_by("distance")
                .limit(top_k)
            )
            result = await session.execute(stmt)
            return [(row[0], float(row[1])) for row in result.fetchall()]

    async def add_documents(self, documents: List[dict]) -> None:
        """Ingest documents with embeddings into pgvector."""
        async with AsyncSessionLocal() as session:
            for doc in documents:
                embedding = await self.embedding_service.embed(doc["content"])
                chunk = DocumentChunk(
                    content=doc["content"],
                    doc_metadata=doc.get("metadata", {}),
                    embedding=embedding,
                )
                session.add(chunk)
            await session.commit()

    async def search(self, query: str, top_k: int = 3) -> List[str]:
        """Backward-compatible search returning just content strings."""
        results = await self.search_with_scores(query, top_k=top_k)
        return [content for content, _ in results]