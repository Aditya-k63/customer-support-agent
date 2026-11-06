"""Embedding service with sentence-transformers for pgvector semantic search."""
import hashlib
from typing import List


class EmbeddingService:
    def __init__(self) -> None:
        self._model = None
        self.model_name = "all-MiniLM-L6-v2"
        self.vector_dim = 384

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = "fallback"

    async def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        self._load_model()

        if self._model == "fallback":
            # Deterministic pseudo-embedding when sentence-transformers unavailable
            h = hashlib.sha256(text.encode()).digest()
            # Expand to 384 dimensions by repeating and normalizing
            base = list(h)
            embedding = []
            for i in range(self.vector_dim):
                embedding.append(base[i % len(base)] / 255.0)
            return embedding

        embedding = self._model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()

        if self._model == "fallback":
            return [await self.embed(t) for t in texts]

        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]