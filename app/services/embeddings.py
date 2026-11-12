"""Semantic embedding service backed by sentence-transformers.

There is intentionally no pseudo-embedding fallback. If the model cannot be
loaded the service raises ``EmbeddingModelUnavailableError`` so callers fail
loudly instead of returning meaningless vectors from hash bytes.
"""
import asyncio
from typing import List

from app.core.config import settings


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the configured embedding model cannot be loaded."""


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self._model = None
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.vector_dim = settings.EMBEDDING_DIM

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingModelUnavailableError(
                    "sentence-transformers is not installed. Install "
                    "requirements-embeddings.txt before using semantic retrieval."
                ) from exc

            try:
                model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingModelUnavailableError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc

            dimension = model.get_sentence_embedding_dimension()
            if dimension != self.vector_dim:
                raise EmbeddingModelUnavailableError(
                    f"Embedding model '{self.model_name}' produces {dimension} "
                    f"dimensions but the database column expects {self.vector_dim}."
                )
            self._model = model
        return self._model

    async def embed(self, text: str) -> List[float]:
        """Generate a semantic embedding vector for a single text."""
        model = await asyncio.to_thread(self._load_model)
        embedding = await asyncio.to_thread(model.encode, text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate semantic embeddings for multiple texts."""
        model = await asyncio.to_thread(self._load_model)
        embeddings = await asyncio.to_thread(model.encode, texts)
        return [e.tolist() for e in embeddings]
