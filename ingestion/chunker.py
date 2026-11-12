"""Word-based document chunking with configurable size and overlap."""
from app.core.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping word windows.

    ``chunk_size`` and ``overlap`` are measured in words. A document shorter
    than ``chunk_size`` is returned unchanged.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Expand source documents into stable, chunk-indexed records."""
    chunked: list[dict] = []
    for doc in documents:
        source_id = doc["id"]
        metadata = doc.get("metadata", {})
        for index, content in enumerate(chunk_text(doc["content"])):
            chunked.append(
                {
                    "doc_id": f"{source_id}#chunk-{index}",
                    "source_doc_id": source_id,
                    "chunk_index": index,
                    "content": content,
                    "metadata": {**metadata, "source_doc_id": source_id},
                }
            )
    return chunked
