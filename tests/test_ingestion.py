"""Ingestion chunking behaviour."""
import pytest

from ingestion.chunker import chunk_documents, chunk_text


def test_short_document_is_a_single_chunk():
    assert chunk_text("Refunds within 30 days.") == ["Refunds within 30 days."]


def test_long_document_is_split_with_overlap():
    words = [f"word{i}" for i in range(250)]

    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # The second chunk starts before the first one ends, so the overlap is
    # carried across the boundary.
    assert "word80" in chunks[0]
    assert "word80" in chunks[1]


def test_chunking_covers_every_word():
    words = [f"word{i}" for i in range(250)]

    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)

    recovered = " ".join(chunks).split()
    for word in words:
        assert word in recovered


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_non_positive_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0)


def test_chunk_records_use_stable_ids():
    documents = [
        {
            "id": "refund_policy",
            "content": " ".join(f"word{i}" for i in range(250)),
            "metadata": {"category": "billing"},
        }
    ]

    chunked = chunk_documents(documents)

    assert chunked[0]["doc_id"] == "refund_policy#chunk-0"
    assert chunked[1]["doc_id"] == "refund_policy#chunk-1"
    assert all(record["source_doc_id"] == "refund_policy" for record in chunked)
    assert chunked[0]["metadata"]["source_doc_id"] == "refund_policy"
    assert chunked[0]["metadata"]["category"] == "billing"


def test_empty_document_produces_no_chunks():
    assert chunk_text("   ") == []
