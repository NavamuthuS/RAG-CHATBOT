"""
tests/test_rag.py — Smoke Tests for RAG Chatbot

Run with:
    pytest tests/test_rag.py -v

Tests covered:
1. split_documents() creates more than one chunk from a long document.
2. Each chunk is a valid LangChain Document with page_content.
3. Chunk size does not exceed CHUNK_SIZE limit.
4. split_documents() raises ValueError on empty input.
"""

import sys
import os

# Make sure the project root is on the path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from langchain_core.documents import Document

from src.text_splitter import split_documents
from config import CHUNK_SIZE


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def long_document():
    """
    Creates a single dummy LangChain Document with enough text
    to guarantee it will be split into multiple chunks.
    """
    # Generate ~5000 characters of dummy text (well above CHUNK_SIZE=1000)
    dummy_text = (
        "This is a test sentence about artificial intelligence. " * 100
    )
    return Document(
        page_content=dummy_text,
        metadata={"source": "test_file.txt", "page": 0},
    )


@pytest.fixture
def short_document():
    """
    Creates a short Document that fits within a single chunk.
    """
    return Document(
        page_content="Short content.",
        metadata={"source": "short_file.txt", "page": 0},
    )


# ── Tests ─────────────────────────────────────────────────────────────

def test_split_creates_multiple_chunks(long_document):
    """
    A document longer than CHUNK_SIZE must produce more than one chunk.
    """
    chunks = split_documents([long_document])
    assert len(chunks) > 1, (
        f"Expected more than 1 chunk for a long document, got {len(chunks)}"
    )


def test_chunks_are_documents(long_document):
    """
    Every item in the returned list must be a LangChain Document
    with non-empty page_content.
    """
    chunks = split_documents([long_document])
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, Document), (
            f"Chunk {i} is not a Document object: {type(chunk)}"
        )
        assert chunk.page_content.strip() != "", (
            f"Chunk {i} has empty page_content"
        )


def test_chunk_size_within_limit(long_document):
    """
    No chunk should exceed CHUNK_SIZE characters.
    (A small tolerance is allowed because the splitter may slightly
    overshoot on the last word boundary.)
    """
    chunks = split_documents([long_document])
    for i, chunk in enumerate(chunks):
        assert len(chunk.page_content) <= CHUNK_SIZE + 50, (
            f"Chunk {i} exceeds CHUNK_SIZE: {len(chunk.page_content)} chars"
        )


def test_metadata_preserved(long_document):
    """
    Each chunk must carry the metadata from the original document.
    """
    chunks = split_documents([long_document])
    for i, chunk in enumerate(chunks):
        assert "source" in chunk.metadata, (
            f"Chunk {i} is missing 'source' in metadata"
        )


def test_short_document_gives_one_chunk(short_document):
    """
    A document shorter than CHUNK_SIZE should produce exactly 1 chunk.
    """
    chunks = split_documents([short_document])
    assert len(chunks) == 1, (
        f"Expected 1 chunk for short document, got {len(chunks)}"
    )


def test_empty_input_raises_value_error():
    """
    Passing an empty list to split_documents() must raise ValueError.
    """
    with pytest.raises(ValueError, match="No documents provided"):
        split_documents([])


def test_multiple_documents_combined(long_document, short_document):
    """
    Splitting multiple documents together should return
    chunks from all of them combined.
    """
    chunks = split_documents([long_document, short_document])
    assert len(chunks) >= 2, (
        "Expected at least 2 chunks when splitting multiple documents"
    )