"""
src/text_splitter.py — Text Splitting Module

Takes a list of LangChain Document objects and splits them into
smaller overlapping chunks using RecursiveCharacterTextSplitter.
Chunk size and overlap are pulled from config.py.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import settings from central config — no hardcoded values here
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on natural boundaries
    (paragraphs → sentences → words → characters) to keep chunks coherent.

    Args:
        documents: List of LangChain Document objects from document_loader.py

    Returns:
        A new list of smaller Document chunks, each with the same
        metadata as its parent (source file, page number, etc.)

    Raises:
        ValueError: If the input documents list is empty.
    """

    # ── Validate input ────────────────────────────────────────────────
    if not documents:
        raise ValueError(
            "No documents provided to split. "
            "Make sure load_documents() returned results first."
        )

    print(f"✂️  Splitting {len(documents)} document(s) into chunks...")
    print(f"    CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}\n")

    # ── Build the splitter ────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        # Try splitting on these separators in order:
        # paragraph → newline → sentence → word → character
        separators=["\n\n", "\n", ". ", " ", ""],

        # Measure chunk length in characters (default, matches chunk_size unit)
        length_function=len,

        # Keep the separator at the end of the chunk so sentences stay intact
        is_separator_regex=False,
    )

    # ── Split all documents ───────────────────────────────────────────
    chunks = splitter.split_documents(documents)
    # split_documents preserves the original metadata (source, page, etc.)
    # on each chunk automatically

    print(f"✅ Created {len(chunks)} chunk(s) from {len(documents)} document(s).\n")

    return chunks