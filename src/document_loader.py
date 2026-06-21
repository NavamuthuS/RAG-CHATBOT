"""
src/document_loader.py — Document Loading Module

Scans the data directory for supported file types (.pdf, .txt, .docx),
loads each file using the appropriate LangChain loader, and returns
a single combined list of LangChain Document objects.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,       # For .pdf files
    TextLoader,        # For .txt files
    Docx2txtLoader,    # For .docx files
)


def load_documents(data_dir: str) -> List[Document]:
    """
    Scan `data_dir` for .pdf, .txt, and .docx files,
    load each one using the matching LangChain loader,
    and return all pages/documents as a single flat list.

    Args:
        data_dir: Path to the folder containing source documents.

    Returns:
        A list of LangChain Document objects (one per page for PDFs,
        one per file for TXT/DOCX).

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no supported files are found in data_dir.
    """

    # ── Validate directory ────────────────────────────────────────────
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            "Please create it and add your documents."
        )

    # Map file extensions to their LangChain loader classes
    SUPPORTED_EXTENSIONS = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
    }

    all_documents: List[Document] = []
    files_found = 0

    # ── Walk through every file in the directory ──────────────────────
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)

        # Skip sub-directories
        if not os.path.isfile(filepath):
            continue

        # Get the file extension (lowercase) and check if it's supported
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  [SKIP]  '{filename}' — unsupported file type '{ext}'")
            continue

        files_found += 1
        LoaderClass = SUPPORTED_EXTENSIONS[ext]

        try:
            print(f"  [LOAD]  '{filename}' using {LoaderClass.__name__} ...", end=" ")

            loader = LoaderClass(filepath)
            docs = loader.load()  # Returns a list of Document objects

            all_documents.extend(docs)
            print(f"→ {len(docs)} document(s) loaded.")

        except Exception as e:
            # Log the error but continue loading other files
            print(f"\n  [ERROR] Failed to load '{filename}': {e}")

    # ── Final validation ──────────────────────────────────────────────
    if files_found == 0:
        raise ValueError(
            f"No supported files (.pdf, .txt, .docx) found in '{data_dir}'. "
            "Please add documents and try again."
        )

    print(f"\n✅ Total documents loaded: {len(all_documents)} "
          f"(from {files_found} file(s))\n")

    return all_documents