"""
src/vector_store.py — Vector Store Module (FAISS)
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
)

FAISS_DIR = CHROMA_DB_DIR  # reuse same config path


def _get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def build_vector_store(chunks: List[Document]):
    if not chunks:
        raise ValueError("No chunks provided to build_vector_store().")

    print(f"🔢 Embedding {len(chunks)} chunk(s) with '{EMBEDDING_MODEL}'...")
    print(f"    Persisting vector store to '{FAISS_DIR}' ...\n")

    embeddings = _get_embedding_function()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_DIR)

    print(f"✅ Vector store built and saved to '{FAISS_DIR}'.\n")
    return vector_store


def load_vector_store():
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(
            f"No vector store found at '{FAISS_DIR}'. "
            "Please click 'Build / Rebuild Knowledge Base' first."
        )

    print(f"📂 Loading vector store from '{FAISS_DIR}' ...")
    embeddings = _get_embedding_function()
    vector_store = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Vector store loaded successfully.\n")
    return vector_store