"""
config.py — Central configuration for the RAG Chatbot.

All settings are loaded here from environment variables (.env file).
Every other file in this project imports from here — no hardcoded values elsewhere.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# ──────────────────────────────────────────────
# Google Gemini API Key (required)
# ──────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Please copy .env.example to .env and fill in your key."
    )

# Make the key available to the Google SDK automatically
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ──────────────────────────────────────────────
# File / Directory Paths
# ──────────────────────────────────────────────

# Where users drop their PDF / TXT / DOCX source documents
DATA_DIR: str = os.getenv("DATA_DIR", "data/documents")

# Where ChromaDB will persist its vector index on disk
CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chroma_db")

# ──────────────────────────────────────────────
# Text Splitting
# ──────────────────────────────────────────────

# Maximum number of characters per chunk
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

# How many characters the next chunk shares with the previous one
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────

# Gemini embedding model
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


# ──────────────────────────────────────────────
# LLM Settings
# ──────────────────────────────────────────────

# Gemini chat model (free tier available)
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Controls randomness: 0.0 = deterministic, 1.0 = very creative
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

# ──────────────────────────────────────────────
# Retrieval Settings
# ──────────────────────────────────────────────
# Auth Settings
# ──────────────────────────────────────────────
USERS_FILE: str = os.getenv("USERS_FILE", "users.json")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR: str = os.getenv("LOG_DIR", "logs")
# ──────────────────────────────────────────────

# Number of document chunks to retrieve from ChromaDB per query
TOP_K: int = int(os.getenv("TOP_K", "4"))