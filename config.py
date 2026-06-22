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
# Google Gemini API Key(s) (required)
# ──────────────────────────────────────────────
# NEW: Support MULTIPLE Gemini API keys so the app can automatically rotate
# to the next key when one hits its free-tier quota/rate limit, instead of
# the whole chatbot breaking.
#
# In your .env file, set GEMINI_API_KEYS as a COMMA-SEPARATED list:
#
#   GEMINI_API_KEYS=AIzaSy_first_key_here,AIzaSy_second_key_here,AIzaSy_third_key_here
#
# (Get extra free keys the same way as before — https://aistudio.google.com/apikey —
# using a different Google account for each one, then add them all here separated
# by commas, no spaces needed around the commas.)
#
# Backward compatible: if you still only have the old single GEMINI_API_KEY set,
# everything keeps working exactly as before — it's just treated as a list of 1.
_raw_keys = os.getenv("GEMINI_API_KEYS", "").strip()
_single_key = os.getenv("GEMINI_API_KEY", "").strip()

if _raw_keys:
    GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
elif _single_key:
    GEMINI_API_KEYS = [_single_key]
else:
    GEMINI_API_KEYS = []

if not GEMINI_API_KEYS:
    raise EnvironmentError(
        "No Gemini API key found. "
        "Please set GEMINI_API_KEYS (comma-separated, recommended) or "
        "GEMINI_API_KEY in your .env file. "
        "Copy .env.example to .env and fill in your key(s)."
    )

# Kept for backward compatibility — anything in this project that still imports
# the old single GEMINI_API_KEY (e.g. the embeddings model, personal-document
# upload feature in app.py) just uses the FIRST key in the list.
GEMINI_API_KEY: str = GEMINI_API_KEYS[0]

# Make the first key available to the Google SDK automatically
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