"""
src/rag_chain.py — RAG Chain Module
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from config import (
    GEMINI_API_KEYS,
    LLM_MODEL,
    TEMPERATURE,
    TOP_K,
)

from src.prompts import RAG_PROMPT


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _extract_text(response) -> str:
    """Extract plain text from Gemini response (handles both str and list content)."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


# ──────────────────────────────────────────────
# NEW: Automatic Gemini API key rotation
# ──────────────────────────────────────────────
# If GEMINI_API_KEYS in config.py has more than one key (comma-separated in
# .env), this wrapper tries the "current" key first. If that key's call fails
# because its quota/rate-limit ran out (free tier 429 / ResourceExhausted /
# RateLimitError-style errors), it automatically moves on to the next key in
# the list and retries — all inside the same chain.invoke() call, completely
# transparent to app.py and the rest of the app.
#
# It only rotates on QUOTA/RATE-LIMIT style errors. Any other kind of error
# (bad prompt, network issue, etc.) is raised immediately instead of wasting
# time cycling through every key for no reason.
#
# Once a key works, it's remembered as the new "current" key for next time —
# so we don't keep retrying an exhausted key over and over on every question.
class _RotatingGeminiLLM:
    # Substrings that typically show up in Gemini quota/rate-limit errors.
    _QUOTA_ERROR_HINTS = (
        "quota", "429", "resourceexhausted", "resource exhausted",
        "rate limit", "ratelimit", "exceeded your current quota",
    )

    def __init__(self, model: str, temperature: float, api_keys: List[str]):
        if not api_keys:
            raise ValueError("_RotatingGeminiLLM needs at least one API key.")
        self._model = model
        self._temperature = temperature
        self._api_keys = api_keys
        self._current_index = 0
        self._llm_cache = {}  # index -> ChatGoogleGenerativeAI instance

    def _llm_for(self, index: int) -> ChatGoogleGenerativeAI:
        if index not in self._llm_cache:
            self._llm_cache[index] = ChatGoogleGenerativeAI(
                model=self._model,
                temperature=self._temperature,
                google_api_key=self._api_keys[index],
            )
        return self._llm_cache[index]

    @staticmethod
    def _looks_like_quota_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(hint in msg for hint in _RotatingGeminiLLM._QUOTA_ERROR_HINTS)

    def invoke(self, prompt_value):
        total_keys = len(self._api_keys)
        last_error = None

        for attempt in range(total_keys):
            idx = (self._current_index + attempt) % total_keys
            llm = self._llm_for(idx)
            try:
                response = llm.invoke(prompt_value)
                if idx != self._current_index:
                    print(f"[rag_chain] Switched to Gemini API key #{idx + 1} "
                          f"(previous key's quota was exhausted).")
                self._current_index = idx
                return response
            except Exception as exc:
                if self._looks_like_quota_error(exc):
                    last_error = exc
                    print(f"[rag_chain] Gemini API key #{idx + 1} hit its quota/rate "
                          f"limit, trying the next key...")
                    continue
                # Not a quota issue (bad request, network error, etc.) -
                # no point rotating keys for this, raise immediately.
                raise

        # Every single key is exhausted.
        raise RuntimeError(
            f"All {total_keys} configured Gemini API key(s) have hit their quota/rate "
            f"limit. Add another key to GEMINI_API_KEYS in .env, or wait for the "
            f"quota to reset. Last error: {last_error}"
        )


def build_rag_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    rotating_llm = _RotatingGeminiLLM(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_keys=GEMINI_API_KEYS,
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | RunnableLambda(rotating_llm.invoke)
        | RunnableLambda(_extract_text)
    )

    return rag_chain