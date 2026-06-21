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
    GEMINI_API_KEY,
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


def build_rag_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        google_api_key=GEMINI_API_KEY,
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | RunnableLambda(_extract_text)
    )

    return rag_chain