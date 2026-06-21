"""
src/prompts.py — Prompt Template Module

Defines the single RAG_PROMPT used by rag_chain.py.

Key rules baked into the prompt:
- Answer ONLY from the provided context.
- If the answer isn't in the context, say "I don't know" — no hallucinating.
- Be concise and cite which part of the context supports the answer.
"""

from langchain_core.prompts import ChatPromptTemplate


# ── System message ────────────────────────────────────────────────────
# Tells the LLM its role and strict rules for answering.
SYSTEM_MESSAGE = """You are a helpful assistant that answers questions \
strictly based on the context provided below.

Rules you MUST follow:
1. Use ONLY the information in the context to answer the question.
2. If the answer is not present in the context, respond with:
   "I don't know based on the provided documents."
   Do NOT make up or infer information that isn't explicitly in the context.
3. Keep your answer clear and concise.
4. If the context contains relevant information, you may quote or paraphrase it directly.

Context:
---------
{context}
---------
"""

# ── Human message ─────────────────────────────────────────────────────
# The actual question from the user.
HUMAN_MESSAGE = "Question: {question}"


# ── Final prompt template ─────────────────────────────────────────────
# ChatPromptTemplate formats the system + human messages into the
# structure expected by OpenAI's chat models.
# Placeholders: {context} and {question} — filled in by rag_chain.py.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", HUMAN_MESSAGE),
    ]
)