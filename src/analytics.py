"""
src/analytics.py — Query Analytics Module (HR-Admin Only, MongoDB-backed)

Same public API as before (log_query, get_top_questions,
get_query_stats) — app.py does NOT need any changes. Only the storage
backend changed: every question asked now gets inserted into a
MongoDB collection called "queries" instead of a local
query_log.json file.

Lets HR admins see which topics employees ask about most — useful for
spotting gaps in the published policy docs or building a quick FAQ page.
"""

from datetime import datetime
from collections import Counter
from typing import List, Dict

from src.db import get_db


def _queries_collection():
    """Returns the MongoDB collection that stores question records."""
    return get_db()["queries"]


def log_query(question: str, username: str, role: str) -> None:
    """
    Insert one record of a question being asked.

    Args:
        question: The exact question text the user typed.
        username: Who asked it.
        role: Their role at the time (employee/manager/hr_admin).
    """
    col = _queries_collection()

    col.insert_one({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "role": role,
        "question": question,
    })


def _normalize(question: str) -> str:
    """
    Lightly normalize a question so near-duplicates group together
    when counting frequency (e.g. "What is the leave policy?" and
    "what is the leave policy" should count as the same question).
    """
    return question.strip().lower().rstrip("?.! ")


def get_top_questions(n: int = 5) -> List[Dict]:
    """
    Return the n most frequently asked questions (normalized), most
    common first, along with how many times each was asked.

    NOTE: this fetches just the "question" field for every logged
    query and normalizes/counts it in Python (via the same _normalize()
    + Counter approach as the old JSON-file version), rather than doing
    the grouping inside MongoDB. For an internal HR-chatbot's query
    volume this is plenty fast; if this collection ever grows into the
    millions of rows, switch to a MongoDB aggregation pipeline
    ($group by a normalized field) instead.

    Args:
        n: How many top questions to return.

    Returns:
        [{"question": ..., "count": ...}, ...]
        sorted by count descending.
    """
    col = _queries_collection()
    questions = [doc["question"] for doc in col.find({}, {"question": 1})]

    if not questions:
        return []

    normalized_counts = Counter(_normalize(q) for q in questions)
    top = normalized_counts.most_common(n)

    return [{"question": q, "count": c} for q, c in top]


def get_query_stats() -> Dict:
    """
    Return overall usage stats for the analytics dashboard.

    Returns:
        {
            "total_queries": int,
            "unique_users": int,
            "queries_by_role": {"employee": int, "manager": int, "hr_admin": int},
        }
    """
    col = _queries_collection()

    total_queries = col.count_documents({})
    unique_users = len(col.distinct("username"))

    role_counts = Counter(doc["role"] for doc in col.find({}, {"role": 1}))

    return {
        "total_queries": total_queries,
        "unique_users": unique_users,
        "queries_by_role": dict(role_counts),
    }