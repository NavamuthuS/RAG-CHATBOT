"""
src/feedback.py — Feedback Logging Module (MongoDB-backed)

Same public API as before (log_feedback, get_feedback_stats) —
app.py does NOT need any changes. Only the storage backend changed:
feedback records now live in a MongoDB collection called "feedback"
instead of a local feedback_log.json file.

Lets users rate each bot answer (👍 / 👎) so HR admins can spot
outdated, unclear, or wrong policy answers and fix the source
documents accordingly.
"""

from datetime import datetime
from typing import List, Dict

from src.db import get_db


def _feedback_collection():
    """Returns the MongoDB collection that stores feedback records."""
    return get_db()["feedback"]


def log_feedback(question: str, answer: str, rating: str, username: str) -> None:
    """
    Insert one feedback record.

    Args:
        question: The question the user asked.
        answer: The bot's answer that is being rated.
        rating: Either "up" or "down".
        username: Who gave the rating (from session/login).

    Raises:
        ValueError: If rating is not "up" or "down".
    """
    if rating not in ("up", "down"):
        raise ValueError(f"Invalid rating '{rating}'. Must be 'up' or 'down'.")

    col = _feedback_collection()

    col.insert_one({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "question": question,
        "answer": answer,
        "rating": rating,
    })

    print(f"[feedback] Logged '{rating}' from '{username}' for question: {question[:60]!r}")


def get_feedback_stats() -> Dict:
    """
    Compute aggregated feedback stats for the HR-admin analytics view.

    Returns:
        {
            "total": int,
            "up": int,
            "down": int,
            "positive_pct": float,           # 0-100, rounded to 1 decimal
            "negative_examples": [           # most recent negatively-rated Q&A pairs first
                {"question": ..., "answer": ..., "username": ..., "timestamp": ...},
                ...
            ]
        }
    """
    col = _feedback_collection()

    total = col.count_documents({})
    up = col.count_documents({"rating": "up"})
    down = col.count_documents({"rating": "down"})

    positive_pct = round((up / total) * 100, 1) if total > 0 else 0.0

    # Most recent negative examples first, so HR can act on the latest issues.
    # {"_id": 0} excludes MongoDB's internal _id field from the results.
    negative_cursor = (
        col.find({"rating": "down"}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(10)
    )
    negative_examples = list(negative_cursor)

    return {
        "total": total,
        "up": up,
        "down": down,
        "positive_pct": positive_pct,
        "negative_examples": negative_examples,
    }