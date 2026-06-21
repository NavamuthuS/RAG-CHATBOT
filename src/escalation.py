"""
src/escalation.py — Escalation to Human HR Module (MongoDB-backed)

Same public API as before (should_escalate, create_escalation_ticket,
get_open_tickets) — app.py does NOT need any changes. Only the storage
backend changed: tickets now live in a MongoDB collection called
"escalations" instead of a local escalation_log.json file.
"""

from datetime import datetime
from typing import List, Dict

from src.db import get_db

# ──────────────────────────────────────────────
# Keywords that indicate a personal/sensitive HR matter —
# things the bot should NOT try to answer generically, even if it
# retrieves something that looks relevant from the policy documents.
# ──────────────────────────────────────────────
SENSITIVE_KEYWORDS = [
    "salary", "pay raise", "hike", "increment", "bonus",
    "disciplinary", "termination", "fired", "layoff",
    "harassment", "complaint", "posh", "grievance",
    "resign", "resignation", "notice period dispute",
    "medical condition", "pregnant", "pregnancy",  # personal medical info
]

# Phrases that mean "the bot didn't actually find an answer"
UNKNOWN_PHRASES = [
    "i don't know",
    "i don't know based on the provided documents",
    "not present in the context",
    "no answer found",
]


def _escalations_collection():
    """Returns the MongoDB collection that stores escalation tickets."""
    return get_db()["escalations"]


def should_escalate(question: str, answer: str) -> bool:
    """
    Decide whether the chat UI should show an "Escalate to HR" button
    for this Q&A pair.

    Triggers True if:
    - The answer text matches one of the UNKNOWN_PHRASES (bot doesn't know), OR
    - The question contains a sensitive keyword (personal/confidential topic),
      even if the bot did answer — personal matters should go to a human
      regardless of how confident the bot sounds.

    Args:
        question: The user's original question.
        answer: The bot's generated answer.

    Returns:
        True if a human HR escalation should be offered, else False.
    """
    answer_lower = (answer or "").lower()
    question_lower = (question or "").lower()

    # Case 1: Bot doesn't know the answer
    for phrase in UNKNOWN_PHRASES:
        if phrase in answer_lower:
            return True

    # Case 2: Question touches a sensitive/personal HR topic
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in question_lower:
            return True

    return False


def create_escalation_ticket(question: str, username: str, reason: str) -> Dict:
    """
    Log a new escalation ticket for HR to follow up on.

    Args:
        question: The original question that needs human attention.
        username: Who raised it.
        reason: Short explanation, e.g. "bot_unknown" or "sensitive_topic".

    Returns:
        The ticket dict that was just created (useful for showing a
        confirmation message / ticket ID in the UI). Has the same
        shape as before: ticket_id, timestamp, username, question,
        reason, status.

    --------------------------------------------------------------
    NOTE: app.py already wires this up to a real EmailJS notification
    (see send_escalation_email() in app.py) right after this ticket is
    created, so HR gets a real email in addition to this MongoDB
    record.
    --------------------------------------------------------------
    """
    col = _escalations_collection()

    # Sequential, zero-padded ticket IDs (ESC-00001, ESC-00002, ...),
    # same numbering style as the old JSON-file version.
    ticket_number = col.count_documents({}) + 1

    ticket = {
        "ticket_id": f"ESC-{ticket_number:05d}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "username": username,
        "question": question,
        "reason": reason,
        "status": "open",  # could later become "resolved" via an admin action
    }

    col.insert_one(ticket)

    # Don't leak MongoDB's internal _id back to callers — keep the
    # returned dict identical in shape to the old JSON-based version.
    ticket.pop("_id", None)

    print(f"[escalation] Created ticket {ticket['ticket_id']} for '{username}' (reason: {reason})")

    return ticket


def get_open_tickets() -> List[Dict]:
    """
    Return all tickets with status "open", most recent first.
    Used by an HR-admin-only view in app.py to action escalations.
    """
    col = _escalations_collection()
    cursor = col.find({"status": "open"}, {"_id": 0}).sort("timestamp", -1)
    return list(cursor)