"""
src/db.py — Shared MongoDB connection helper

All the storage modules (auth.py, feedback.py, escalation.py,
analytics.py) import get_db() from here instead of each opening their
own MongoClient. This keeps exactly ONE client/connection pool for the
whole app (Mongo's recommended pattern), and means switching the
connection string only needs to happen in one place.

SETUP (one-time):
1. Install the driver:  pip install pymongo
2. Get a connection string:
   - MongoDB Atlas (free tier, cloud-hosted, recommended):
     https://www.mongodb.com/cloud/atlas/register -> create a free
     cluster -> "Connect" -> "Drivers" -> copy the connection string.
   - OR run MongoDB locally (e.g. `docker run -p 27017:27017 mongo`)
     and just use the default local URI below.
3. Add to your .env:
     MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/
     MONGODB_DB_NAME=hr_chatbot
   If you don't set these, the app defaults to a local MongoDB at
   mongodb://localhost:27017 with database name "hr_chatbot".
"""

import os
import certifi
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "hr_chatbot")

_client = None  # created lazily, on first use


def get_client() -> MongoClient:
    """
    Returns a singleton MongoClient, creating it on first call.

    tlsCAFile=certifi.where() is passed explicitly because on some
    Windows setups, Python's SSL module can't find a valid local CA
    bundle on its own, which causes connections to MongoDB Atlas to
    fail with errors like:
        SSL handshake failed: ... [SSL: TLSV1_ALERT_INTERNAL_ERROR]
    Pointing pymongo at certifi's well-maintained CA bundle fixes this
    in the vast majority of cases.
    """
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    return _client


def get_db():
    """
    Returns the app's Database object (a dict-like accessor for
    collections, e.g. get_db()["users"] or get_db().users).
    """
    return get_client()[MONGODB_DB_NAME]