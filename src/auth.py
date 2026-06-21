"""
src/auth.py — Role-Based Authentication Module (MongoDB-backed)

Same public API as before (verify_login, signup_user, add_user,
VALID_ROLES, SELF_SIGNUP_ROLES, get_accessible_doc_tags) — app.py does
NOT need any changes. Only the storage backend changed: instead of a
local users.json file, user records now live in a MongoDB collection
called "users" (one document per user).

⚠️ NOTE: This is still a SIMPLIFIED demo auth system for a
project/portfolio build. Passwords are hashed (SHA-256, not salted
per-user beyond what hashlib provides), there is no session/token
expiry, and no brute-force protection. Do NOT use this as-is for a
real production HR system — swap in a proper auth provider (e.g.
Flask-Login + a real DB, or an SSO provider) before deploying with
real employee data.
"""

import hashlib
from typing import Optional, List, Dict

from src.db import get_db


# ──────────────────────────────────────────────
# Role → accessible document tag mapping
# ──────────────────────────────────────────────
# Documents can be tagged (via metadata, e.g. "tag": "confidential")
# when loaded/split. This map controls which tags each role can see.
# "all" means no filtering — every document is visible to that role.
ROLE_DOC_ACCESS: Dict[str, List[str]] = {
    "employee":  ["general"],
    "manager":   ["general", "manager"],
    "hr_admin":  ["all"],
}

VALID_ROLES = list(ROLE_DOC_ACCESS.keys())

# Roles a person is allowed to grant THEMSELVES via the public signup
# form. "hr_admin" is intentionally excluded — see signup_user() below.
SELF_SIGNUP_ROLES = ["employee", "manager"]


def _users_collection():
    """Returns the MongoDB collection that stores user documents."""
    return get_db()["users"]


def _hash_password(password: str) -> str:
    """
    Hash a plain-text password using SHA-256.

    NOTE: For a real system, use a slow, salted hash like bcrypt or
    argon2 instead of plain SHA-256. SHA-256 is used here only to keep
    this demo dependency-free (no extra pip install needed).
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _ensure_default_admin_exists() -> None:
    """
    If the "users" collection is empty (first run against a fresh
    MongoDB database), create one default hr_admin account so the app
    is usable immediately. Also makes sure "username" has a unique
    index, so two people can never end up with the same username even
    under concurrent signups.

    Default login: username="admin", password="admin123"
    ⚠️ Change this immediately after first login in a real deployment.
    """
    col = _users_collection()
    col.create_index("username", unique=True)

    if col.count_documents({}) == 0:
        print("[auth] No users found in MongoDB — creating a default hr_admin account.")
        col.insert_one({
            "username": "admin",
            "password_hash": _hash_password("admin123"),
            "role": "hr_admin",
        })


def load_users() -> Dict[str, dict]:
    """
    Load all users from MongoDB.

    Returns:
        A dict mapping username -> {"password_hash": ..., "role": ...}
        (same shape as the old JSON-file version, for compatibility).
    """
    _ensure_default_admin_exists()
    col = _users_collection()
    return {
        doc["username"]: {"password_hash": doc["password_hash"], "role": doc["role"]}
        for doc in col.find({})
    }


def add_user(username: str, password: str, role: str) -> None:
    """
    Add a new user. This is the ADMIN-SIDE function — it allows ANY
    role, including "hr_admin". It must only be called from a part of
    app.py that is itself gated behind an `if session["role"] ==
    "hr_admin":` check, so that only an already-logged-in HR Admin can
    use it to onboard another HR Admin (or any other role).

    Do NOT wire this function up to the public signup form — use
    signup_user() for that instead.

    Args:
        username: Unique login name.
        password: Plain-text password (will be hashed before saving).
        role: One of "employee", "manager", "hr_admin".

    Raises:
        ValueError: If role is invalid or username already exists.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}.")

    if not username or not password:
        raise ValueError("Username and password are required.")

    _ensure_default_admin_exists()
    col = _users_collection()

    if col.find_one({"username": username}):
        raise ValueError(f"Username '{username}' already exists.")

    col.insert_one({
        "username": username,
        "password_hash": _hash_password(password),
        "role": role,
    })

    print(f"[auth] (admin) Added new user '{username}' with role '{role}'.")


def signup_user(username: str, password: str, role: str) -> None:
    """
    Self-service signup, used by the "Sign Up" tab on the login screen
    in app.py. Any new employee/manager can create their OWN account
    here — no existing login is required to call this.

    ⚠️ SECURITY NOTE: This function deliberately does NOT allow a
    person to self-signup as "hr_admin". If it did, anyone could open
    the public signup form, pick "HR Admin" from a dropdown, and grant
    themselves access to confidential policies (POSH cases,
    disciplinary records, payroll, etc.) — defeating the entire point
    of role-based access control in this app.

    New HR Admin accounts must instead be created by an EXISTING HR
    Admin who is already logged in, using `add_user()` above (wired
    up in app.py behind an `if role == "hr_admin"` check, e.g. on an
    "Add Team Member" panel in the sidebar).

    Args:
        username: Desired unique login name.
        password: Plain-text password (will be hashed before saving).
        role: Must be "employee" or "manager". "hr_admin" is rejected.

    Raises:
        ValueError: If role is "hr_admin", role is otherwise invalid,
                    username/password is empty, or username is taken.
    """
    if not username or not password:
        raise ValueError("Username and password are required.")

    if role not in SELF_SIGNUP_ROLES:
        raise ValueError(
            f"Self-signup only allows roles {SELF_SIGNUP_ROLES}. "
            f"'{role}' accounts must be created by an existing HR Admin."
        )

    _ensure_default_admin_exists()
    col = _users_collection()

    if col.find_one({"username": username}):
        raise ValueError(f"Username '{username}' is already taken.")

    col.insert_one({
        "username": username,
        "password_hash": _hash_password(password),
        "role": role,
    })

    print(f"[auth] (self-signup) New '{role}' account created for '{username}'.")


def verify_login(username: str, password: str) -> Optional[str]:
    """
    Check a username/password pair against MongoDB.

    Args:
        username: Login name entered by the user.
        password: Plain-text password entered by the user.

    Returns:
        The user's role (str) if credentials are valid, otherwise None.
    """
    if not username or not password:
        return None

    _ensure_default_admin_exists()
    col = _users_collection()
    user = col.find_one({"username": username})

    if user is None:
        return None  # Username not found

    if user["password_hash"] != _hash_password(password):
        return None  # Wrong password

    return user["role"]


def get_accessible_doc_tags(role: str) -> List[str]:
    """
    Return the list of document tags a given role is allowed to query.

    Used by rag_chain.py to filter retrieved chunks by metadata before
    they're shown to the LLM/user — so e.g. an "employee" never sees
    chunks tagged "confidential" even if they're semantically relevant.

    Args:
        role: One of "employee", "manager", "hr_admin".

    Returns:
        A list of tags, or ["all"] meaning no filtering should be applied.
    """
    return ROLE_DOC_ACCESS.get(role, ["general"])  # default = least access