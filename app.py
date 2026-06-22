import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

"""
app.py - Flask Chat Interface for the HR Policy RAG Chatbot
Run with: python app.py

Features wired in:
- Role-based login (employee / manager / hr_admin) via src/auth.py
- Build/Rebuild Knowledge Base (hr_admin only)
- Role-aware chat
- Thumbs up/down feedback on every bot answer (src/feedback.py)
- "Escalate to HR" button when the bot doesn't know or the topic is
  sensitive (src/escalation.py)
- Per-question analytics logging + an hr_admin-only dashboard (src/analytics.py)
- "Export conversation as PDF" button (src/export_utils.py)

NEW in this version (all implemented with stock Python/Flask + the browser's
own built-in APIs - no extra paid services required unless noted):
- Dark / Light theme toggle              (pure CSS variables + localStorage)
- Voice input                            (browser Web Speech API - SpeechRecognition)
- Voice output (TTS)                     (browser Web Speech API - speechSynthesis)
- Tamil / English UI toggle              (client-side dictionary swap of UI strings)
- Real-time analytics charts             (Plotly.js, fed by a small JSON API)
- Email escalation (via EmailJS)         (api.emailjs.com REST API - needs a
                                           free EmailJS account, see notes below)
- Per-user chat history saved to JSON    (data/chat_history/<user>.json)
- Keyword highlighting in bot answers    (server-side regex -> <mark> tags)
- User profile page                      (/profile)
- Confidence score on answers            (heuristic - see compute_confidence())
- Password show/hide eye-toggle on Sign In + Sign Up forms (pure HTML/JS)

IMPORTANT, please read:
- Voice input/output and the theme toggle work 100% inside the browser, so
  there's nothing to sign up for - the browser will just ask the user for
  mic permission the first time they click the mic button, which is normal.
- The Tamil/English toggle here only swaps the *static UI text* (buttons,
  labels, placeholders). The bot's actual answers come straight out of your
  RAG chain in whatever language the source documents/LLM produce - making
  it auto-translate the live answers too would need a translation API
  (e.g. Google Translate / Azure Translator), which means signing up for a
  key on another site. I left a clear extension point for that
  (see TODO near send_escalation_email-style helpers) instead of wiring one
  in without your say-so.
- Real email escalation is done through EmailJS (https://www.emailjs.com/),
  since it lets you send real emails without running your own mail server.
  This DOES require creating a free account on emailjs.com yourself (Claude
  can't do that step for you):
    1. Sign up at https://www.emailjs.com/ and add an "Email Service"
       (connect your Gmail/Outlook/SMTP - whichever you already use).
    2. Create an "Email Template" with these variables in the template body:
       {{ticket_id}}, {{username}}, {{reason}}, {{question}}
    3. Copy your Service ID, Template ID, and Public Key from the dashboard.
    4. Go to Account -> Security, turn ON "Allow EmailJS API for non-browser
       applications", and copy the Private Key shown there (this app calls
       EmailJS from the Flask server, not the browser, so it needs this).
    5. Add these to your .env: EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID,
       EMAILJS_PUBLIC_KEY, EMAILJS_PRIVATE_KEY.
  If any of these are missing the app keeps working exactly as before - it
  just logs a message instead of sending an email, so nothing breaks for
  people who don't set this up.
- The confidence score is a heuristic (keyword overlap + answer length +
  "I don't know"-style phrase detection) since the RAG chain interface in
  this project (src/rag_chain.py) doesn't expose retrieval similarity
  scores to app.py. If you later expose a real similarity/confidence value
  from build_rag_chain()/chain.invoke(), swap it in inside compute_confidence().
"""

from flask import Flask, request, jsonify, session, redirect, url_for, send_file
from functools import wraps
import os
import re
import json
import threading
import secrets
import urllib.request
import urllib.error
from datetime import datetime
from collections import Counter

from config import DATA_DIR, GEMINI_API_KEY, EMBEDDING_MODEL
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import build_vector_store, load_vector_store
from src.rag_chain import build_rag_chain
from src.auth import verify_login, signup_user, add_user, VALID_ROLES, SELF_SIGNUP_ROLES
from src.feedback import log_feedback, get_feedback_stats
from src.escalation import should_escalate, create_escalation_ticket, get_open_tickets
from src.analytics import log_query, get_top_questions, get_query_stats
from src.export_utils import export_conversation_to_pdf

# NEW: used only for the personal "ask about my own document" feature -
# same loaders/embeddings/vector-store tech as the main knowledge base
# (document_loader.py / vector_store.py), just applied to ONE user-uploaded
# file at a time instead of the whole data/documents/ folder.
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

app = Flask(__name__)
# Secret key for signing session cookies. Set FLASK_SECRET_KEY in .env for a
# stable key across restarts (otherwise everyone is logged out on restart).
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

_rag_chain = None
_chain_lock = threading.Lock()
_build_status = {"running": False, "message": "", "success": None}
_chat_results = {}
_chat_lock = threading.Lock()

EXPORT_DIR = "exports"
HISTORY_DIR = "data/chat_history"

# ──────────────────────────────────────────────
# Email escalation config (optional - set in .env)
# Uses EmailJS (https://www.emailjs.com/) so we don't need to manage SMTP
# credentials directly. See the docstring at the top of this file for the
# one-time setup steps on the EmailJS dashboard.
# ──────────────────────────────────────────────
EMAILJS_SERVICE_ID = os.getenv("EMAILJS_SERVICE_ID", "")
EMAILJS_TEMPLATE_ID = os.getenv("EMAILJS_TEMPLATE_ID", "")
EMAILJS_PUBLIC_KEY = os.getenv("EMAILJS_PUBLIC_KEY", "")
EMAILJS_PRIVATE_KEY = os.getenv("EMAILJS_PRIVATE_KEY", "")
HR_ESCALATION_EMAIL = os.getenv("HR_ESCALATION_EMAIL", "")
EMAIL_ESCALATION_ENABLED = bool(
    EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_PUBLIC_KEY and HR_ESCALATION_EMAIL
)

STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    "what", "how", "do", "does", "i", "my", "can", "you", "please", "tell",
    "me", "about", "with", "that", "this", "will", "would", "should", "could",
}

# ──────────────────────────────────────────────
# NEW: Document upload settings (both HR-official and personal uploads)
# ──────────────────────────────────────────────
ALLOWED_DOC_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Maps extension -> LangChain loader class, same mapping document_loader.py
# uses internally, just exposed here so we can load ONE uploaded file
# directly (instead of re-scanning the whole data/documents/ folder).
PERSONAL_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

PERSONAL_UPLOAD_DIR = "data/personal_uploads"

# In-memory store of each user's personal "ask about my document" RAG
# chain. Lives only as long as the Flask process runs (same lifetime as
# _rag_chain, the main shared knowledge base chain) - if you restart the
# server, users need to re-upload their personal document.
_personal_chains = {}
_personal_chain_lock = threading.Lock()


def _secure_filename(filename: str) -> str:
    """
    Minimal, dependency-free version of werkzeug's secure_filename().
    Strips directory separators and any character that isn't
    alphanumeric/dot/dash/underscore, so an uploaded filename can never
    be used for path traversal (e.g. "../../etc/passwd").
    """
    filename = os.path.basename(filename or "")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return cleaned or "uploaded_file"


def _personal_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Same embedding model/config as the main knowledge base (vector_store.py)."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)


def _load_single_file(filepath: str):
    """Load ONE file (pdf/txt/docx) into LangChain Document objects."""
    ext = os.path.splitext(filepath)[1].lower()
    LoaderClass = PERSONAL_LOADER_MAP.get(ext)
    if LoaderClass is None:
        raise ValueError(f"Unsupported file type '{ext}'.")
    loader = LoaderClass(filepath)
    return loader.load()


# ──────────────────────────────────────────────
# Auth decorators
# ──────────────────────────────────────────────
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return view_func(*args, **kwargs)
    return wrapper


def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login_page"))
            if session.get("role") not in allowed_roles:
                return jsonify({"error": "Forbidden: your role cannot access this."}), 403
            return view_func(*args, **kwargs)
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# RAG chain / knowledge base helpers
# ──────────────────────────────────────────────
def get_rag_chain():
    global _rag_chain
    with _chain_lock:
        if _rag_chain is not None:
            return _rag_chain
        try:
            vector_store = load_vector_store()
            _rag_chain = build_rag_chain(vector_store)
            return _rag_chain
        except Exception:
            return None


def do_build():
    global _rag_chain, _build_status
    try:
        _build_status = {"running": True, "message": "Loading documents...", "success": None}
        documents = load_documents(DATA_DIR)
        _build_status["message"] = "Splitting into chunks..."
        chunks = split_documents(documents)
        _build_status["message"] = "Embedding into vector store..."
        build_vector_store(chunks)
        with _chain_lock:
            _rag_chain = None
        _build_status = {
            "running": False,
            "message": f"{len(documents)} document(s) -> {len(chunks)} chunks done!",
            "success": True,
        }
    except Exception as e:
        _build_status = {"running": False, "message": str(e), "success": False}


# ──────────────────────────────────────────────
# NEW: HTML escaping + keyword highlighting
# ──────────────────────────────────────────────
def esc_html(text):
    """Server-side HTML escaping for text we'll later inject as innerHTML."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def highlight_keywords(question, escaped_answer_html):
    """
    Wraps words from the question that also appear in the answer with
    <mark> tags, so the user can quickly spot which parts of the answer
    are directly relevant to what they asked.

    IMPORTANT: call this AFTER esc_html() on the answer, and only with
    plain alphabetic keywords (regex below), so we never introduce any
    unescaped/unsafe HTML from user input.
    """
    words = re.findall(r"[A-Za-z]{3,}", question or "")
    keywords = sorted(set(w.lower() for w in words if w.lower() not in STOPWORDS), key=len, reverse=True)
    if not keywords:
        return escaped_answer_html

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped_answer_html)


# ──────────────────────────────────────────────
# NEW: Confidence score (heuristic)
# ──────────────────────────────────────────────
def compute_confidence(question, answer):
    """
    Heuristic confidence score (0-100) for an answer, used since the RAG
    chain interface here doesn't expose a real retrieval/similarity score
    to app.py. Looks at:
      - known "I don't know" / error-style phrases -> low confidence
      - how many of the question's keywords show up in the answer
      - answer length (very short answers are usually less complete)
    This is NOT a substitute for a real model-reported confidence value -
    swap it out if/when build_rag_chain() starts returning one.
    """
    if not answer:
        return 0

    low_signal_phrases = [
        "i don't know", "i do not know", "not sure", "no answer found",
        "knowledge base not built", "error:", "cannot find", "couldn't find",
        "no information", "unable to find",
    ]
    lower_answer = answer.lower()
    if any(p in lower_answer for p in low_signal_phrases):
        return 15

    q_words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", question or "")) - STOPWORDS
    a_words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", lower_answer))
    overlap_ratio = (len(q_words & a_words) / len(q_words)) if q_words else 0.5

    length_score = min(len(answer) / 400.0, 1.0)

    score = 40 + (overlap_ratio * 40) + (length_score * 20)
    return int(max(5, min(round(score), 97)))


# ──────────────────────────────────────────────
# NEW: Per-user chat history persisted to JSON
# ──────────────────────────────────────────────
def history_file_path(username):
    safe = "".join(c for c in username if c.isalnum() or c in ("-", "_")) or "user"
    return os.path.join(HISTORY_DIR, f"{safe}.json")


def save_chat_turn(username, question, answer):
    """Appends one Q/A turn to the user's persistent JSON chat history file."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = history_file_path(username)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []

    data.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[app] Failed to save chat history: {e}")


def load_chat_history(username, limit=200):
    """Returns the last `limit` saved Q/A turns for this user, oldest first."""
    path = history_file_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-limit:]
    except Exception:
        return []


def history_to_session_format(history):
    """Converts saved [{question, answer, timestamp}, ...] into the flat
    [{role, content}, ...] shape used by session['history'] / PDF export."""
    flat = []
    for turn in history:
        flat.append({"role": "user", "content": turn.get("question", "")})
        flat.append({"role": "bot", "content": turn.get("answer", "")})
    return flat


def render_history_messages(history):
    """Renders previously-saved chat turns as read-only message bubbles for
    the initial page load, so a user's chat resumes where they left off."""
    html = ""
    for turn in history:
        role = turn.get("role")
        content = esc_html(turn.get("content", ""))
        if role == "user":
            html += (
                '<div class="msg user"><div class="msg-row">'
                '<div class="avatar">&#128100;</div>'
                f'<div class="bubble">{content}</div>'
                '</div></div>'
            )
        else:
            html += (
                '<div class="msg bot"><div class="msg-row">'
                '<div class="avatar">&#129302;</div>'
                f'<div class="bubble">{content}</div>'
                '</div></div>'
            )
    return html


# ──────────────────────────────────────────────
# NEW: Email escalation via EmailJS
# ──────────────────────────────────────────────
def send_escalation_email(ticket):
    """
    Sends a real email notification to HR when a question is escalated,
    using the EmailJS REST API (https://www.emailjs.com/) so we don't need
    to manage SMTP credentials ourselves. No-ops quietly (just logs to
    console) if EMAILJS_* env vars aren't fully configured - this keeps
    local/dev usage working without forcing anyone to set up EmailJS just
    to run the app.

    The template_params keys below (ticket_id, username, reason, question,
    to_email) must match the {{variable}} placeholders used inside your
    EmailJS email template - see the setup notes at the top of this file.
    """
    if not EMAIL_ESCALATION_ENABLED:
        print("[app] Email escalation skipped - EMAILJS_* env vars not fully configured in .env.")
        return False

    payload = {
        "service_id": EMAILJS_SERVICE_ID,
        "template_id": EMAILJS_TEMPLATE_ID,
        "user_id": EMAILJS_PUBLIC_KEY,
        "template_params": {
            "ticket_id": ticket.get("ticket_id", ""),
            "username": ticket.get("username", ""),
            "reason": ticket.get("reason", ""),
            "question": ticket.get("question", ""),
            "to_email": HR_ESCALATION_EMAIL,
        },
    }
    # The Private Key lets the server (a "non-browser" caller, as far as
    # EmailJS is concerned) send mail - turn this on under Account > Security
    # on the EmailJS dashboard, see the setup notes at the top of this file.
    if EMAILJS_PRIVATE_KEY:
        payload["accessToken"] = EMAILJS_PRIVATE_KEY

    try:
        req = urllib.request.Request(
            "https://api.emailjs.com/api/v1.0/email/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                print(f"[app] EmailJS responded with unexpected status {resp.status}")
            return ok
    except urllib.error.HTTPError as e:
        print(f"[app] EmailJS HTTP error {e.code}: {e.read().decode(errors='ignore')}")
        return False
    except Exception as e:
        print(f"[app] Email escalation (EmailJS) failed: {e}")
        return False


def do_chat(chat_id, question, username, role):
    """
    Run one question through the RAG chain, log it for analytics, decide
    whether it should be flagged for HR escalation, compute a confidence
    score + keyword-highlighted HTML, and persist the turn to this user's
    JSON chat history file.
    """
    global _chat_results
    try:
        chain = get_rag_chain()
        if chain is None:
            answer = "Knowledge base not built yet. Please ask an HR admin to build it first."
        else:
            answer = chain.invoke(question)
            if not answer:
                answer = "No answer found. Try rebuilding the knowledge base."
    except Exception as e:
        answer = "Error: " + str(e)

    # Log every question for the analytics dashboard, regardless of outcome
    try:
        log_query(question, username, role)
    except Exception as e:
        print(f"[app] analytics logging failed: {e}")

    escalate_flag = should_escalate(question, answer)
    confidence = compute_confidence(question, answer)
    answer_html = highlight_keywords(question, esc_html(answer))

    try:
        save_chat_turn(username, question, answer)
    except Exception as e:
        print(f"[app] failed to persist chat history: {e}")

    with _chat_lock:
        _chat_results[chat_id] = {
            "answer": answer,
            "answer_html": answer_html,
            "confidence": confidence,
            "escalate": escalate_flag,
        }


def do_personal_chat(chat_id, question, username):
    """
    Like do_chat(), but answers ONLY from this user's own personally
    uploaded document (see /personal_upload), not the shared HR
    knowledge base. Writes into the SAME _chat_results dict as do_chat()
    so the existing /chat_result polling endpoint works for both.

    Deliberately does NOT call log_query()/should_escalate() - personal
    document questions aren't company HR policy questions, so they
    shouldn't show up in the HR-wide analytics dashboard or trigger an
    "Escalate to HR" button.
    """
    with _personal_chain_lock:
        chain = _personal_chains.get(username)

    if chain is None:
        answer = "You haven't uploaded a personal document yet. Use 'Upload My Document' in the sidebar first."
    else:
        try:
            answer = chain.invoke(question)
            if not answer:
                answer = "No answer found in your uploaded document."
        except Exception as e:
            answer = "Error: " + str(e)

    confidence = compute_confidence(question, answer)
    answer_html = highlight_keywords(question, esc_html(answer))

    try:
        save_chat_turn(username, question, answer)
    except Exception as e:
        print(f"[app] failed to persist personal chat history: {e}")

    with _chat_lock:
        _chat_results[chat_id] = {
            "answer": answer,
            "answer_html": answer_html,
            "confidence": confidence,
            "escalate": False,  # personal doc Q&A never triggers HR escalation
        }


# ──────────────────────────────────────────────
# HTML — Login Page
# ──────────────────────────────────────────────
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Policy Chatbot — Login</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0;
         display: flex; align-items: center; justify-content: center; height: 100vh; }
  .card { background: #1a1d27; border: 1px solid #2d3144; border-radius: 14px;
          padding: 36px 32px; width: 340px; }
  .card h1 { font-size: 20px; margin-bottom: 6px; }
  .card p.sub { font-size: 13px; color: #7a82a0; margin-bottom: 22px; }
  label { font-size: 12px; color: #a0a8c0; display: block; margin-bottom: 6px; }
  input, select { width: 100%; background: #0f1117; border: 1px solid #2d3144; border-radius: 8px;
          color: #e0e0e0; font-size: 14px; padding: 10px 12px; margin-bottom: 16px; outline: none; }
  input:focus, select:focus { border-color: #3b4fd8; }
  button { width: 100%; background: #3b4fd8; border: none; border-radius: 8px; color: #fff;
           padding: 11px; font-size: 14px; cursor: pointer; }
  button:hover { background: #4c5fe8; }
  .error { background: #3a1a1a; color: #f87171; font-size: 12px; padding: 8px 12px;
           border-radius: 6px; margin-bottom: 16px; }
  .success { background: #1a3a2a; color: #4ade80; font-size: 12px; padding: 8px 12px;
           border-radius: 6px; margin-bottom: 16px; }
  .info { background: #1a2a3a; color: #93c5fd; font-size: 12px; padding: 8px 12px;
           border-radius: 6px; margin-bottom: 16px; line-height: 1.5; }
  .hint { font-size: 11px; color: #4a5270; margin-top: 18px; line-height: 1.6; }
  .tabs { display: flex; border-bottom: 1px solid #2d3144; margin-bottom: 20px; }
  .tab { flex: 1; text-align: center; padding: 10px 0; font-size: 13px; font-weight: 600;
         color: #7a82a0; cursor: pointer; border-bottom: 2px solid transparent; }
  .tab.active { color: #6c8cff; border-bottom-color: #3b4fd8; }
  .panel { display: none; }
  .panel.active { display: block; }
  /* NEW: password show/hide eye-toggle */
  .pwd-wrap { position: relative; margin-bottom: 16px; }
  .pwd-wrap input { margin-bottom: 0; padding-right: 38px; }
  .pwd-toggle { position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
                cursor: pointer; font-size: 16px; user-select: none; opacity: 0.7; line-height: 1; }
  .pwd-toggle:hover { opacity: 1; }
</style>
</head>
<body>
  <div class="card">
    <h1>HR Policy Assistant</h1>
    <p class="sub">Sign in to ask questions about company policies.</p>

    <div class="tabs">
      <div class="tab __SIGNIN_TAB_ACTIVE__" id="tab-signin" onclick="showTab('signin')">Sign In</div>
      <div class="tab __SIGNUP_TAB_ACTIVE__" id="tab-signup" onclick="showTab('signup')">Sign Up</div>
    </div>

    <!-- ---------- SIGN IN PANEL ---------- -->
    <div class="panel __SIGNIN_TAB_ACTIVE__" id="panel-signin">
      __ERROR_BLOCK__
      __SIGNUP_SUCCESS_BLOCK__
      <form method="POST" action="/login">
        <label for="username">Username</label>
        <input type="text" id="username" name="username" autocomplete="username" required>
        <label for="password">Password</label>
        <div class="pwd-wrap">
          <input type="password" id="password" name="password" autocomplete="current-password" required>
          <span class="pwd-toggle" onclick="togglePwd('password', this)">&#128065;</span>
        </div>
        <button type="submit">Sign In</button>
      </form>
      <p class="hint">First time running this app? Default HR admin login is
        <strong>admin / admin123</strong> — created automatically on first run.
        Please change it.</p>
    </div>

    <!-- ---------- SIGN UP PANEL ---------- -->
    <div class="panel __SIGNUP_TAB_ACTIVE__" id="panel-signup">
      <div class="info">
        New employees/managers can create their own account here.
        <strong>HR Admin accounts</strong> can only be created by an existing
        HR Admin (from the sidebar after logging in) — this keeps confidential
        HR data from being self-granted.
      </div>
      __SIGNUP_ERROR_BLOCK__
      <form method="POST" action="/signup">
        <label for="su_username">Choose a username</label>
        <input type="text" id="su_username" name="username" required>
        <label for="su_password">Choose a password</label>
        <div class="pwd-wrap">
          <input type="password" id="su_password" name="password" required>
          <span class="pwd-toggle" onclick="togglePwd('su_password', this)">&#128065;</span>
        </div>
        <label for="su_confirm">Confirm password</label>
        <div class="pwd-wrap">
          <input type="password" id="su_confirm" name="confirm" required>
          <span class="pwd-toggle" onclick="togglePwd('su_confirm', this)">&#128065;</span>
        </div>
        <label for="su_role">I am a...</label>
        <select id="su_role" name="role" required>
          __SIGNUP_ROLE_OPTIONS__
        </select>
        <button type="submit">Create Account</button>
      </form>
    </div>
  </div>
  <script>
    function showTab(tab) {
      document.getElementById('panel-signin').classList.toggle('active', tab === 'signin');
      document.getElementById('panel-signup').classList.toggle('active', tab === 'signup');
      document.getElementById('tab-signin').classList.toggle('active', tab === 'signin');
      document.getElementById('tab-signup').classList.toggle('active', tab === 'signup');
    }

    // NEW: password show/hide eye-toggle (Sign In + Sign Up forms)
    function togglePwd(inputId, iconEl) {
      var input = document.getElementById(inputId);
      if (input.type === 'password') {
        input.type = 'text';
        iconEl.innerHTML = '&#128064;';
      } else {
        input.type = 'password';
        iconEl.innerHTML = '&#128065;';
      }
    }
  </script>
</body>
</html>"""


def render_login(error="", signup_error="", signup_success="", active_tab="signin"):
    """
    Fills in LOGIN_HTML's placeholders for the sign-in/sign-up tabs.
    Kept as simple string replacement to match the rest of this file's
    style (no Jinja templates used anywhere else in this app).
    """
    error_html = f'<div class="error">{error}</div>' if error else ""
    signup_error_html = f'<div class="error">{signup_error}</div>' if signup_error else ""
    signup_success_html = f'<div class="success">{signup_success}</div>' if signup_success else ""

    role_options = "".join(
        f'<option value="{r}">{r.replace("_", " ").title()}</option>'
        for r in SELF_SIGNUP_ROLES
    )

    signin_active = "active" if active_tab == "signin" else ""
    signup_active = "active" if active_tab == "signup" else ""

    return (
        LOGIN_HTML
        .replace("__ERROR_BLOCK__", error_html)
        .replace("__SIGNUP_ERROR_BLOCK__", signup_error_html)
        .replace("__SIGNUP_SUCCESS_BLOCK__", signup_success_html)
        .replace("__SIGNUP_ROLE_OPTIONS__", role_options)
        .replace("__SIGNIN_TAB_ACTIVE__", signin_active)
        .replace("__SIGNUP_TAB_ACTIVE__", signup_active)
    )


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        if "username" in session:
            return redirect(url_for("index"))
        return render_login()

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    role = verify_login(username, password)
    if role is None:
        return render_login(error="Invalid username or password."), 401

    session["username"] = username
    session["role"] = role
    # NEW: resume the user's previous conversation (from their saved JSON
    # history) instead of always starting from a blank slate.
    session["history"] = history_to_session_format(load_chat_history(username, limit=50))
    return redirect(url_for("index"))


# ──────────────────────────────────────────────
# Signup route — self-service, employee/manager only
# ──────────────────────────────────────────────
@app.route("/signup", methods=["POST"])
def signup_page():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")
    role = request.form.get("role", "")

    if password != confirm:
        return render_login(signup_error="Passwords do not match.", active_tab="signup"), 400

    try:
        signup_user(username, password, role)
        return render_login(
            signup_success=f"Account created for '{username}'. Please sign in.",
            active_tab="signin",
        )
    except ValueError as e:
        return render_login(signup_error=str(e), active_tab="signup"), 400


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ──────────────────────────────────────────────
# HTML — Main Chat Page
# ──────────────────────────────────────────────
CHAT_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HR Policy Chatbot</title>
<style>
  :root {
    --bg: #0f1117;
    --bg-alt: #1a1d27;
    --bg-input: #0f1117;
    --bg-input-alt: #252840;
    --border: #2d3144;
    --text: #e0e0e0;
    --text-dim: #7a82a0;
    --text-faint: #4a5270;
    --accent: #3b4fd8;
    --accent-hover: #4c5fe8;
    --bubble-user-bg: #252f5a;
    --bubble-user-border: #3b4fd8;
    --green-bg: #1a3a2a; --green-text: #4ade80;
    --red-bg: #3a1a1a; --red-text: #f87171;
    --blue-bg: #1a2a3a; --blue-text: #60a5fa;
    --yellow-bg: #3a2a1a; --yellow-text: #fbbf24; --yellow-border: #5a4020;
    --mark-bg: #5a4a1a; --mark-text: #ffe08a;
    --code: #7dd3b8;
  }
  html[data-theme="light"] {
    --bg: #f3f4f8;
    --bg-alt: #ffffff;
    --bg-input: #eef0f6;
    --bg-input-alt: #e7e9f2;
    --border: #dadfe8;
    --text: #1c1f2b;
    --text-dim: #5c6480;
    --text-faint: #8a90a8;
    --accent: #3b4fd8;
    --accent-hover: #2f3fc0;
    --bubble-user-bg: #e3e8ff;
    --bubble-user-border: #3b4fd8;
    --green-bg: #d8f5e3; --green-text: #15803d;
    --red-bg: #fde2e2; --red-text: #b91c1c;
    --blue-bg: #dceafe; --blue-text: #1d4ed8;
    --yellow-bg: #fef0d6; --yellow-text: #92600a; --yellow-border: #f0d098;
    --mark-bg: #fff3b0; --mark-text: #6b4e00;
    --code: #0d9488;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
  .sidebar { width: 280px; background: var(--bg-alt); border-right: 1px solid var(--border); padding: 24px 16px; display: flex; flex-direction: column; gap: 16px; flex-shrink: 0; overflow-y: auto; }
  .sidebar h2 { font-size: 14px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }
  .sidebar p { font-size: 13px; color: var(--text-dim); line-height: 1.5; }
  .sidebar code { background: var(--bg-input-alt); color: var(--code); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 10px 14px; font-size: 13px; cursor: pointer; width: 100%; text-align: left; transition: background 0.2s; }
  .btn:hover { background: var(--accent-hover); }
  .btn:disabled { background: var(--bg-input-alt); color: #888; cursor: not-allowed; }
  .btn.secondary { background: var(--bg-input-alt); color: var(--text); }
  .btn.secondary:hover { filter: brightness(1.1); }
  .divider { border: none; border-top: 1px solid var(--border); }
  .how-to { font-size: 13px; color: var(--text-dim); line-height: 1.8; }
  .status { font-size: 12px; padding: 8px 12px; border-radius: 6px; display: none; }
  .status.success { background: var(--green-bg); color: var(--green-text); display: block; }
  .status.error { background: var(--red-bg); color: var(--red-text); display: block; }
  .status.loading { background: var(--blue-bg); color: var(--blue-text); display: block; }
  .user-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 12px; }
  .user-card .name { font-weight: 600; color: var(--text); }
  .user-card .role { color: var(--code); text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; }
  .user-card .links { display: flex; gap: 10px; margin-top: 6px; }
  .user-card a { color: var(--text-dim); text-decoration: none; font-size: 11px; }
  .user-card a.logout-link { color: var(--red-text); }
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .header { padding: 20px 24px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header p { font-size: 13px; color: var(--text-dim); margin-top: 2px; }
  .header-actions { display: flex; gap: 8px; flex-shrink: 0; }
  .pill-btn { background: var(--bg-input-alt); border: 1px solid var(--border); color: var(--text); border-radius: 20px; padding: 7px 13px; font-size: 13px; cursor: pointer; }
  .pill-btn:hover { filter: brightness(1.15); }
  .messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { display: flex; gap: 10px; align-items: flex-start; max-width: 780px; flex-direction: column; }
  .msg.user { align-self: flex-end; }
  .msg-row { display: flex; gap: 10px; align-items: flex-start; }
  .msg.user .msg-row { flex-direction: row-reverse; }
  .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
  .msg.user .avatar { background: var(--accent); }
  .msg.bot .avatar { background: var(--bg-input-alt); }
  .bubble { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; font-size: 14px; line-height: 1.6; max-width: 640px; }
  .bubble mark { background: var(--mark-bg); color: var(--mark-text); padding: 0 2px; border-radius: 3px; }
  .msg.user .bubble { background: var(--bubble-user-bg); border-color: var(--bubble-user-border); }
  .thinking { color: var(--blue-text); font-style: italic; }
  .msg-actions { display: flex; gap: 8px; align-items: center; margin-left: 42px; flex-wrap: wrap; }
  .icon-btn { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 6px; color: var(--text-dim);
              font-size: 12px; padding: 4px 9px; cursor: pointer; }
  .icon-btn:hover { background: var(--bg-input-alt); }
  .icon-btn.active-up { background: var(--green-bg); color: var(--green-text); border-color: var(--green-bg); }
  .icon-btn.active-down { background: var(--red-bg); color: var(--red-text); border-color: var(--red-bg); }
  .icon-btn.escalate { background: var(--yellow-bg); color: var(--yellow-text); border-color: var(--yellow-border); }
  .icon-btn.escalate:hover { filter: brightness(1.1); }
  .confidence-badge { font-size: 11px; padding: 3px 9px; border-radius: 10px; font-weight: 600; }
  .confidence-high { background: var(--green-bg); color: var(--green-text); }
  .confidence-mid { background: var(--yellow-bg); color: var(--yellow-text); }
  .confidence-low { background: var(--red-bg); color: var(--red-text); }
  .tiny-note { font-size: 11px; color: var(--text-faint); margin-left: 42px; }
  .input-bar { padding: 16px 24px; border-top: 1px solid var(--border); display: flex; gap: 10px; }
  .input-bar input { flex: 1; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 14px; padding: 10px 14px; outline: none; }
  .input-bar input:focus { border-color: var(--accent); }
  .input-bar input::placeholder { color: var(--text-faint); }
  .mic-btn { background: var(--bg-input-alt); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 10px 14px; font-size: 15px; cursor: pointer; }
  .mic-btn.listening { background: var(--red-bg); color: var(--red-text); border-color: var(--red-bg); }
  .send-btn { background: var(--accent); border: none; border-radius: 8px; color: #fff; padding: 10px 18px; font-size: 14px; cursor: pointer; }
  .send-btn:disabled { background: var(--bg-input-alt); cursor: not-allowed; }
  .admin-form input, .admin-form select { width: 100%; background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font-size: 13px; padding: 8px 10px; margin-bottom: 10px; outline: none; }
  .admin-form label { font-size: 11px; color: var(--text-dim); display: block; margin-bottom: 4px; }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="user-card">
    <div class="name">__USERNAME__</div>
    <div class="role">__ROLE__</div>
    <div class="links">
      <a href="/profile" data-i18n="profile">My Profile</a>
      <a href="/logout" class="logout-link" data-i18n="logout">Log out</a>
    </div>
  </div>

  __ADMIN_SECTION__

  <hr class="divider">
  <h2 data-i18n="personal_doc_title">My Document</h2>
  <p data-i18n="personal_doc_hint">Upload your own file to ask questions about just that document — private to you, not shared with anyone else.</p>
  <form class="admin-form" id="personalUploadForm" onsubmit="submitPersonalUpload(event)">
    <input type="file" id="personal_doc_file" accept=".pdf,.txt,.docx" required>
    <button class="btn secondary" type="submit" data-i18n="personal_upload_btn">Upload My Document</button>
  </form>
  <div class="status" id="personalUploadStatus"></div>
  <label style="font-size:12px;color:var(--text-dim);display:flex;align-items:center;gap:6px;margin-top:8px;cursor:pointer;">
    <input type="checkbox" id="personalModeToggle" onchange="togglePersonalMode()">
    <span data-i18n="personal_mode_label">Ask about my document instead of HR policy</span>
  </label>

  <hr class="divider">
  <button class="btn secondary" data-i18n="export_btn" onclick="exportChat()">Export conversation (PDF)</button>

  <hr class="divider">
  <div class="how-to">
    <strong style="color:var(--text-dim)" data-i18n="howto_title">How to use:</strong><br>
    <span data-i18n="howto_body">Ask anything about company HR policies — leave, payroll, WFH, code of conduct, etc.
    Every answer shows you can rate with &#128077;/&#128078;, and you can escalate to a human
    HR rep any time. Each answer also shows an estimated confidence score.</span>
  </div>
</aside>
<main class="main">
  <div class="header">
    <div>
      <h1 data-i18n="title">HR Policy Assistant</h1>
      <p data-i18n="subtitle">Ask questions about company HR policies.</p>
    </div>
    <div class="header-actions">
      <button class="pill-btn" id="langBtn" onclick="toggleLanguage()" title="Tamil / English">த</button>
      <button class="pill-btn" id="themeBtn" onclick="toggleTheme()" title="Toggle dark/light theme">&#127769;</button>
    </div>
  </div>
  <div class="messages" id="messages">
    __HISTORY_MESSAGES__
    <div class="msg bot">
      <div class="msg-row">
        <div class="avatar">&#129302;</div>
        <div class="bubble">Hello __USERNAME__! Ask me anything about HR policy — leave, payroll, WFH, conduct, and more.</div>
      </div>
    </div>
  </div>
  <div class="input-bar">
    <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="Voice input">&#127908;</button>
    <input type="text" id="userInput" data-i18n-placeholder="placeholder" placeholder="Ask a question about HR policy..." onkeydown="if(event.key==='Enter') sendMsg()">
    <button class="send-btn" id="sendBtn" data-i18n="send" onclick="sendMsg()">Send</button>
  </div>
</main>
<script>
function escHtml(t) {
  return t.split('&').join('&amp;').split('<').join('&lt;').split('>').join('&gt;').split('\n').join('<br>');
}

var buildPoll = null;

// ────────────── NEW: Theme toggle (dark/light, saved to localStorage) ──────────────
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('hr_theme', theme);
  document.getElementById('themeBtn').textContent = theme === 'light' ? '\u{1F319}' : '\u{1F305}';
}
function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}
(function initTheme() {
  applyTheme(localStorage.getItem('hr_theme') || 'dark');
})();

// ────────────── NEW: Tamil / English UI toggle (static strings only) ──────────────
var TRANSLATIONS = {
  en: {
    title: "HR Policy Assistant",
    subtitle: "Ask questions about company HR policies.",
    placeholder: "Ask a question about HR policy...",
    send: "Send",
    howto_title: "How to use:",
    howto_body: "Ask anything about company HR policies \u2014 leave, payroll, WFH, code of conduct, etc. Every answer shows you can rate with \uD83D\uDC4D/\uD83D\uDC4E, and you can escalate to a human HR rep any time. Each answer also shows an estimated confidence score.",
    export_btn: "Export conversation (PDF)",
    kb_title: "Knowledge Base",
    build_btn: "Build / Rebuild Knowledge Base",
    analytics_btn: "View Analytics Dashboard",
    adduser_title: "Add HR Admin / Team Member",
    logout: "Log out",
    profile: "My Profile",
    upload_hint: "Upload a new HR policy file (PDF / TXT / DOCX) below — no need to touch the server's files directly.",
    upload_btn: "Upload Document",
    personal_doc_title: "My Document",
    personal_doc_hint: "Upload your own file to ask questions about just that document — private to you, not shared with anyone else.",
    personal_upload_btn: "Upload My Document",
    personal_mode_label: "Ask about my document instead of HR policy"
  },
  ta: {
    title: "\u0BAE\u0BA9\u0BBF\u0BA4\u0BB5\u0BB3 \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8 \u0B89\u0BA4\u0BB5\u0BBF\u0BAF\u0BBE\u0BB3\u0BB0\u0BCD",
    subtitle: "\u0BA8\u0BBF\u0BB1\u0BC1\u0BB5\u0BA9 HR \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8\u0B95\u0BB3\u0BCD \u0BAA\u0BB1\u0BCD\u0BB1\u0BBF \u0B95\u0BC7\u0BB3\u0BCD\u0BB5\u0BBF\u0B95\u0BB3\u0BCD \u0B95\u0BC7\u0BB3\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD.",
    placeholder: "HR \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8 \u0BAA\u0BB1\u0BCD\u0BB1\u0BBF \u0B92\u0BB0\u0BC1 \u0B95\u0BC7\u0BB3\u0BCD\u0BB5\u0BBF \u0B95\u0BC7\u0BB3\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD...",
    send: "\u0B85\u0BA9\u0BC1\u0BAA\u0BCD\u0BAA\u0BC1",
    howto_title: "\u0BAA\u0BAF\u0BA9\u0BCD\u0BAA\u0B9F\u0BC1\u0BA4\u0BCD\u0BA4\u0BC1\u0BAE\u0BCD \u0BAE\u0BC1\u0BB1\u0BC8:",
    howto_body: "\u0BB5\u0BBF\u0B9F\u0BC1\u0BAA\u0BCD\u0BAA\u0BC1, \u0B9A\u0BAE\u0BCD\u0BAA\u0BB3\u0BAE\u0BCD, WFH, \u0BA8\u0B9F\u0BA4\u0BCD\u0BA4\u0BC8 \u0BB5\u0BBF\u0BA4\u0BBF\u0B95\u0BB3\u0BCD \u0BAA\u0BCB\u0BA9\u0BCD\u0BB1 HR \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8\u0B95\u0BB3\u0BCD \u0BAA\u0BB1\u0BCD\u0BB1\u0BBF \u0B8E\u0BA4\u0BC8\u0BAF\u0BC1\u0BAE\u0BCD \u0B95\u0BC7\u0BB3\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD. \u0BA4\u0BC7\u0BB0\u0BCD\u0BB5\u0BC1 \u0B9A\u0BBF\u0BA8\u0BCD\u0BA4\u0BBF\u0BB5\u0BC1\u0BA4\u0BCD\u0BA4\u0BC8 \u0BAA\u0BAF\u0BA9\u0BCD\u0BAA\u0B9F\u0BC1\u0BA4\u0BCD\u0BA4\u0BBF \u0B85\u0BA4\u0BC8 \u0BAE\u0BA4\u0BBF\u0BAA\u0BCD\u0BAA\u0BC0\u0B9F\u0BC1 \u0B9A\u0BC6\u0BAF\u0BCD\u0BAF\u0BB2\u0BBE\u0BAE\u0BCD, \u0B8E\u0BAA\u0BCD\u0BAA\u0BCB\u0BA4\u0BC1\u0BAE\u0BCD HR-\u0B95\u0BCD\u0B95\u0BC1 \u0B85\u0BA9\u0BC1\u0BAA\u0BCD\u0BAA\u0BB2\u0BBE\u0BAE\u0BCD.",
    export_btn: "\u0B89\u0BB0\u0BC8\u0BAF\u0BBE\u0B9F\u0BB2\u0BC8 \u0B8E\u0B95\u0BCD\u0BB8\u0BCD\u0BAA\u0BCB\u0BB0\u0BCD\u0B9F\u0BCD \u0B9A\u0BC6\u0BAF\u0BCD (PDF)",
    kb_title: "\u0B85\u0BB1\u0BBF\u0BB5\u0BC1\u0BA4\u0BCD \u0BA4\u0BB3\u0BAE\u0BCD",
    build_btn: "\u0B85\u0BB1\u0BBF\u0BB5\u0BC1\u0BA4\u0BCD \u0BA4\u0BB3\u0BA4\u0BCD\u0BA4\u0BC8 \u0B89\u0BB0\u0BC1\u0BB5\u0BBE\u0B95\u0BCD\u0B95\u0BC1 / \u0BAA\u0BC1\u0BA4\u0BC1\u0BAA\u0BCD\u0BAA\u0BBF",
    analytics_btn: "\u0BAA\u0B95\u0BC1\u0BAA\u0BCD\u0BAA\u0BBE\u0BAF\u0BCD\u0BB5\u0BC1 \u0B9F\u0BBE\u0BB7\u0BCD\u0BAA\u0BCB\u0BB0\u0BCD\u0B9F\u0BC8\u0BAA\u0BCD \u0BAA\u0BBE\u0BB0\u0BCD",
    adduser_title: "HR \u0B85\u0B9F\u0BCD\u0BAE\u0BBF\u0BA9\u0BCD / \u0B9F\u0BC0\u0BAE\u0BCD \u0B89\u0BAA\u0BAA\u0BB0\u0BCD \u0B9A\u0BC7\u0BB0\u0BCD",
    logout: "\u0BB5\u0BC6\u0BB3\u0BBF\u0BAF\u0BC7\u0BB1\u0BC1",
    profile: "\u0B8E\u0BA9\u0BCD \u0B9A\u0BC1\u0BAF\u0BB5\u0BBF\u0BB5\u0BB0\u0BAE\u0BCD",
    upload_hint: "\u0BAA\u0BC1\u0BA4\u0BBF\u0BAF HR \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8 \u0B95\u0BCB\u0BAA\u0BCD\u0BAA\u0BC8 (PDF / TXT / DOCX) \u0B95\u0BC0\u0BB4\u0BC7 \u0BAA\u0BA4\u0BBF\u0BB5\u0BC7\u0BB1\u0BCD\u0BB1\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD \u2014 \u0B9A\u0BB0\u0BCD\u0BB5\u0BB0\u0BCD \u0B95\u0BCB\u0BAA\u0BCD\u0BAA\u0BC1\u0B95\u0BB3\u0BC8 \u0BA8\u0BC7\u0BB0\u0B9F\u0BBF\u0BAF\u0BBE\u0B95 \u0BA4\u0BCA\u0B9F\u0BA4\u0BCD \u0BA4\u0BC7\u0BB5\u0BC8\u0BAF\u0BBF\u0BB2\u0BCD\u0BB2\u0BC8.",
    upload_btn: "\u0B86\u0BB5\u0BA3\u0BA4\u0BCD\u0BA4\u0BC8 \u0BAA\u0BA4\u0BBF\u0BB5\u0BC7\u0BB1\u0BCD\u0BB1\u0BC1",
    personal_doc_title: "\u0B8E\u0BA9\u0BCD \u0B86\u0BB5\u0BA3\u0BAE\u0BCD",
    personal_doc_hint: "\u0B89\u0B99\u0BCD\u0B95\u0BB3\u0BCD \u0B9A\u0BCA\u0BA8\u0BCD\u0BA4 \u0B95\u0BCB\u0BAA\u0BCD\u0BAA\u0BC8 \u0BAA\u0BA4\u0BBF\u0BB5\u0BC7\u0BB1\u0BCD\u0BB1\u0BBF, \u0B85\u0BA8\u0BCD\u0BA4 \u0B86\u0BB5\u0BA3\u0BA4\u0BCD\u0BA4\u0BC8 \u0BAA\u0BB1\u0BCD\u0BB1\u0BBF \u0BAE\u0B9F\u0BCD\u0B9F\u0BC1\u0BAE\u0BCD \u0B95\u0BC7\u0BB3\u0BCD\u0BB5\u0BBF\u0B95\u0BB3\u0BCD \u0B95\u0BC7\u0BB3\u0BC1\u0B99\u0BCD\u0B95\u0BB3\u0BCD \u2014 \u0B87\u0BA4\u0BC1 \u0B89\u0B99\u0BCD\u0B95\u0BB3\u0BC1\u0B95\u0BCD\u0B95\u0BC1 \u0BAE\u0B9F\u0BCD\u0B9F\u0BC1\u0BAE\u0BCD, \u0BAF\u0BBE\u0BB0\u0BC1\u0B9F\u0BA9\u0BC1\u0BAE\u0BCD \u0BAA\u0B95\u0BBF\u0BB0\u0BAA\u0BCD\u0BAA\u0B9F\u0BBE\u0BA4\u0BC1.",
    personal_upload_btn: "\u0B8E\u0BA9\u0BCD \u0B86\u0BB5\u0BA3\u0BA4\u0BCD\u0BA4\u0BC8 \u0BAA\u0BA4\u0BBF\u0BB5\u0BC7\u0BB1\u0BCD\u0BB1\u0BC1",
    personal_mode_label: "HR \u0B95\u0BCA\u0BB3\u0BCD\u0B95\u0BC8\u0B95\u0BCD\u0B95\u0BC1 \u0BAA\u0BA4\u0BBF\u0BB2\u0BBE\u0B95 \u0B8E\u0BA9\u0BCD \u0B86\u0BB5\u0BA3\u0BA4\u0BCD\u0BA4\u0BC8 \u0BAA\u0BB1\u0BCD\u0BB1\u0BBF \u0B95\u0BC7\u0BB3\u0BCD"
  }
};
var currentLang = localStorage.getItem('hr_lang') || 'en';
function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('hr_lang', lang);
  var dict = TRANSLATIONS[lang] || TRANSLATIONS.en;
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    if (dict[key]) el.textContent = dict[key];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
    var key = el.getAttribute('data-i18n-placeholder');
    if (dict[key]) el.placeholder = dict[key];
  });
  document.getElementById('langBtn').textContent = lang === 'en' ? '\u0BA4' : 'EN';
}
function toggleLanguage() {
  applyLanguage(currentLang === 'en' ? 'ta' : 'en');
}
applyLanguage(currentLang);

// ────────────── NEW: Voice input (browser Web Speech API) ──────────────
var recognition = null;
var listening = false;
(function initVoiceInput() {
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.onresult = function(e) {
    document.getElementById('userInput').value = e.results[0][0].transcript;
  };
  recognition.onend = function() {
    listening = false;
    document.getElementById('micBtn').classList.remove('listening');
  };
  recognition.onerror = function() {
    listening = false;
    document.getElementById('micBtn').classList.remove('listening');
  };
})();
function toggleMic() {
  if (!recognition) {
    alert('Voice input is not supported in this browser. Try Chrome or Edge.');
    return;
  }
  if (listening) {
    recognition.stop();
    return;
  }
  recognition.lang = currentLang === 'ta' ? 'ta-IN' : 'en-IN';
  listening = true;
  document.getElementById('micBtn').classList.add('listening');
  recognition.start();
}

// ────────────── NEW: Voice output / TTS (browser Web Speech API) ──────────────
function speakText(text) {
  if (!('speechSynthesis' in window)) {
    alert('Voice output is not supported in this browser.');
    return;
  }
  window.speechSynthesis.cancel();
  var utter = new SpeechSynthesisUtterance(text);
  utter.lang = currentLang === 'ta' ? 'ta-IN' : 'en-IN';
  window.speechSynthesis.speak(utter);
}

// ────────────── NEW: Confidence badge helper ──────────────
function confidenceBadge(score) {
  var cls = score >= 70 ? 'confidence-high' : (score >= 40 ? 'confidence-mid' : 'confidence-low');
  return '<span class="confidence-badge ' + cls + '">' + score + '% confidence</span>';
}

function buildKB() {
  var btn = document.getElementById('buildBtn');
  var status = document.getElementById('buildStatus');
  btn.disabled = true;
  status.className = 'status loading';
  status.textContent = 'Starting build...';
  fetch('/build_start', { method: 'POST' }).then(function() {
    buildPoll = setInterval(checkBuildStatus, 2000);
  }).catch(function(e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
    btn.disabled = false;
  });
}

function checkBuildStatus() {
  fetch('/build_status').then(function(r) { return r.json(); }).then(function(data) {
    var status = document.getElementById('buildStatus');
    var btn = document.getElementById('buildBtn');
    if (data.running) {
      status.className = 'status loading';
      status.textContent = data.message;
    } else if (data.success === true) {
      clearInterval(buildPoll);
      status.className = 'status success';
      status.textContent = 'Done! ' + data.message;
      btn.disabled = false;
    } else if (data.success === false) {
      clearInterval(buildPoll);
      status.className = 'status error';
      status.textContent = 'Error: ' + data.message;
      btn.disabled = false;
    }
  });
}

function exportChat() {
  window.location.href = '/export';
}

function sendFeedback(btnUp, btnDown, question, answer, rating) {
  btnUp.disabled = true;
  btnDown.disabled = true;
  if (rating === 'up') { btnUp.classList.add('active-up'); }
  else { btnDown.classList.add('active-down'); }

  fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question, answer: answer, rating: rating })
  }).catch(function(e) { console.error('Feedback failed', e); });
}

function escalate(btn, question) {
  btn.disabled = true;
  btn.textContent = 'Escalating...';
  fetch('/escalate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question, reason: 'flagged_by_bot' })
  }).then(function(r) { return r.json(); }).then(function(data) {
    btn.textContent = 'Escalated (' + data.ticket_id + ')';
  }).catch(function(e) {
    btn.textContent = 'Escalate to HR';
    btn.disabled = false;
  });
}

// ────────────── NEW: Personal document upload + mode toggle ──────────────
var personalDocActive = false;

function submitPersonalUpload(event) {
  event.preventDefault();
  var status = document.getElementById('personalUploadStatus');
  var fileInput = document.getElementById('personal_doc_file');
  if (!fileInput.files.length) return;

  var formData = new FormData();
  formData.append('file', fileInput.files[0]);

  status.className = 'status loading';
  status.textContent = 'Uploading and indexing... (can take ~10-20s)';

  fetch('/personal_upload', { method: 'POST', body: formData })
    .then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
    .then(function(res) {
      if (res.ok) {
        status.className = 'status success';
        status.textContent = res.data.message;
        document.getElementById('personalModeToggle').checked = true;
        personalDocActive = true;
      } else {
        status.className = 'status error';
        status.textContent = res.data.error || 'Upload failed.';
      }
    }).catch(function(e) {
      status.className = 'status error';
      status.textContent = 'Error: ' + e.message;
    });
}

function togglePersonalMode() {
  personalDocActive = document.getElementById('personalModeToggle').checked;
}

function submitUploadDoc(event) {
  event.preventDefault();
  var status = document.getElementById('uploadDocStatus');
  var fileInput = document.getElementById('doc_file');
  if (!fileInput.files.length) return;

  var formData = new FormData();
  formData.append('file', fileInput.files[0]);

  status.className = 'status loading';
  status.textContent = 'Uploading...';

  fetch('/admin/upload_document', {
    method: 'POST',
    body: formData
  }).then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
    .then(function(res) {
      if (res.ok) {
        status.className = 'status success';
        status.textContent = res.data.message + ' Now click "Build / Rebuild Knowledge Base" to add it.';
        fileInput.value = '';
      } else {
        status.className = 'status error';
        status.textContent = res.data.error || 'Upload failed.';
      }
    }).catch(function(e) {
      status.className = 'status error';
      status.textContent = 'Error: ' + e.message;
    });
}

function submitAddUser(event) {
  event.preventDefault();
  var status = document.getElementById('addUserStatus');
  var form = document.getElementById('addUserForm');
  var payload = {
    username: document.getElementById('au_username').value.trim(),
    password: document.getElementById('au_password').value,
    role: document.getElementById('au_role').value
  };
  status.className = 'status loading';
  status.textContent = 'Creating account...';

  fetch('/admin/add_user', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(function(r) { return r.json().then(function(data) { return { ok: r.ok, data: data }; }); })
    .then(function(res) {
      if (res.ok) {
        status.className = 'status success';
        status.textContent = res.data.message;
        form.reset();
      } else {
        status.className = 'status error';
        status.textContent = res.data.error || 'Could not create account.';
      }
    }).catch(function(e) {
      status.className = 'status error';
      status.textContent = 'Error: ' + e.message;
    });
}

function sendMsg() {
  var input = document.getElementById('userInput');
  var sendBtn = document.getElementById('sendBtn');
  var q = input.value.trim();
  if (!q) return;
  input.value = '';
  sendBtn.disabled = true;

  var usingPersonalDoc = personalDocActive;
  var modeTag = usingPersonalDoc ? '<span style="font-size:11px;opacity:0.7">&#128196; ' : '';
  var modeTagEnd = usingPersonalDoc ? '</span><br>' : '';

  var msgs = document.getElementById('messages');
  msgs.innerHTML += '<div class="msg user"><div class="msg-row"><div class="avatar">&#128100;</div><div class="bubble">' + modeTag + (usingPersonalDoc ? 'My Document' : '') + modeTagEnd + escHtml(q) + '</div></div></div>';
  var thinkId = 'think_' + Date.now();
  msgs.innerHTML += '<div class="msg bot" id="' + thinkId + '"><div class="msg-row"><div class="avatar">&#129302;</div><div class="bubble thinking">Thinking...</div></div></div>';
  msgs.scrollTop = msgs.scrollHeight;

  var startUrl = usingPersonalDoc ? '/personal_chat_start' : '/chat_start';

  fetch(startUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: q })
  }).then(function(r) { return r.json(); }).then(function(data) {
    var chatId = data.chat_id;
    var poll = setInterval(function() {
      fetch('/chat_result?id=' + chatId).then(function(r) { return r.json(); }).then(function(res) {
        if (res.ready) {
          clearInterval(poll);

          var actionsId = thinkId + '_actions';
          var escalateHtml = '';
          if (res.escalate) {
            var safeQuestion = encodeURIComponent(q);

escalateHtml =
  '<button class="icon-btn escalate" data-question="' +
  safeQuestion +
  '">Escalate to HR</button>';
  setTimeout(function () {
  document.querySelectorAll('.icon-btn.escalate').forEach(function(btn) {
    if (!btn.dataset.bound) {
      btn.dataset.bound = "1";

      btn.addEventListener('click', function() {
        escalate(
          this,
          decodeURIComponent(this.dataset.question)
        );
      });
    }
  });
}, 0);
          }

          var html =
            '<div class="msg bot">' +
              '<div class="msg-row"><div class="avatar">&#129302;</div><div class="bubble">' + res.answer_html + '</div></div>' +
              '<div class="msg-actions" id="' + actionsId + '">' +
                confidenceBadge(res.confidence) +
                '<button class="icon-btn" id="' + actionsId + '_up">&#128077;</button>' +
                '<button class="icon-btn" id="' + actionsId + '_down">&#128078;</button>' +
                '<button class="icon-btn" id="' + actionsId + '_speak">&#128266;</button>' +
                escalateHtml +
              '</div>' +
            '</div>';

          document.getElementById(thinkId).outerHTML = html;

          var upBtn = document.getElementById(actionsId + '_up');
          var downBtn = document.getElementById(actionsId + '_down');
          var speakBtn = document.getElementById(actionsId + '_speak');
          upBtn.onclick = function() { sendFeedback(upBtn, downBtn, q, res.answer, 'up'); };
          downBtn.onclick = function() { sendFeedback(upBtn, downBtn, q, res.answer, 'down'); };
          speakBtn.onclick = function() { speakText(res.answer); };

          msgs.scrollTop = msgs.scrollHeight;
          sendBtn.disabled = false;
          input.focus();
        }
      });
    }, 1500);
  }).catch(function(e) {
    document.getElementById(thinkId).outerHTML = '<div class="msg bot"><div class="msg-row"><div class="avatar">&#129302;</div><div class="bubble" style="color:var(--red-text)">Error: ' + e.message + '</div></div></div>';
    sendBtn.disabled = false;
  });
}
</script>
</body>
</html>"""

ADMIN_SECTION_HTML = r"""
  <hr class="divider">
  <h2 data-i18n="kb_title">Knowledge Base</h2>
  <p data-i18n="upload_hint">Upload a new HR policy file (PDF / TXT / DOCX) below — no need to touch the server's files directly.</p>
  <form class="admin-form" id="uploadDocForm" onsubmit="submitUploadDoc(event)">
    <input type="file" id="doc_file" accept=".pdf,.txt,.docx" required>
    <button class="btn" type="submit" data-i18n="upload_btn">Upload Document</button>
  </form>
  <div class="status" id="uploadDocStatus"></div>
  <button class="btn secondary" id="buildBtn" data-i18n="build_btn" onclick="buildKB()">Build / Rebuild Knowledge Base</button>
  <div class="status" id="buildStatus"></div>
  <hr class="divider">
  <a href="/analytics" style="text-decoration:none">
    <button class="btn secondary" data-i18n="analytics_btn">View Analytics Dashboard</button>
  </a>
  <hr class="divider">
  <h2 data-i18n="adduser_title">Add HR Admin / Team Member</h2>
  <p>Only HR Admins can grant the HR Admin role to someone.</p>
  <form class="admin-form" id="addUserForm" onsubmit="submitAddUser(event)">
    <label for="au_username">New username</label>
    <input type="text" id="au_username" required>
    <label for="au_password">Temporary password</label>
    <input type="password" id="au_password" required>
    <label for="au_role">Role</label>
    <select id="au_role" required>
      __ADMIN_ROLE_OPTIONS__
    </select>
    <button class="btn" type="submit">Create Account</button>
  </form>
  <div class="status" id="addUserStatus"></div>
"""


@app.route("/")
@login_required
def index():
    username = session["username"]
    role = session["role"]

    if role == "hr_admin":
        admin_role_options = "".join(
            f'<option value="{r}">{r.replace("_", " ").title()}</option>'
            for r in VALID_ROLES
        )
        admin_section = ADMIN_SECTION_HTML.replace("__ADMIN_ROLE_OPTIONS__", admin_role_options)
    else:
        admin_section = ""

    # NEW: render any previously-saved messages so chat history resumes
    # across logins/restarts instead of always starting empty.
    history_html = render_history_messages(session.get("history", []))

    html = (
        CHAT_HTML_TEMPLATE
        .replace("__USERNAME__", username)
        .replace("__ROLE__", role)
        .replace("__ADMIN_SECTION__", admin_section)
        .replace("__HISTORY_MESSAGES__", history_html)
    )
    return html


# ──────────────────────────────────────────────
# Knowledge base build routes (hr_admin only)
# ──────────────────────────────────────────────
@app.route("/build_start", methods=["POST"])
@role_required("hr_admin")
def build_start():
    global _build_status
    if _build_status["running"]:
        return jsonify({"ok": True})
    t = threading.Thread(target=do_build)
    t.daemon = True
    t.start()
    return jsonify({"ok": True})


@app.route("/build_status")
@role_required("hr_admin")
def build_status():
    return jsonify(_build_status)


# ──────────────────────────────────────────────
# Admin: add any-role user (hr_admin only)
# ──────────────────────────────────────────────
@app.route("/admin/add_user", methods=["POST"])
@role_required("hr_admin")
def admin_add_user():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    role = data.get("role") or ""

    try:
        add_user(username, password, role)
        return jsonify({
            "message": f"Created '{username}' as {role.replace('_', ' ').title()}."
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────
# NEW: Admin: upload a document straight into the shared knowledge base
# (replaces having to manually drop files into data/documents/ via the
# file system / VS Code - hr_admin still needs to click "Build / Rebuild
# Knowledge Base" afterwards to actually index the new file).
# ──────────────────────────────────────────────
@app.route("/admin/upload_document", methods=["POST"])
@role_required("hr_admin")
def admin_upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, DOCX."}), 400

    safe_name = _secure_filename(file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest_path = os.path.join(DATA_DIR, safe_name)

    # Don't silently overwrite an existing document with the same name -
    # append a short random suffix instead.
    if os.path.exists(dest_path):
        base, dot_ext = os.path.splitext(safe_name)
        dest_path = os.path.join(DATA_DIR, f"{base}_{os.urandom(3).hex()}{dot_ext}")

    file.save(dest_path)

    print(f"[app] HR admin '{session['username']}' uploaded new document: {os.path.basename(dest_path)}")

    return jsonify({
        "message": f"'{os.path.basename(dest_path)}' uploaded successfully."
    })


# ──────────────────────────────────────────────
# NEW: Personal document upload + Q&A (any logged-in role)
# Lets ANY user (employee/manager/hr_admin) upload their OWN file and
# ask questions about just that document. This is completely separate
# from the shared HR knowledge base - it's never written to
# data/documents/, never indexed into the shared vector store, and
# never visible to anyone else. Each user can have only ONE active
# personal document at a time (uploading a new one replaces the old).
# ──────────────────────────────────────────────
@app.route("/personal_upload", methods=["POST"])
@login_required
def personal_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in PERSONAL_LOADER_MAP:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, DOCX."}), 400

    username = session["username"]
    safe_username = "".join(c for c in username if c.isalnum() or c in ("-", "_")) or "user"
    user_dir = os.path.join(PERSONAL_UPLOAD_DIR, safe_username)
    os.makedirs(user_dir, exist_ok=True)

    # Only one personal document per user at a time - clear out any
    # previous upload before saving the new one.
    for old_file in os.listdir(user_dir):
        try:
            os.remove(os.path.join(user_dir, old_file))
        except OSError:
            pass

    safe_name = _secure_filename(file.filename)
    dest_path = os.path.join(user_dir, safe_name)
    file.save(dest_path)

    try:
        docs = _load_single_file(dest_path)
        if not docs:
            return jsonify({"error": "Could not extract any text from this file."}), 400

        chunks = split_documents(docs)
        embeddings = _personal_embedding_function()
        vector_store = FAISS.from_documents(chunks, embeddings)
        chain = build_rag_chain(vector_store)

        with _personal_chain_lock:
            _personal_chains[username] = chain

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 500

    print(f"[app] '{username}' uploaded a personal document: {safe_name}")

    return jsonify({
        "message": f"'{safe_name}' indexed! Toggle 'Ask about my document' below to query it."
    })


@app.route("/personal_chat_start", methods=["POST"])
@login_required
def personal_chat_start():
    data = request.get_json()
    question = data.get("question", "").strip()
    chat_id = str(id(question)) + os.urandom(4).hex()
    username = session["username"]

    do_personal_chat(chat_id, question, username)

    return jsonify({"chat_id": chat_id})


# ──────────────────────────────────────────────
# Chat routes (any logged-in role)
# ──────────────────────────────────────────────
@app.route("/chat_start", methods=["POST"])
@login_required
def chat_start():
    data = request.get_json()
    question = data.get("question", "").strip()
    chat_id = str(id(question)) + os.urandom(4).hex()

    username = session["username"]
    role = session["role"]

    # Run directly (no thread) - Windows grpc fix, kept from original design
    do_chat(chat_id, question, username, role)

    # Save this exchange into the session history so it can be exported later
    with _chat_lock:
        result = _chat_results.get(chat_id, {})
    answer = result.get("answer", "")

    history = session.get("history", [])
    history.append({"role": "user", "content": question})
    history.append({"role": "bot", "content": answer})
    session["history"] = history

    return jsonify({"chat_id": chat_id})


@app.route("/chat_result")
@login_required
def chat_result():
    chat_id = request.args.get("id", "")
    with _chat_lock:
        if chat_id in _chat_results:
            result = _chat_results.pop(chat_id)
            return jsonify({
                "ready": True,
                "answer": result["answer"],
                "answer_html": result["answer_html"],
                "confidence": result["confidence"],
                "escalate": result["escalate"],
            })
    return jsonify({"ready": False})


# ──────────────────────────────────────────────
# Feedback route
# ──────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
@login_required
def feedback_route():
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")
    rating = data.get("rating", "")

    try:
        log_feedback(question, answer, rating, session["username"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"ok": True})


# ──────────────────────────────────────────────
# Escalation route
# ──────────────────────────────────────────────
@app.route("/escalate", methods=["POST"])
@login_required
def escalate_route():
    data = request.get_json()
    question = data.get("question", "")
    reason = data.get("reason", "user_requested")

    ticket = create_escalation_ticket(question, session["username"], reason)
    # NEW: also try to send a real email to HR via EmailJS (no-ops if not configured)
    send_escalation_email(ticket)
    return jsonify(ticket)


# ──────────────────────────────────────────────
# Export route
# ──────────────────────────────────────────────
@app.route("/export")
@login_required
def export_route():
    history = session.get("history", [])
    if not history:
        return "No conversation yet to export.", 400

    username = session["username"]
    safe_username = "".join(c for c in username if c.isalnum()) or "user"
    filename = os.path.join(EXPORT_DIR, f"chat_{safe_username}_{os.urandom(4).hex()}.pdf")

    export_conversation_to_pdf(history, filename)

    return send_file(filename, as_attachment=True, download_name="hr_chat_conversation.pdf")


# ──────────────────────────────────────────────
# NEW: Raw chat history JSON (per logged-in user)
# ──────────────────────────────────────────────
@app.route("/history")
@login_required
def history_route():
    return jsonify(load_chat_history(session["username"]))


# ──────────────────────────────────────────────
# NEW: User profile page
# ──────────────────────────────────────────────
@app.route("/profile")
@login_required
def profile_page():
    username = session["username"]
    role = session["role"]
    history = load_chat_history(username)
    total_questions = len(history)
    last_active = history[-1]["timestamp"] if history else "No activity yet"

    recent_rows = "".join(
        f"<tr><td>{esc_html(h['question'])}</td><td>{esc_html(h['timestamp'])}</td></tr>"
        for h in reversed(history[-10:])
    ) or "<tr><td colspan='2'>No questions asked yet.</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>My Profile</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1117; color:#e0e0e0; padding:30px; }}
  a {{ color:#7dd3b8; text-decoration:none; }}
  .card {{ background:#1a1d27; border:1px solid #2d3144; border-radius:12px; padding:24px; max-width:560px; margin-top:16px; }}
  .role-pill {{ display:inline-block; background:#252840; color:#7dd3b8; font-size:11px; text-transform:uppercase;
                letter-spacing:0.05em; padding:4px 10px; border-radius:12px; margin-top:4px; }}
  .stats {{ display:flex; gap:16px; margin: 18px 0; }}
  .stat-card {{ background:#0f1117; border:1px solid #2d3144; border-radius:10px; padding:14px 18px; flex:1; }}
  .stat-card .num {{ font-size:22px; font-weight:700; }}
  .stat-card .label {{ font-size:11px; color:#7a82a0; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #2d3144; font-size:13px; }}
  th {{ color:#a0a8c0; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size:14px; color:#a0a8c0; text-transform:uppercase; letter-spacing:0.05em; margin-top:24px; margin-bottom:8px; }}
</style></head>
<body>
  <a href="/">&larr; Back to chat</a>
  <h1>My Profile</h1>
  <div class="card">
    <div style="font-size:18px;font-weight:600">{esc_html(username)}</div>
    <div class="role-pill">{esc_html(role)}</div>
    <div class="stats">
      <div class="stat-card"><div class="num">{total_questions}</div><div class="label">Questions Asked</div></div>
      <div class="stat-card"><div class="num" style="font-size:14px">{esc_html(str(last_active))}</div><div class="label">Last Active</div></div>
    </div>
    <h2>Recent Questions</h2>
    <table><tr><th>Question</th><th>When</th></tr>{recent_rows}</table>
  </div>
</body></html>"""
    return html


# ──────────────────────────────────────────────
# Analytics dashboard (hr_admin only)
# ──────────────────────────────────────────────
@app.route("/analytics/data")
@role_required("hr_admin")
def analytics_data():
    """NEW: small JSON feed used by the Plotly charts on /analytics."""
    top_questions = get_top_questions(10)
    feedback_stats = get_feedback_stats()
    open_tickets = get_open_tickets()
    reason_counts = Counter(t.get("reason", "unknown") for t in open_tickets)

    return jsonify({
        "top_questions": {
            "labels": [q["question"][:40] for q in top_questions],
            "values": [q["count"] for q in top_questions],
        },
        "feedback": {
            "positive_pct": feedback_stats["positive_pct"],
            "negative_pct": round(100 - feedback_stats["positive_pct"], 1),
        },
        "tickets_by_reason": {
            "labels": list(reason_counts.keys()),
            "values": list(reason_counts.values()),
        },
    })


@app.route("/analytics")
@role_required("hr_admin")
def analytics_route():
    top_questions = get_top_questions(10)
    query_stats = get_query_stats()
    feedback_stats = get_feedback_stats()
    open_tickets = get_open_tickets()

    rows_top_q = "".join(
        f"<tr><td>{q['question']}</td><td>{q['count']}</td></tr>" for q in top_questions
    ) or "<tr><td colspan='2'>No queries logged yet.</td></tr>"

    rows_neg = "".join(
        f"<tr><td>{n['question']}</td><td>{n['username']}</td><td>{n['timestamp']}</td></tr>"
        for n in feedback_stats["negative_examples"]
    ) or "<tr><td colspan='3'>No negative feedback yet.</td></tr>"

    rows_tickets = "".join(
        f"<tr><td>{t['ticket_id']}</td><td>{t['username']}</td><td>{t['question']}</td><td>{t['reason']}</td></tr>"
        for t in open_tickets
    ) or "<tr><td colspan='4'>No open escalation tickets.</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HR Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1117; color:#e0e0e0; padding:30px; }}
  h1 {{ margin-bottom: 4px; }}
  a {{ color:#7dd3b8; text-decoration:none; }}
  .stats {{ display:flex; gap:16px; margin: 20px 0; }}
  .stat-card {{ background:#1a1d27; border:1px solid #2d3144; border-radius:10px; padding:16px 22px; }}
  .stat-card .num {{ font-size:24px; font-weight:700; }}
  .stat-card .label {{ font-size:12px; color:#7a82a0; }}
  table {{ width:100%; border-collapse: collapse; margin-bottom: 30px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #2d3144; font-size:13px; }}
  th {{ color:#a0a8c0; }}
  h2 {{ font-size:15px; color:#a0a8c0; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px; }}
  .charts {{ display:flex; flex-wrap:wrap; gap:18px; margin-bottom: 30px; }}
  .chart-box {{ background:#1a1d27; border:1px solid #2d3144; border-radius:10px; padding:10px; flex:1; min-width:320px; }}
  .live-tag {{ font-size:11px; color:#4ade80; }}
</style></head>
<body>
  <a href="/">&larr; Back to chat</a>
  <h1>HR Analytics Dashboard</h1>
  <p class="live-tag">&#9679; Live - charts refresh every 10 seconds</p>
  <div class="stats">
    <div class="stat-card"><div class="num">{query_stats['total_queries']}</div><div class="label">Total Queries</div></div>
    <div class="stat-card"><div class="num">{query_stats['unique_users']}</div><div class="label">Unique Users</div></div>
    <div class="stat-card"><div class="num">{feedback_stats['positive_pct']}%</div><div class="label">Positive Feedback</div></div>
    <div class="stat-card"><div class="num">{len(open_tickets)}</div><div class="label">Open Escalations</div></div>
  </div>

  <div class="charts">
    <div class="chart-box" id="chartTopQ" style="height:340px"></div>
    <div class="chart-box" id="chartFeedback" style="height:340px"></div>
    <div class="chart-box" id="chartTickets" style="height:340px"></div>
  </div>

  <h2>Top Asked Questions</h2>
  <table><tr><th>Question</th><th>Times Asked</th></tr>{rows_top_q}</table>

  <h2>Recent Negative Feedback</h2>
  <table><tr><th>Question</th><th>User</th><th>When</th></tr>{rows_neg}</table>

  <h2>Open Escalation Tickets</h2>
  <table><tr><th>Ticket</th><th>User</th><th>Question</th><th>Reason</th></tr>{rows_tickets}</table>

  <script>
    var darkLayout = {{
      paper_bgcolor: '#1a1d27', plot_bgcolor: '#1a1d27',
      font: {{ color: '#e0e0e0', size: 11 }},
      margin: {{ t: 36, l: 40, r: 16, b: 60 }}
    }};

    function renderCharts(data) {{
      Plotly.newPlot('chartTopQ', [{{
        x: data.top_questions.labels, y: data.top_questions.values,
        type: 'bar', marker: {{ color: '#3b4fd8' }}
      }}], Object.assign({{ title: 'Top Asked Questions' }}, darkLayout), {{ responsive: true, displayModeBar: false }});

      Plotly.newPlot('chartFeedback', [{{
        labels: ['Positive', 'Negative'],
        values: [data.feedback.positive_pct, data.feedback.negative_pct],
        type: 'pie', marker: {{ colors: ['#4ade80', '#f87171'] }}
      }}], Object.assign({{ title: 'Feedback Split' }}, darkLayout), {{ responsive: true, displayModeBar: false }});

      Plotly.newPlot('chartTickets', [{{
        x: data.tickets_by_reason.labels, y: data.tickets_by_reason.values,
        type: 'bar', marker: {{ color: '#fbbf24' }}
      }}], Object.assign({{ title: 'Open Tickets by Reason' }}, darkLayout), {{ responsive: true, displayModeBar: false }});
    }}

    function refreshCharts() {{
      fetch('/analytics/data').then(function(r) {{ return r.json(); }}).then(renderCharts);
    }}
    refreshCharts();
    setInterval(refreshCharts, 10000);
  </script>
</body></html>"""

    return html


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    app.run(debug=False, port=5000, threaded=False, processes=1)