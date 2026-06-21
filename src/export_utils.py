"""
src/export_utils.py — Conversation Export Module

Lets a user download their chat history as a simple, readable PDF —
useful as a reference or proof of what HR policy answer they were given.

Uses fpdf2 (pip install fpdf2) — lightweight, no external system
dependencies (unlike some PDF libraries that need wkhtmltopdf etc.)
"""

import os
from datetime import datetime
from typing import List, Dict

from fpdf import FPDF


def _sanitize_text(text: str) -> str:
    """
    fpdf2's default core fonts only support Latin-1 characters.
    Replace any character outside that range so export never crashes
    on emojis, smart quotes, etc. pasted into the chat.
    """
    replacements = {
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash  (இதுவே உங்கள் error!)
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote / apostrophe
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
        "\u2022": "--",    # bullet
        "\u2122": "(TM)",
        "\u00ae": "(R)",
        "\u00a9": "(C)",
        "\u20b9": "Rs.",  # Indian Rupee sign
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u00d7": "x",    # multiplication sign
        "\u00f7": "/",    # division sign
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # மீதமுள்ள எந்த Unicode-ஐயும் Latin-1-க்கு replace பண்ணு
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ConversationPDF(FPDF):
    """Small FPDF subclass so we can customize the header/footer once."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "HR Policy Chatbot -- Conversation Transcript", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _sanitize_text(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_conversation_to_pdf(messages: List[Dict], filename: str) -> str:
    """
    Generate a PDF transcript of a chat conversation.

    Args:
        messages: List of dicts like {"role": "user"|"bot", "content": str}
                  in the order they were sent.
        filename: Output path for the PDF (e.g. "exports/chat_alice_20250619.pdf").

    Returns:
        The filename that was written to (same as input `filename`),
        so the caller can send/serve that file.

    Raises:
        ValueError: If messages is empty.
    """
    if not messages:
        raise ValueError("No messages provided -- nothing to export.")

    pdf = _ConversationPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for msg in messages:
        role = msg.get("role", "bot")
        content = _sanitize_text(msg.get("content", ""))

        # Label and color-code by role
        if role == "user":
            label = "You"
            pdf.set_text_color(30, 60, 150)
        else:
            label = "HR Bot"
            pdf.set_text_color(20, 110, 70)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"{label}:", ln=True)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)

    # Make sure the output directory exists before writing
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pdf.output(filename)
    print(f"[export] Conversation exported to '{filename}'")

    return filename