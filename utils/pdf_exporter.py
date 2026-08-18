"""
utils/pdf_exporter.py
──────────────────────
Converts a chat-session summary into a downloadable PDF using fpdf2.

Fix: All text is sanitized through _safe() before being written to the PDF.
     Built-in FPDF fonts (Helvetica) only support latin-1. Special Unicode
     characters like en-dash (–), em-dash (—), smart quotes, bullets, etc.
     are replaced with their ASCII equivalents to prevent UnicodeEncodeError.
"""

from __future__ import annotations
from io import BytesIO
from typing import List, Dict

# ── Unicode → ASCII replacement map ──────────────────────────────────────────
_UNICODE_MAP = str.maketrans({
    "\u2013": "-",   # en dash      –
    "\u2014": "--",  # em dash      —
    "\u2018": "'",   # left quote   '
    "\u2019": "'",   # right quote  '
    "\u201c": '"',   # left dquote  "
    "\u201d": '"',   # right dquote "
    "\u2022": "*",   # bullet       •
    "\u2026": "...", # ellipsis     …
    "\u00b7": "*",   # middle dot   ·
    "\u2039": "<",   # single angle ‹
    "\u203a": ">",   # single angle ›
    "\u00a0": " ",   # non-breaking space
    "\u2192": "->",  # right arrow  →
    "\u2190": "<-",  # left arrow   ←
    "\u2264": "<=",  # less equal   ≤
    "\u2265": ">=",  # greater eq   ≥
    "\u03b5": "epsilon",  # ε
    "\u03b4": "delta",    # δ
    "\u03a3": "Sigma",    # Σ
    "\u2208": "in",       # ∈
    "\u2205": "{}",       # ∅
    "\u03bb": "lambda",   # λ
})


def _safe(text: str) -> str:
    """
    Replace common Unicode characters with ASCII equivalents,
    then drop anything else that can't be encoded as latin-1.
    """
    text = text.translate(_UNICODE_MAP)
    # Strip any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_session_as_pdf(
    messages: List[Dict[str, str]],
    title: str = "ACD Mentor - Study Notes",
) -> bytes:
    """
    Render the list of chat messages as a PDF and return raw bytes.

    Parameters
    ----------
    messages : list of {"role": str, "content": str}
    title    : PDF document title

    Returns
    -------
    bytes : raw PDF data suitable for st.download_button
    """
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: return plain-text bytes
        text = f"{title}\n{'='*60}\n\n"
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            text += f"[{role}]\n{content}\n\n{'─'*40}\n\n"
        return text.encode("utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 14, _safe(title), ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(100, 100, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Messages ──────────────────────────────────────────────────────────────
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            pdf.set_fill_color(240, 244, 255)
            pdf.set_text_color(30, 30, 100)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "You asked:", ln=True, fill=True)
        else:
            pdf.set_fill_color(245, 255, 245)
            pdf.set_text_color(20, 80, 20)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "ACD Mentor:", ln=True, fill=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, _safe(content))
        pdf.ln(4)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
