"""
utils/pdf_exporter.py
──────────────────────
Converts a chat-session summary into a downloadable PDF using fpdf2.
"""

from __future__ import annotations
from io import BytesIO
from typing import List, Dict


def export_session_as_pdf(messages: List[Dict[str, str]], title: str = "ACD Mentor – Study Notes") -> bytes:
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

    # ── Title ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 14, title, ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(100, 100, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Messages ───────────────────────────────────────────────────────────
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            pdf.set_fill_color(240, 244, 255)
            pdf.set_text_color(30, 30, 100)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "🎓 You asked:", ln=True, fill=True)
        else:
            pdf.set_fill_color(245, 255, 245)
            pdf.set_text_color(20, 80, 20)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "🤖 ACD Mentor:", ln=True, fill=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        # multi_cell handles line wrapping
        safe_content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(4)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
