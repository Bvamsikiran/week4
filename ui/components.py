"""
ui/components.py
─────────────────
Reusable Streamlit UI helper functions.
Renders:
  • App header
  • Chat message bubbles
  • Response tabs (explanation, visual, solution, quiz)
  • Quality gate badge
  • Progress tracker panel
  • Citation tags
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List


# ── CSS Loader ────────────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── App Header ────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="acd-header">
        <div class="logo">⚙️</div>
        <div class="title-block">
            <h1>ACD Mentor</h1>
            <p>Multi-Agent RAG · Automata & Compiler Design · All 5 Units</p>
        </div>
        <span class="acd-badge">AI Powered</span>
    </div>
    """, unsafe_allow_html=True)


# ── Chat Bubbles ──────────────────────────────────────────────────────────────
def render_user_message(text: str):
    st.markdown(f"""
    <div class="chat-bubble user">
        <div class="chat-avatar user-avatar">🎓</div>
        <div class="chat-content user">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_bot_intro(syllabus_info: dict):
    unit = syllabus_info.get("unit", "")
    topic = syllabus_info.get("topic", "")
    difficulty = syllabus_info.get("difficulty", "intermediate")
    unit_n = syllabus_info.get("unit_number", 0)
    pill_class = f"unit-{unit_n}" if 1 <= unit_n <= 5 else "unit-1"

    st.markdown(f"""
    <div class="chat-bubble">
        <div class="chat-avatar bot-avatar">🤖</div>
        <div class="chat-content bot">
            <span class="pill {pill_class}">{unit}</span>&nbsp;
            <strong>{topic}</strong> &nbsp;·&nbsp;
            <em style="color:#94a3b8;font-size:0.82rem">{difficulty}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Citations ─────────────────────────────────────────────────────────────────
def render_citations(citations: List[str]):
    if not citations:
        return
    tags = "".join(
        f'<span class="citation-tag">📄 {c}</span>' for c in citations
    )
    st.markdown(
        f'<div style="margin-bottom:1rem;">📚 <strong style="color:#94a3b8;font-size:0.78rem;">Sources:</strong>&nbsp;{tags}</div>',
        unsafe_allow_html=True
    )


# ── Quality Gate Badge ────────────────────────────────────────────────────────
def render_quality_badge(review: dict):
    if not review:
        return
    verdict = review.get("verdict", "acceptable")
    score = review.get("score", 7)
    confidence = review.get("confidence", "medium")
    icons = {"good": "✅", "acceptable": "⚠️", "needs_correction": "❌"}
    icon = icons.get(verdict, "⚠️")

    st.markdown(
        f'<span class="quality-badge {verdict}">{icon} Quality: {score}/10 · {verdict.replace("_"," ").title()} ({confidence} confidence)</span>',
        unsafe_allow_html=True
    )
    issues = review.get("issues", [])
    if issues:
        with st.expander("🔍 Critic's notes", expanded=False):
            for issue in issues:
                st.markdown(f"- {issue}")
            corrections = review.get("corrections", "")
            if corrections:
                st.info(f"💡 Suggested correction: {corrections}")


# ── Mermaid Diagram ───────────────────────────────────────────────────────────
def render_visual(visual: dict):
    if not visual or visual.get("type") == "none":
        st.info("ℹ️ No diagram generated for this topic.")
        return

    vtype = visual.get("type", "ascii")
    content = visual.get("content", "")
    caption = visual.get("caption", "")

    if caption:
        st.caption(f"📊 {caption}")

    if vtype == "mermaid":
        # Try streamlit-mermaid, fallback to HTML
        try:
            from streamlit_mermaid import st_mermaid  # type: ignore
            st_mermaid(content, height=400)
        except ImportError:
            # Inline HTML fallback using mermaid.js CDN
            mermaid_html = f"""
            <div class="mermaid" style="background:#1e293b;padding:1rem;border-radius:10px;">
            {content}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
            """
            st.components.v1.html(mermaid_html, height=420, scrolling=True)

    elif vtype == "table":
        st.markdown(content)

    else:  # ascii / steps
        st.code(content, language="")


# ── MCQ Renderer ──────────────────────────────────────────────────────────────
def render_quiz(quiz: dict):
    if not quiz:
        st.info("No quiz generated.")
        return

    mcqs = quiz.get("mcqs", [])
    short_answers = quiz.get("short_answers", [])

    if mcqs:
        st.markdown("#### 📝 Multiple Choice Questions")
        for i, mcq in enumerate(mcqs, 1):
            with st.expander(f"Q{i}. {mcq.get('q', '')}", expanded=(i == 1)):
                options = mcq.get("options", {})
                answer = mcq.get("answer", "")
                explanation = mcq.get("explanation", "")

                for opt_key, opt_text in options.items():
                    is_correct = opt_key == answer
                    badge_class = "correct" if is_correct else ""
                    row_class = "correct" if is_correct else ""
                    st.markdown(
                        f'<div class="option-row {row_class}">'
                        f'<span class="option-badge {badge_class}">{opt_key}</span>'
                        f'{opt_text}'
                        f'{"  ✓" if is_correct else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                if explanation:
                    st.markdown(
                        f'<div class="explanation-box">💡 {explanation}</div>',
                        unsafe_allow_html=True
                    )

    if short_answers:
        st.markdown("#### ✍️ Short Answer Questions")
        for i, sa in enumerate(short_answers, 1):
            with st.expander(f"SA{i}. {sa.get('q', '')}", expanded=False):
                hint = sa.get("hint", "")
                if hint:
                    st.markdown(f"💭 **Hint:** {hint}")
                st.markdown(f"✅ **Model Answer:** {sa.get('answer', '')}")


# ── Progress Tracker Panel ────────────────────────────────────────────────────
def render_progress_panel():
    from utils.progress_tracker import get_progress

    progress = get_progress()
    st.markdown("### 📈 Your Progress")

    for unit, info in progress.items():
        pct = info["pct"]
        done = info["done"]
        total = info["total"]
        label = unit.split("–")[0].strip()  # "Unit I"

        st.markdown(
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.6rem;">'
            f'{label} &nbsp;<span style="color:#e2e8f0;font-weight:600;">{done}/{total}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True
        )
