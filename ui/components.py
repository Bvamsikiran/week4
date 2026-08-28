"""
ui/components.py
─────────────────
Reusable Streamlit UI helper functions.
Renders:
  • App header (space-black theme)
  • Chat message bubbles
  • Response tabs (explanation, visual, solution, quiz)
  • Quality gate badge
  • Progress tracker panel
  • Citation tags (notes + books)
  • Memory context indicator
  • Voice response block
  • Ingestion results
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
            <p>Multi-Agent RAG · Memory · Voice · Automata &amp; Compiler Design · All 5 Units</p>
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
    unit       = syllabus_info.get("unit", "")
    topic      = syllabus_info.get("topic", "")
    difficulty = syllabus_info.get("difficulty", "intermediate")
    unit_n     = syllabus_info.get("unit_number", 0)
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
def render_citations(citations: List[str], book_citations: List[str] = None):
    if not citations and not (book_citations or []):
        return

    book_set = set(book_citations or [])
    all_citations = list(book_citations or []) + [c for c in citations if c not in book_set]

    if not all_citations:
        return

    tags = ""
    for c in all_citations:
        if c in book_set or c.startswith("📚"):
            tags += f'<span class="book-citation">📚 {c.replace("📚 ", "")}</span>'
        else:
            tags += f'<span class="citation-tag">📄 {c}</span>'

    st.markdown(
        f'<div style="margin-bottom:1rem;">📚 <strong style="color:#94a3b8;font-size:0.78rem;">Sources:</strong>&nbsp;{tags}</div>',
        unsafe_allow_html=True
    )


# ── Memory Context Indicator ──────────────────────────────────────────────────
def render_memory_indicator(memory_context: str):
    """Show a small chip if Mem0 memory was injected into this response."""
    if not memory_context:
        return
    lines = [l for l in memory_context.splitlines() if l.startswith("•")]
    count = len(lines)
    if count > 0:
        st.markdown(
            f'<span class="memory-chip">🧠 {count} memor{"y" if count == 1 else "ies"} recalled</span>',
            unsafe_allow_html=True
        )


# ── Quality Gate Badge ────────────────────────────────────────────────────────
def render_quality_badge(review: dict):
    if not review:
        return
    verdict    = review.get("verdict", "acceptable")
    score      = review.get("score", 7)
    confidence = review.get("confidence", "medium")
    icons = {"good": "✅", "acceptable": "⚠️", "needs_correction": "❌"}
    icon  = icons.get(verdict, "⚠️")

    st.markdown(
        f'<span class="quality-badge {verdict}">{icon} Quality: {score}/10 · '
        f'{verdict.replace("_", " ").title()} ({confidence} confidence)</span>',
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


# ── Voice Response Block ───────────────────────────────────────────────────────
def render_voice_response(voice_response: str):
    """Render Voiceflow response if available."""
    if not voice_response:
        return
    st.markdown(
        f'<div class="voice-box">🎙️ <strong>Voiceflow:</strong><br>{voice_response}</div>',
        unsafe_allow_html=True
    )


# ── Ingestion Results ─────────────────────────────────────────────────────────
def render_ingestion_results(ingestion_results: List[dict], total_chunks: int):
    if not ingestion_results:
        return
    with st.expander(f"📥 Indexed {total_chunks} chunks from {len(ingestion_results)} file(s)", expanded=False):
        for r in ingestion_results:
            fname  = r.get("filename", "?")
            chunks = r.get("chunks", 0)
            error  = r.get("error", "")
            if error:
                st.markdown(
                    f'<div class="ingest-result">❌ {fname}: {error}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ingest-result">✅ {fname} → {chunks} chunks</div>',
                    unsafe_allow_html=True
                )


# ── Mermaid Diagram ───────────────────────────────────────────────────────────
def render_visual(visual: dict):
    if not visual or visual.get("type") == "none":
        st.info("ℹ️ No diagram generated for this topic.")
        return

    vtype   = visual.get("type", "ascii")
    content = visual.get("content", "")
    caption = visual.get("caption", "")

    if caption:
        st.caption(f"📊 {caption}")

    if vtype == "mermaid":
        try:
            from agents.visualizer import clean_mermaid_code
            content = clean_mermaid_code(content)
        except (ImportError, ValueError):
            vtype = "ascii"

    if vtype == "mermaid":
        try:
            from streamlit_mermaid import st_mermaid  # type: ignore
            st_mermaid(content, height=400)
        except ImportError:
            mermaid_html = f"""
            <div class="mermaid" style="background:#020504;padding:1rem;border-radius:10px;border:1px solid rgba(22,163,74,0.2);">
            {content}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'dark', themeVariables:{{primaryColor:'#16a34a',background:'#000000',mainBkg:'#020504',nodeBorder:'#22c55e',lineColor:'#4ade80',textColor:'#e2e8f0'}}}});</script>
            """
            st.components.v1.html(mermaid_html, height=420, scrolling=True)
        except Exception:
            st.code(content, language="")

    elif vtype == "table":
        st.markdown(content)
    else:
        st.code(content, language="")


# ── Interactive MCQ Renderer ──────────────────────────────────────────────────
def render_quiz(quiz: dict):
    if not quiz:
        st.info("No quiz generated.")
        return

    mcqs          = quiz.get("mcqs", [])
    short_answers = quiz.get("short_answers", [])

    if mcqs:
        st.markdown("#### 📝 Multiple Choice Questions")
        for i, mcq in enumerate(mcqs, 1):
            question    = mcq.get("q", f"Question {i}")
            options     = mcq.get("options", {})
            correct_key = mcq.get("answer", "")
            explanation = mcq.get("explanation", "")

            with st.expander(f"Q{i}. {question}", expanded=(i == 1)):
                if not options:
                    st.warning("No options available.")
                    continue

                option_labels = [f"{k}: {v}" for k, v in options.items()]

                selected = st.radio(
                    "Select your answer:",
                    option_labels,
                    index=None,
                    key=f"quiz_mcq_{id(quiz)}_{i}",
                    label_visibility="collapsed",
                )

                if selected is not None:
                    selected_key = selected.split(":")[0].strip()
                    if selected_key == correct_key:
                        st.success(f"✅ Correct! **{correct_key}** is right.")
                    else:
                        st.error(
                            f"❌ Incorrect. You chose **{selected_key}**, "
                            f"but the correct answer is **{correct_key}: {options.get(correct_key, '')}**"
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
                st.text_area(
                    "Your answer:",
                    key=f"quiz_sa_{id(quiz)}_{i}",
                    height=80,
                    label_visibility="collapsed",
                    placeholder="Type your answer here (optional)...",
                )
                if st.button(f"Show Answer", key=f"quiz_sa_btn_{id(quiz)}_{i}"):
                    st.markdown(f"✅ **Model Answer:** {sa.get('answer', '')}")


# ── Progress Tracker Panel ────────────────────────────────────────────────────
def render_progress_panel():
    from utils.progress_tracker import get_progress

    progress = get_progress()
    st.markdown("### 📈 Your Progress")

    for unit, info in progress.items():
        pct   = info["pct"]
        done  = info["done"]
        total = info["total"]
        label = unit.split("–")[0].strip()

        st.markdown(
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.6rem;">'
            f'{label} &nbsp;<span style="color:#e2e8f0;font-weight:600;">{done}/{total}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True
        )
