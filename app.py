"""
app.py — ACD Mentor  (v2 — Extended Multi-Agent)
══════════════════════════════════════════════════
Main Streamlit entry point for the multi-agent RAG-powered
Automata and Compiler Design student assistant.

New in v2:
  • Mem0 memory agent (per-session persistence)
  • Voiceflow voice agent (optional)
  • Multi-format file ingestion (.md, .docx, .xlsx, .csv, .pdf)
  • Book knowledge source (separate ChromaDB collection)
  • Deep space black theme
  • Memory context indicator + voice response block
  • Ingestion results display

Run with:
    streamlit run app.py

API key configuration:
    Local  → create a .env file (copy from .env.example)
    Cloud  → add secrets in Streamlit Cloud dashboard
"""

from __future__ import annotations
import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ACD Mentor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "About": "ACD Mentor v2 — Multi-Agent RAG · Memory · Voice · Automata & Compiler Design",
    },
)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from ui.components import (
    load_css, render_header, render_bot_intro,
    render_citations, render_quality_badge,
    render_visual, render_quiz, render_progress_panel,
    render_memory_indicator, render_voice_response,
    render_ingestion_results,
)
from utils.config import has_valid_api_key, LLM_PROVIDER, active_model, ENABLE_WEB_KNOWLEDGE
from utils.progress_tracker import mark_topic, reset_progress, UNIT_MAP
from utils.pdf_exporter import export_session_as_pdf

# ── Apply custom CSS ──────────────────────────────────────────────────────────
load_css()


# ══════════════════════════════════════════════════════════════════════════════
# Session State Initialisation
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "messages":            [],
        "kb_loaded":           False,
        "uploaded_files":      [],
        "eli15":               False,
        "show_mistakes":       False,
        "use_only_user_docs":  False,
        "selected_unit":       "Auto-detect",
        "last_result":         None,
        "fast_mode":           False,
        "pending_prompt":      None,
        "voice_mode":          False,
        # Stable user_id per browser session for Mem0
        "user_id":             str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════
def render_result(result: dict):
    """Render a full orchestrator result dict into tabs."""
    syllabus_info     = result.get("syllabus_info", {})
    citations         = result.get("citations", [])
    book_citations    = result.get("book_citations", [])
    explanation       = result.get("explanation", "")
    visual            = result.get("visual", {})
    solution          = result.get("solution", "")
    quiz              = result.get("quiz", {})
    review            = result.get("review", {})
    voice_response    = result.get("voice_response")
    memory_context    = result.get("memory_context", "")
    ingestion_results = result.get("ingestion_results", [])
    total_chunks      = result.get("total_chunks_indexed", 0)
    error             = result.get("error")

    if error and not explanation:
        st.error(f"Pipeline error: {error}")
        return

    # Metadata row
    render_bot_intro(syllabus_info)
    render_citations(citations, book_citations)
    render_memory_indicator(memory_context)
    render_quality_badge(review)

    # Ingestion results (if files were processed this turn)
    render_ingestion_results(ingestion_results, total_chunks)

    # Voice response (if Voiceflow active)
    render_voice_response(voice_response)

    # ── Response Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs(["📖 Explanation", "📊 Diagram", "🔢 Step-by-Step", "🧪 Quiz"])

    with tabs[0]:
        if explanation:
            st.markdown(explanation)
        else:
            st.info("No explanation generated.")

    with tabs[1]:
        render_visual(visual)

    with tabs[2]:
        if solution:
            st.markdown(solution)
        else:
            st.info("No step-by-step solution for this query type.")

    with tabs[3]:
        render_quiz(quiz)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ ACD Mentor")
    st.caption(f"Provider: **{LLM_PROVIDER.upper()}** · Model: `{active_model()}`")
    st.divider()

    # ── API Key warning ───────────────────────────────────────────────────────
    if not has_valid_api_key():
        st.error(
            "⚠️ **No API key detected!**\n\n"
            "1. Copy `.env.example` → `.env`\n"
            "2. Add your `GROQ_API_KEY` or `OPENAI_API_KEY`\n\n"
            "For Streamlit Cloud: add keys in **Settings → Secrets**."
        )
        st.stop()

    # ── File Upload (extended formats) ────────────────────────────────────────
    st.markdown("### 📁 Upload Study Materials")
    uploaded = st.file_uploader(
        "PDFs, DOCX, XLSX, CSV, MD or TXT files",
        type=["pdf", "pptx", "ppt", "txt", "md", "docx", "xlsx", "csv"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbook chapters, spreadsheets, or slides.",
        label_visibility="collapsed",
    )

    # Is-book toggle for textbook ingestion into separate collection
    is_book_upload = st.toggle(
        "📚 Treat as textbook (book collection)",
        value=False,
        help="ON = indexed into the books collection (cited with priority). OFF = regular notes.",
    )

    if uploaded:
        new_files_data = []
        for uf in uploaded:
            if uf.name not in st.session_state.uploaded_files:
                new_files_data.append({
                    "filename": uf.name,
                    "data":     uf.getvalue(),
                    "is_book":  is_book_upload,
                })
                st.session_state.uploaded_files.append(uf.name)

        if new_files_data:
            with st.spinner(f"Indexing {len(new_files_data)} file(s)…"):
                from agents.ingestion_agent import ingest_file
                total_chunks = 0
                results = []
                for item in new_files_data:
                    chunks, err = ingest_file(
                        item["filename"], item["data"], is_book=item["is_book"]
                    )
                    total_chunks += chunks
                    results.append({"filename": item["filename"], "chunks": chunks, "error": err})

                for r in results:
                    if r["error"]:
                        st.warning(f"⚠️ {r['filename']}: {r['error']}")
                    else:
                        kind = "📚 book" if is_book_upload else "📄 notes"
                        st.success(f"✅ {r['filename']} → {r['chunks']} chunks ({kind})")

    if st.session_state.uploaded_files:
        st.caption("📄 " + " · ".join(st.session_state.uploaded_files[-5:]))

    st.divider()

    # ── Knowledge Source ──────────────────────────────────────────────────────
    st.markdown("### 🌐 Knowledge Source")
    use_only_user = st.toggle(
        "Use only my uploaded notes",
        value=st.session_state.use_only_user_docs,
        help="OFF = also search GeeksforGeeks articles.",
    )
    st.session_state.use_only_user_docs = use_only_user

    if ENABLE_WEB_KNOWLEDGE and not st.session_state.kb_loaded and not use_only_user:
        if st.button("🔄 Load GFG Knowledge Base", use_container_width=True):
            with st.spinner("Fetching GeeksforGeeks ACD articles…"):
                try:
                    from knowledge.gfg_scraper import load_all_gfg_articles
                    from rag.document_loader import load_raw_text_as_docs
                    from rag.vector_store import add_documents, build_or_load
                    build_or_load()
                    prog = st.progress(0)
                    def _cb(idx, total, topic):
                        prog.progress((idx + 1) / total, text=f"⏳ {topic}")
                    articles = load_all_gfg_articles(progress_callback=_cb)
                    total_chunks = 0
                    for art in articles:
                        docs = load_raw_text_as_docs(art["text"], source=art["url"])
                        total_chunks += add_documents(docs, pre_chunked=True)
                    st.session_state.kb_loaded = True
                    prog.empty()
                    st.success(f"✅ {len(articles)} articles · {total_chunks} chunks")
                except Exception as e:
                    st.error(f"GFG load failed: {e}")

    if st.session_state.kb_loaded:
        st.success("✅ GFG Knowledge Base active")

    st.divider()

    # ── Syllabus Filter ───────────────────────────────────────────────────────
    st.markdown("### 🗂️ Syllabus Focus")
    unit_options = ["Auto-detect"] + list(UNIT_MAP.keys())
    st.session_state.selected_unit = st.selectbox(
        "Filter to Unit", unit_options, index=0, label_visibility="collapsed"
    )

    st.divider()

    # ── Learning Mode ─────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Learning Mode")
    st.session_state.eli15 = st.toggle(
        "🧒 Explain like I'm 15",
        value=st.session_state.eli15,
    )
    st.session_state.show_mistakes = st.toggle(
        "⚠️ Show common mistakes",
        value=st.session_state.show_mistakes,
    )

    st.divider()

    # ── Voice Mode ────────────────────────────────────────────────────────────
    st.markdown("### 🎙️ Voice (Voiceflow)")
    voice_configured = bool(
        os.getenv("VOICEFLOW_API_KEY", "").strip() and
        os.getenv("VOICEFLOW_PROJECT_ID", "").strip()
    )
    if voice_configured:
        st.session_state.voice_mode = st.toggle(
            "Enable Voiceflow agent",
            value=st.session_state.voice_mode,
            help="Routes query through your Voiceflow project.",
        )
        if st.session_state.voice_mode:
            st.caption('<span class="voice-chip">🎙️ Voice Active</span>', unsafe_allow_html=True)
    else:
        st.caption("🔇 Voiceflow not configured. Add `VOICEFLOW_API_KEY` + `VOICEFLOW_PROJECT_ID` to `.env`.")

    st.divider()

    # ── Speed / Pipeline ──────────────────────────────────────────────────────
    st.markdown("### ⚡ Pipeline Speed")
    fast_mode = st.toggle(
        "⚡ Fast Mode (explanation only)",
        value=st.session_state.fast_mode,
        help="Skips diagram, step-by-step, quiz and critic. Cuts response time by ~60%.",
    )
    st.session_state.fast_mode = fast_mode
    if fast_mode:
        st.caption(
            '<span class="mode-chip fast">⚡ Fast</span> Diagram / Quiz / Critic skipped',
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            '<span class="mode-chip full">✦ Full</span> All agents active',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Mem0 Memory ───────────────────────────────────────────────────────────
    st.markdown("### 🧠 Memory (Mem0)")
    mem0_configured = bool(os.getenv("MEM0_API_KEY", "").strip())
    if mem0_configured:
        st.caption(f'<span class="memory-chip">🧠 Mem0 Active</span> user: `{st.session_state.user_id[:8]}…`', unsafe_allow_html=True)
    else:
        st.caption("💾 Local session only. Add `MEM0_API_KEY` to `.env` for cross-session memory.")

    st.divider()

    # ── Utilities ─────────────────────────────────────────────────────────────
    st.markdown("### 🛠️ Tools")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()
    with col_b:
        if st.button("🔄 Reset KB", use_container_width=True):
            from rag.vector_store import reset_vectorstore
            reset_vectorstore()
            st.session_state.kb_loaded = False
            st.session_state.uploaded_files = []
            reset_progress()
            st.rerun()

    if st.session_state.messages:
        pdf_data = export_session_as_pdf(st.session_state.messages)
        st.download_button(
            "📥 Download Notes (PDF)",
            data=pdf_data,
            file_name="acd_mentor_notes.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.divider()
    render_progress_panel()


# ══════════════════════════════════════════════════════════════════════════════
# Main Chat Area
# ══════════════════════════════════════════════════════════════════════════════
render_header()

# ── Welcome screen ────────────────────────────────────────────────────────────
EXAMPLE_PROMPTS = [
    "Convert (a|b)*abb from NFA to DFA",
    "Explain LL(1) parsing with a table",
    "What is peephole optimization?",
    "Derive FIRST and FOLLOW for a grammar",
    "Explain shift-reduce parsing conflicts",
]

if not st.session_state.messages:
    chips_html = "".join(
        f'<span class="prompt-chip" onclick="navigator.clipboard.writeText(\'{p.replace(chr(39), chr(92)+chr(39))}\').then(()=>{{}})">{p}</span>'
        for p in EXAMPLE_PROMPTS
    )
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem 2rem;color:#64748b;">
        <div style="font-size:3.5rem;margin-bottom:1rem;filter:drop-shadow(0 0 16px rgba(34,197,94,0.5));">⚙️</div>
        <h3 style="color:#94a3b8;font-weight:500;margin-bottom:0.5rem;">Welcome to ACD Mentor!</h3>
        <p style="font-size:0.9rem;max-width:560px;margin:0 auto;line-height:1.75;color:#64748b;">
            Your AI-powered tutor for <strong style="color:#4ade80;">Automata Theory</strong> and
            <strong style="color:#7dd3fc;">Compiler Design</strong>.<br>
            Now with <strong style="color:#c4b5fd;">Mem0 memory</strong>,
            <strong style="color:#7dd3fc;">Voiceflow voice</strong>, and
            <strong style="color:#fcd34d;">book knowledge sources</strong>.<br>
            Click a prompt below to copy it, then paste it in the chat.
        </p>
        <div style="margin-top:1.75rem;display:flex;justify-content:center;gap:0.75rem;flex-wrap:wrap;">
            {chips_html}
        </div>
    </div>
    <div class="shortcut-bar">
        <span class="shortcut-item">
            <span class="kbd">↑</span> / <span class="kbd">↓</span> previous messages
        </span>
        <span class="shortcut-item">
            <span class="kbd">Ctrl</span>+<span class="kbd">Enter</span> submit
        </span>
        <span class="shortcut-item">
            <span class="kbd">Esc</span> clear input
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Keyboard shortcut JS ──────────────────────────────────────────────────────
st.components.v1.html("""
<script>
(function(){
  function getInput(){
    return document.querySelector('[data-testid="stChatInput"] textarea');
  }
  var histIdx = -1;
  document.addEventListener('keydown', function(e){
    var inp = getInput();
    if(!inp) return;
    if(e.ctrlKey && e.key === 'Enter'){
      e.preventDefault();
      var btn = document.querySelector('[data-testid="stChatInput"] button');
      if(btn) btn.click();
      return;
    }
    if(e.key === 'Escape' && document.activeElement === inp){
      e.preventDefault();
      var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
      nativeSetter.call(inp,'');
      inp.dispatchEvent(new Event('input', {bubbles:true}));
      histIdx = -1;
      return;
    }
    if(document.activeElement === inp){
      var userMsgs = Array.from(document.querySelectorAll('[data-testid="stChatMessage"]'))
        .filter(m => m.querySelector('[aria-label="user avatar"]'))
        .map(m => m.querySelector('p') ? m.querySelector('p').innerText : '');
      if(e.key === 'ArrowUp'){
        if(userMsgs.length && histIdx < userMsgs.length - 1){
          histIdx++;
          var val = userMsgs[userMsgs.length - 1 - histIdx];
          var ns = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
          ns.call(inp, val);
          inp.dispatchEvent(new Event('input', {bubbles:true}));
          setTimeout(()=>{ inp.setSelectionRange(val.length,val.length); },0);
          e.preventDefault();
        }
      } else if(e.key === 'ArrowDown' && histIdx > 0){
        histIdx--;
        var val2 = userMsgs[userMsgs.length - 1 - histIdx] || '';
        var ns2 = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
        ns2.call(inp, val2);
        inp.dispatchEvent(new Event('input', {bubbles:true}));
        e.preventDefault();
      }
    }
  });
})();
</script>
""", height=0)

# ── Render persisted chat history ─────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        stored_result = msg.get("result")
        if msg["role"] == "assistant" and stored_result:
            render_result(stored_result)
        else:
            st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about Automata & Compiler Design…"):

    st.session_state.messages.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user", avatar="🎓"):
        st.markdown(prompt)

    fast       = st.session_state.fast_mode
    run_viz    = not fast
    run_solver = not fast
    run_quiz   = not fast
    run_critic = not fast

    if fast:
        steps = [
            ("🧠", "Recalling memory (Mem0)"),
            ("🗺️", "Routing + mapping to ACD syllabus"),
            ("🔍", "Hybrid RAG retrieval"),
            ("🤖", "Explainer agent (fast mode)"),
        ]
    else:
        steps = [
            ("🧠", "Recalling memory (Mem0)"),
            ("🗺️", "Routing + mapping to ACD syllabus"),
            ("🔍", "Hybrid RAG retrieval (notes + books)"),
            ("🤖", "Running agents — Explainer, Visualizer, Solver, Quiz"),
            ("✅", "Quality review (Critic)"),
            ("🎙️", "Voice agent (optional)"),
        ]

    with st.chat_message("assistant", avatar="🤖"):
        mode_label = "⚡ Fast Mode" if fast else "✦ Full Pipeline"
        status = st.status(f"🤖 {mode_label} — processing…", expanded=True)

        for emoji, label in steps:
            status.write(f"{emoji} {label}…")

        try:
            from rag.vector_store import build_or_load
            build_or_load()

            from agents.orchestrator import run_pipeline

            t0 = time.time()

            result = run_pipeline(
                query              = prompt,
                user_id            = st.session_state.user_id,
                use_only_user_docs = st.session_state.use_only_user_docs,
                eli15              = st.session_state.eli15,
                show_mistakes      = st.session_state.show_mistakes,
                run_visualizer     = run_viz,
                run_solver         = run_solver,
                run_quiz           = run_quiz,
                run_critic         = run_critic,
                voice_mode         = st.session_state.voice_mode,
            )

            elapsed = time.time() - t0
            status.update(
                label    = f"✅ Done in {elapsed:.1f}s!",
                state    = "complete",
                expanded = False,
            )

            si = result.get("syllabus_info", {})
            if si.get("unit") and si.get("topic"):
                mark_topic(si["unit"], si["topic"])

            mem_chip = ""
            if result.get("memory_context"):
                mem_chip = ' <span class="memory-chip">🧠 Memory</span>'
            voice_chip = ""
            if result.get("voice_response"):
                voice_chip = ' <span class="voice-chip">🎙️ Voice</span>'
            book_chip = ""
            if result.get("book_citations"):
                book_chip = f' <span class="book-citation">📚 {len(result["book_citations"])} book src</span>'

            st.markdown(
                f'<span class="elapsed-chip">⏱ {elapsed:.1f}s</span>'
                + (f' <span class="mode-chip fast">⚡ Fast</span>' if fast else
                   f' <span class="mode-chip full">✦ Full</span>')
                + mem_chip + voice_chip + book_chip,
                unsafe_allow_html=True,
            )

            render_result(result)

            explanation_preview = result.get("explanation", "")
            st.session_state.messages.append({
                "role":    "assistant",
                "content": explanation_preview[:400] + "…" if len(explanation_preview) > 400 else explanation_preview,
                "result":  result,
            })
            st.session_state.last_result = result

        except Exception as exc:
            status.update(label="❌ Pipeline error", state="error")
            err_msg = f"❌ Error: {exc}"
            st.error(err_msg)
            st.exception(exc)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": err_msg,
                "result":  None,
            })
