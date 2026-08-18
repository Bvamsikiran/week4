# ACD Mentor 🎓

A **multi-agent RAG-powered** Streamlit web application for mastering **Automata and Compiler Design (ACD)**. Built with LangChain, LangGraph, ChromaDB, Groq, and custom CSS.

---

## 📁 Project Structure

```
acd_mentor/
├── app.py                      ← Main Streamlit entry point
├── requirements.txt
├── .env.example                ← Copy to .env and add your keys
├── .gitignore
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py         ← LangGraph orchestration pipeline
│   ├── syllabus_mapper.py      ← Maps queries to unit + topic
│   ├── explainer.py            ← Simple, analogy-rich explanations
│   ├── visualizer.py           ← Mermaid / ASCII diagrams
│   ├── problem_solver.py       ← Step-by-step problem solutions
│   ├── quiz_master.py          ← MCQ + short-answer generator
│   └── critic.py               ← Quality gate / accuracy checker
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py      ← PDF + PPT ingestion
│   ├── embedder.py             ← Chunking + embedding pipeline
│   ├── vector_store.py         ← Chroma / FAISS wrapper
│   └── retriever.py            ← Hybrid retrieval (docs + web)
│
├── knowledge/
│   ├── __init__.py
│   ├── gfg_scraper.py          ← GeeksforGeeks ACD topic fetcher
│   └── curated_topics.py       ← Hardcoded GFG URLs + fallback blurbs
│
├── utils/
│   ├── __init__.py
│   ├── config.py               ← Env loading + settings
│   ├── llm_factory.py          ← LLM provider switcher
│   ├── pdf_exporter.py         ← Download notes as PDF
│   └── progress_tracker.py     ← Student progress state
│
├── ui/
│   ├── styles.css              ← Custom CSS (dark + light)
│   └── components.py           ← Reusable Streamlit UI helpers
│
└── data/                       ← Auto-created at runtime
    ├── chroma_db/
    ├── faiss_index/
    └── uploaded_docs/
```

---

## 🚀 Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/<you>/acd-mentor.git
cd acd-mentor
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (or OPENAI_API_KEY)
```

Or, for Streamlit Cloud, add keys in **Settings → Secrets** as TOML:

```toml
GROQ_API_KEY = "gsk_..."
LLM_PROVIDER = "groq"
```

### 3. Run

```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `.env` is in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo + `app.py` as entry point.
4. Under **Advanced settings → Secrets**, paste your `.env` keys in TOML format.
5. Click **Deploy** — done!

---

## 🔑 Supported LLM Providers

| Provider | Model | How to get key |
|----------|-------|----------------|
| **Groq** (recommended — free & fast) | `llama-3.3-70b-versatile` | [console.groq.com](https://console.groq.com) |
| OpenAI | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) |

Set `LLM_PROVIDER=groq` or `LLM_PROVIDER=openai` in `.env`.

---

## 🤖 Agents

| Agent | Role |
|-------|------|
| **Syllabus Mapper** | Routes query to the correct ACD unit & topic |
| **Explainer** | Analogy-rich, beginner-friendly explanations |
| **Visualizer** | Mermaid diagrams, conversion tables, parse trees |
| **Problem Solver** | Step-by-step worked solutions |
| **Quiz Master** | MCQs + short-answer questions with explanations |
| **Critic** | Quality gate — checks accuracy & completeness |

---

## 📚 Syllabus Coverage

- **Unit I**: Formal Languages, RE, DFA, NFA, conversions, Pumping Lemma, Lex
- **Unit II**: Compiler phases, Lexical Analysis, CFG, Parse Trees, LL(1), LR, LALR, YACC
- **Unit III**: SDT, S/L-attributed grammars, Intermediate Code, AST
- **Unit IV**: Runtime Environments, Storage, Code Optimization, Peephole, Flow Graphs
- **Unit V**: Code Generation, Register Allocation, DAG
