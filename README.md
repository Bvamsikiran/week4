# ACD Mentor v2 — Multi-Agent RAG Tutor

> AI-powered tutor for **Automata Theory & Compiler Design** (5 Units).  
> Built on **LangGraph** · **ChromaDB** · **Mem0** · **Voiceflow** · **Streamlit**

---

## What's New in v2

| Feature | Details |
|---|---|
| **MemoryAgent** | Mem0 SDK — persists facts/topics across browser sessions per user |
| **IngestionAgent** | Multi-format: `.pdf .docx .xlsx .csv .md .txt` |
| **Book Collection** | Textbooks indexed into separate `acd_books` ChromaDB collection, cited with priority |
| **VoiceAgent** | Optional Voiceflow hand-off for voice I/O |
| **RouterAgent** | Intent classification (explain / solve / quiz / visualise / ingest) + syllabus mapping |
| **KnowledgeAgent** | Merged RAG from notes + books with RRF fusion |
| **Space Black Theme** | `~90% #000000` background, deep-green accents, white star twinkle |

---

## Multi-Agent Graph

```
MemoryAgent  ──→  RouterAgent  ──→  IngestionAgent (if files)
                      │
                  KnowledgeAgent  (RAG notes + books)
                      │
              ┌───────┴───────────────────┐
          Explainer  Visualizer  Solver  QuizMaster   (parallel)
              └───────┬───────────────────┘
                  CriticAgent  (quality gate)
                      │
                  VoiceAgent   (optional Voiceflow)
                      │
                     END
```

### Node Descriptions

| Node | File | Role |
|---|---|---|
| `MemoryAgent` | `agents/memory_agent.py` | Search + store user memories via Mem0 |
| `RouterAgent` | `agents/orchestrator.py` | Classify intent; delegate to SyllabusMapper |
| `IngestionAgent` | `agents/ingestion_agent.py` | Parse + chunk + embed multi-format files |
| `KnowledgeAgent` | `agents/orchestrator.py` | Hybrid RAG (notes + books, BM25 + dense + RRF) |
| `TutorAgent` (parallel) | `agents/explainer.py` + others | Explanation, diagram, step-by-step, quiz |
| `CriticAgent` | `agents/critic.py` | Verify symbols, equations, correctness |
| `VoiceAgent` | `agents/voice_agent.py` | Voiceflow REST Interact API hand-off |

---

## Setup

### 1. Clone & install

```bash
git clone <repo>
cd ACD-subject-agent
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required keys:
- `GROQ_API_KEY` **or** `OPENAI_API_KEY` (at least one)

Optional keys (features degrade gracefully without them):
- `MEM0_API_KEY` — cross-session memory ([get free key](https://app.mem0.ai/))
- `VOICEFLOW_API_KEY` + `VOICEFLOW_PROJECT_ID` — voice mode

### 3. Run

```bash
# Streamlit frontend
streamlit run app.py

# FastAPI backend (separate terminal)
uvicorn api:app --port 8000 --reload
```

---

## Free-Tier LLM Configuration

To minimise token costs, set in `.env`:

```env
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant   # free tier
# Fallback (auto-used on rate-limit):
OPENAI_MODEL=gpt-4o-mini
```

---

## File Upload & Book Ingestion

### Supported formats
`.pdf` · `.docx` · `.xlsx` · `.csv` · `.md` · `.txt` · `.pptx`

### Using the sidebar uploader
1. Click **Upload Study Materials** in the sidebar
2. Toggle **"Treat as textbook"** ON for textbook PDFs
3. Files are parsed, chunked, and embedded automatically
4. Textbooks go into `acd_books` collection — answers cite them preferentially

### Indexing books via CLI (optional)

```python
from agents.ingestion_agent import ingest_file

with open("automata_sipser.pdf", "rb") as f:
    chunks, err = ingest_file("automata_sipser.pdf", f.read(), is_book=True)
print(f"Indexed {chunks} chunks")
```

---

## API (backward-compatible)

`POST http://localhost:8000/ask`

```json
{
  "query": "Convert (a|b)*abb NFA to DFA",
  "user_id": "student_42",
  "eli15": false,
  "fast_mode": false
}
```

Response includes: `explanation`, `visual`, `solution`, `quiz`, `review`, `citations`, `book_citations`, `memory_context`, `voice_response`.

---

## Project Structure

```
ACD-subject-agent/
├── app.py                     # Streamlit UI (updated)
├── api.py                     # FastAPI backend
├── agents/
│   ├── orchestrator.py        # LangGraph StateGraph (7 nodes) ← updated
│   ├── memory_agent.py        # Mem0 ← NEW
│   ├── ingestion_agent.py     # Multi-format ingest ← NEW
│   ├── voice_agent.py         # Voiceflow ← NEW
│   ├── critic.py              # Quality gate
│   ├── explainer.py           # Core ACD explanation
│   ├── visualizer.py          # Diagrams / tables
│   ├── problem_solver.py      # Step-by-step solutions
│   ├── quiz_master.py         # MCQ + short answers
│   └── syllabus_mapper.py     # Unit / topic classification
├── rag/
│   ├── retriever.py           # Hybrid RAG (dense + BM25 + RRF)
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── embedder.py            # Sentence-transformers
│   └── document_loader.py     # PDF / PPTX / text loaders
├── ui/
│   ├── components.py          # Streamlit helpers ← updated
│   └── styles.css             # Space-black theme ← updated
├── utils/
│   ├── config.py
│   ├── progress_tracker.py
│   └── pdf_exporter.py
├── data/
│   ├── chroma_db/             # Main vector store
│   ├── books/                 # Saved textbook files
│   └── uploaded_docs/         # Saved note files
├── .env.example               # ← updated with Mem0 + Voiceflow keys
└── requirements.txt           # ← updated (mem0ai, python-docx, openpyxl)
```

---

## Content Quality Rules

All agents follow strict ACD content guidelines:
- ✅ Correct mathematical symbols and LaTeX/KaTeX equations
- ✅ Step-by-step indented explanations with clear hierarchy
- ✅ ASCII state-transition visualizations where applicable
- ✅ Transition tables with proper notation (δ, Σ, Q, q₀, F)
- ❌ Never invents wrong symbols or equations (enforced by CriticAgent)
