"""
api.py — ACD Mentor FastAPI Backend  (v2)
══════════════════════════════════════════
POST /ask  — main question endpoint (backward-compatible)

New optional fields in request body:
  user_id    : str  — Mem0 user key (defaults to "anonymous")
  eli15      : bool — explain like I'm 15
  show_mistakes: bool
  voice_mode : bool — route through Voiceflow

Run:
    uvicorn api:app --port 8000 --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from agents.orchestrator import run_pipeline

app = FastAPI(title="ACD Mentor API", version="2.0")


class QueryRequest(BaseModel):
    query:         str
    user_id:       Optional[str]  = "anonymous"
    eli15:         Optional[bool] = False
    show_mistakes: Optional[bool] = False
    voice_mode:    Optional[bool] = False


@app.post("/ask")
def ask_mentor(request: QueryRequest):
    """
    Run the full multi-agent pipeline.
    Returns explanation text (and optionally voice_response).
    """
    try:
        result = run_pipeline(
            query         = request.query,
            user_id       = request.user_id or "anonymous",
            eli15         = request.eli15,
            show_mistakes = request.show_mistakes,
            voice_mode    = request.voice_mode,
            run_visualizer= False,
            run_quiz      = False,
            run_solver    = False,
        )
        reply = result.get("explanation", "I couldn't generate an explanation.")
        voice = result.get("voice_response")
        return {
            "reply":          reply,
            "voice_response": voice,
            "citations":      result.get("citations", []),
            "book_citations": result.get("book_citations", []),
            "memory_active":  bool(result.get("memory_context")),
        }
    except Exception as exc:
        return {"reply": f"An error occurred: {exc}"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}
