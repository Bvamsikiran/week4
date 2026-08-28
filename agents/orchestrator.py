"""
agents/orchestrator.py
───────────────────────
Multi-agent orchestration pipeline using LangGraph StateGraph.

Extended Graph (v2):
┌──────────────┐
│  MemoryAgent │  (search + store per-user memory via Mem0)
└──────┬───────┘
       │
┌──────▼───────┐
│  RouterAgent │  (classify intent: explain / solve / quiz / visualise / ingest)
└──────┬───────┘
       │
   ┌───┴────────────────────────┐
   │                            │
┌──▼──────────┐       ┌─────────▼──────┐
│KnowledgeAgent│       │IngestionAgent  │
│(RAG + books) │       │(only if files) │
└──────┬───────┘       └────────────────┘
       │
┌──────▼───────────────────────────────┐
│       Parallel specialist agents      │
│  Explainer · Visualizer              │
│  ProblemSolver · QuizMaster          │
└──────┬───────────────────────────────┘
       │
┌──────▼──────┐
│ CriticAgent │  (quality gate: verify symbols / equations / correctness)
└──────┬──────┘
       │
┌──────▼──────┐
│  VoiceAgent │  (optional Voiceflow hand-off)
└─────────────┘
       │
     END

All new nodes are additive; the existing POST /ask API is 100% backward-compatible.
"""

from __future__ import annotations
from typing import TypedDict, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import agents.syllabus_mapper as syllabus_mapper
import agents.explainer as explainer
import agents.visualizer as visualizer
import agents.problem_solver as problem_solver
import agents.quiz_master as quiz_master
import agents.critic as critic
import agents.memory_agent as memory_agent
import agents.voice_agent as voice_agent
from rag.retriever import retrieve_context


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Unified state flowing through the extended LangGraph pipeline."""
    # ── inputs ────────────────────────────────────────────────────────────────
    query:             str
    user_id:           str           # session / user identifier for Mem0
    use_only_user_docs: bool
    eli15:             bool
    show_mistakes:     bool
    run_quiz:          bool
    run_solver:        bool
    run_visualizer:    bool
    run_critic:        bool
    voice_mode:        bool          # True → route through VoiceAgent
    pending_files:     List[dict]    # [{"filename", "data", "is_book"}]

    # ── intermediate ─────────────────────────────────────────────────────────
    intent:            str           # router classification
    memory_context:    str           # injected from Mem0
    syllabus_info:     dict
    context:           str           # RAG context string
    citations:         List[str]
    book_context:      str           # extra context from books collection
    book_citations:    List[str]

    # ── outputs ───────────────────────────────────────────────────────────────
    explanation:       str
    visual:            dict
    solution:          str
    quiz:              dict
    review:            dict
    voice_response:    Optional[str]
    ingestion_results: List[dict]
    total_chunks_indexed: int
    error:             Optional[str]


# ── Node: MemoryAgent ─────────────────────────────────────────────────────────

def node_memory_agent(state: AgentState) -> AgentState:
    """Fetch Mem0 memories + store user query. Called first on every turn."""
    try:
        state = memory_agent.run(state)
    except Exception as exc:
        state["memory_context"] = ""
        print(f"[MemoryAgent node] {exc}")
    return state


# ── Node: RouterAgent ─────────────────────────────────────────────────────────

def node_router(state: AgentState) -> AgentState:
    """
    Classify query intent:
      explain | solve | quiz | visualise | ingest | general

    Also delegates to SyllabusMapper for unit identification.
    """
    try:
        state["syllabus_info"] = syllabus_mapper.run(state["query"])
    except Exception as exc:
        state["syllabus_info"] = {
            "unit": "General", "unit_number": 0,
            "topic": "ACD General", "refined_query": state["query"],
            "difficulty": "intermediate", "keywords": [],
        }
        state["error"] = f"Router/Syllabus mapper: {exc}"

    # Simple rule-based intent classification (no extra LLM call → free)
    q_lower = state["query"].lower()
    if state.get("pending_files"):
        intent = "ingest"
    elif any(kw in q_lower for kw in ["quiz", "test me", "question", "mcq"]):
        intent = "quiz"
    elif any(kw in q_lower for kw in ["solve", "step", "derive", "calculate", "prove"]):
        intent = "solve"
    elif any(kw in q_lower for kw in ["draw", "diagram", "table", "automaton", "dfa", "nfa", "cfg", "pda"]):
        intent = "visualise"
    elif any(kw in q_lower for kw in ["explain", "what is", "define", "describe", "how"]):
        intent = "explain"
    else:
        intent = "general"

    state["intent"] = intent
    return state


# ── Node: IngestionAgent ──────────────────────────────────────────────────────

def node_ingestion(state: AgentState) -> AgentState:
    """Index pending file uploads (skipped when no files pending)."""
    if not state.get("pending_files"):
        return state
    try:
        from agents.ingestion_agent import run as ingest_run
        state = ingest_run(state)
    except Exception as exc:
        state["ingestion_results"] = []
        state["total_chunks_indexed"] = 0
        print(f"[IngestionAgent node] {exc}")
    return state


# ── Node: KnowledgeAgent (RAG + Books) ───────────────────────────────────────

def node_knowledge(state: AgentState) -> AgentState:
    """
    Retrieves context from:
      1. Main ChromaDB collection (notes + GFG)
      2. Books collection (acd_books) — cited preferentially
    Merges both into state["context"] with book excerpts first.
    """
    try:
        refined_q = state.get("syllabus_info", {}).get("refined_query", state["query"])

        # Primary RAG retrieval
        context, citations = retrieve_context(
            refined_q, k=5,
            use_only_user_docs=state.get("use_only_user_docs", False),
        )

        # Books collection retrieval (separate Chroma collection)
        book_context, book_citations = _retrieve_books(refined_q, k=3)

        # Books take priority — prepend them
        if book_context:
            merged_context = (
                "📚 [From textbooks]\n" + book_context +
                "\n\n---\n\n📄 [From notes / web]\n" + context
            )
            merged_citations = book_citations + [c for c in citations if c not in book_citations]
        else:
            merged_context = context
            merged_citations = citations

        state["context"]       = merged_context
        state["citations"]     = merged_citations
        state["book_context"]  = book_context
        state["book_citations"] = book_citations

    except Exception as exc:
        state["context"]       = ""
        state["citations"]     = []
        state["book_context"]  = ""
        state["book_citations"] = []
        state["error"] = f"KnowledgeAgent: {exc}"

    return state


def _retrieve_books(query: str, k: int = 3):
    """Retrieve from acd_books collection. Returns (context_str, citations)."""
    try:
        from rag.vector_store import build_or_load
        from langchain_community.vectorstores import Chroma
        from rag.embedder import get_embeddings
        from utils.config import CHROMA_PERSIST_DIR

        vs = Chroma(
            collection_name="acd_books",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return "", []

        parts, citations = [], []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "textbook")
            page = doc.metadata.get("page", "")
            page_str = f", p.{page}" if page else ""
            parts.append(f"[Book {i}] ({src}{page_str}):\n{doc.page_content}")
            label = f"📚 {src}{page_str}"
            if label not in citations:
                citations.append(label)

        return "\n\n---\n\n".join(parts), citations
    except Exception as exc:
        print(f"[KnowledgeAgent] Books retrieval failed: {exc}")
        return "", []


# ── Node: Parallel Specialists (TutorAgent) ───────────────────────────────────

def node_parallel_agents(state: AgentState) -> AgentState:
    """
    Run Explainer, Visualizer, ProblemSolver, QuizMaster in parallel.
    Memory context is prepended to RAG context for richer answers.
    """
    query        = state["query"]
    syllabus_info = state.get("syllabus_info", {})
    raw_context  = state.get("context", "")
    memory_ctx   = state.get("memory_context", "")
    eli15        = state.get("eli15", False)
    show_mistakes = state.get("show_mistakes", False)

    # Prepend memory context (keeps prompts lean)
    context = (memory_ctx + "\n\n" + raw_context).strip() if memory_ctx else raw_context

    tasks: Dict[str, Any] = {}
    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks["explanation"] = executor.submit(
            explainer.run, query, syllabus_info, context, eli15, show_mistakes
        )
        if state.get("run_visualizer", True):
            tasks["visual"] = executor.submit(
                visualizer.run, query, syllabus_info, context
            )
        if state.get("run_solver", True):
            tasks["solution"] = executor.submit(
                problem_solver.run, query, syllabus_info, context
            )
        if state.get("run_quiz", True):
            tasks["quiz"] = executor.submit(
                quiz_master.run, query, syllabus_info, context
            )

        for key, future in tasks.items():
            try:
                results[key] = future.result(timeout=90)
            except Exception as exc:
                results[key] = f"⚠️ Agent error: {exc}"

    state["explanation"] = results.get("explanation", "")
    state["visual"]      = results.get("visual", {"type": "none", "content": "", "caption": ""})
    state["solution"]    = results.get("solution", "")
    state["quiz"]        = results.get("quiz", {})
    return state


# ── Node: CriticAgent ────────────────────────────────────────────────────────

def node_critic(state: AgentState) -> AgentState:
    """Quality gate — verify symbols, equations and correctness."""
    if not state.get("run_critic", True):
        return state

    explanation_str = state.get("explanation", "")
    solution_str    = state.get("solution", "")

    if isinstance(explanation_str, str) and explanation_str.startswith("⚠️"):
        state["review"] = {
            "score": 0, "verdict": "needs_correction",
            "issues": ["Explanation agent failed"], "corrections": "",
            "confidence": "low",
        }
        return state

    try:
        if isinstance(explanation_str, str) and isinstance(solution_str, str):
            state["review"] = critic.run(
                state["query"], explanation_str, solution_str,
                state.get("syllabus_info", {}),
            )
        else:
            state["review"] = {}
    except Exception as exc:
        state["review"] = {
            "score": 7, "verdict": "acceptable",
            "issues": [f"Critic error: {exc}"], "corrections": "",
            "confidence": "low",
        }
    return state


# ── Node: VoiceAgent ─────────────────────────────────────────────────────────

def node_voice(state: AgentState) -> AgentState:
    """Optional Voiceflow hand-off — pass-through when voice_mode is False."""
    try:
        state = voice_agent.run(state)
    except Exception as exc:
        state["voice_response"] = None
        print(f"[VoiceAgent node] {exc}")
    return state


# ── Build Graph ───────────────────────────────────────────────────────────────

def _build_graph():
    """Construct the full extended LangGraph StateGraph pipeline."""
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        # Register all nodes
        graph.add_node("memory_agent",    node_memory_agent)
        graph.add_node("router",          node_router)
        graph.add_node("ingestion",       node_ingestion)
        graph.add_node("knowledge",       node_knowledge)
        graph.add_node("parallel_agents", node_parallel_agents)
        graph.add_node("critic",          node_critic)
        graph.add_node("voice",           node_voice)

        # Entry point
        graph.set_entry_point("memory_agent")

        # Edges: memory → router → (ingestion if files) → knowledge → parallel → critic → voice → END
        graph.add_edge("memory_agent", "router")

        def _route_after_router(state: AgentState) -> str:
            """If files pending, run ingestion first; always then proceed to knowledge."""
            if state.get("pending_files"):
                return "ingestion"
            return "knowledge"

        graph.add_conditional_edges(
            "router",
            _route_after_router,
            {"ingestion": "ingestion", "knowledge": "knowledge"},
        )
        graph.add_edge("ingestion",       "knowledge")
        graph.add_edge("knowledge",       "parallel_agents")
        graph.add_edge("parallel_agents", "critic")
        graph.add_edge("critic",          "voice")
        graph.add_edge("voice",           END)

        return graph.compile()

    except ImportError:
        return None  # LangGraph not installed → use sequential fallback


# Compile once at module load
_compiled_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    query:              str,
    user_id:            str  = "anonymous",
    use_only_user_docs: bool = False,
    eli15:              bool = False,
    show_mistakes:      bool = False,
    run_quiz:           bool = True,
    run_solver:         bool = True,
    run_visualizer:     bool = True,
    run_critic:         bool = True,
    voice_mode:         bool = False,
    pending_files:      List[dict] = None,
) -> Dict[str, Any]:
    """
    Full multi-agent pipeline (backward-compatible with v1 API).

    New optional parameters
    -----------------------
    user_id       : Mem0 user key (defaults to "anonymous")
    voice_mode    : route final answer through Voiceflow
    pending_files : list of {"filename": str, "data": bytes, "is_book": bool}
    """
    initial_state: AgentState = {
        "query":              query,
        "user_id":            user_id,
        "use_only_user_docs": use_only_user_docs,
        "eli15":              eli15,
        "show_mistakes":      show_mistakes,
        "run_quiz":           run_quiz,
        "run_solver":         run_solver,
        "run_visualizer":     run_visualizer,
        "run_critic":         run_critic,
        "voice_mode":         voice_mode,
        "pending_files":      pending_files or [],
        "intent":             "general",
        "memory_context":     "",
        "syllabus_info":      {},
        "context":            "",
        "citations":          [],
        "book_context":       "",
        "book_citations":     [],
        "explanation":        "",
        "visual":             {"type": "none", "content": "", "caption": ""},
        "solution":           "",
        "quiz":               {},
        "review":             {},
        "voice_response":     None,
        "ingestion_results":  [],
        "total_chunks_indexed": 0,
        "error":              None,
    }

    if _compiled_graph is not None:
        try:
            result = _compiled_graph.invoke(initial_state)
            return dict(result)
        except Exception as exc:
            initial_state["error"] = f"LangGraph pipeline error: {exc}"
            # Fall through to sequential fallback

    # ── Sequential fallback ───────────────────────────────────────────────────
    state = initial_state
    state = node_memory_agent(state)
    state = node_router(state)
    if state.get("pending_files"):
        state = node_ingestion(state)
    state = node_knowledge(state)
    state = node_parallel_agents(state)
    state = node_critic(state)
    state = node_voice(state)
    return dict(state)
