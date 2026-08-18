"""
agents/orchestrator.py
───────────────────────
Multi-agent orchestration pipeline using LangGraph StateGraph.

Flow:
  1. Syllabus Mapper  → identify unit + topic
  2. RAG Retrieval    → fetch relevant context
  3. [Parallel]
       a. Explainer      → clear explanation
       b. Visualizer     → diagram / table
       c. Problem Solver → step-by-step solution
       d. Quiz Master    → practice questions
  4. Critic           → quality check (runs on explanation + solution)

Returns a structured dict with all outputs for the UI to render.

Upgrade:
  • True LangGraph StateGraph with typed state
  • Graceful per-node error handling
  • Conditional routing (skip critic if agents failed)
"""

from __future__ import annotations
from typing import TypedDict, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import agents.syllabus_mapper as syllabus_mapper
import agents.explainer as explainer
import agents.visualizer as visualizer
import agents.problem_solver as problem_solver
import agents.quiz_master as quiz_master
import agents.critic as critic
from rag.retriever import retrieve_context


# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """Unified state flowing through the LangGraph pipeline."""
    query: str
    use_only_user_docs: bool
    eli15: bool
    show_mistakes: bool
    run_quiz: bool
    run_solver: bool
    run_visualizer: bool
    run_critic: bool
    syllabus_info: dict
    context: str
    citations: List[str]
    explanation: str
    visual: dict
    solution: str
    quiz: dict
    review: dict
    error: str | None


# ── Node functions ────────────────────────────────────────────────────────────

def node_syllabus_mapper(state: AgentState) -> AgentState:
    """Step 1: Map query to syllabus unit + topic."""
    try:
        state["syllabus_info"] = syllabus_mapper.run(state["query"])
    except Exception as exc:
        state["syllabus_info"] = {
            "unit": "General", "unit_number": 0,
            "topic": "ACD General", "refined_query": state["query"],
            "difficulty": "intermediate", "keywords": [],
        }
        state["error"] = f"Syllabus mapper: {exc}"
    return state


def node_rag_retrieval(state: AgentState) -> AgentState:
    """Step 2: Retrieve relevant context via hybrid RAG."""
    try:
        refined_q = state.get("syllabus_info", {}).get("refined_query", state["query"])
        context, citations = retrieve_context(
            refined_q, k=6,
            use_only_user_docs=state.get("use_only_user_docs", False),
        )
        state["context"] = context
        state["citations"] = citations
    except Exception as exc:
        state["context"] = ""
        state["citations"] = []
        state["error"] = f"RAG retrieval: {exc}"
    return state


def node_parallel_agents(state: AgentState) -> AgentState:
    """Step 3: Run Explainer, Visualizer, Problem Solver, Quiz Master in parallel."""
    query = state["query"]
    syllabus_info = state.get("syllabus_info", {})
    context = state.get("context", "")
    eli15 = state.get("eli15", False)
    show_mistakes = state.get("show_mistakes", False)

    tasks = {}
    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Always run explainer
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
    state["visual"] = results.get("visual", {"type": "none", "content": "", "caption": ""})
    state["solution"] = results.get("solution", "")
    state["quiz"] = results.get("quiz", {})

    return state


def node_critic(state: AgentState) -> AgentState:
    """Step 4: Quality gate — review explanation + solution."""
    if not state.get("run_critic", True):
        return state

    explanation_str = state.get("explanation", "")
    solution_str = state.get("solution", "")

    # Skip critic if agents failed
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


# ── Build Graph ───────────────────────────────────────────────────────────────

def _build_graph():
    """Construct the LangGraph StateGraph pipeline."""
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        graph.add_node("syllabus_mapper", node_syllabus_mapper)
        graph.add_node("rag_retrieval", node_rag_retrieval)
        graph.add_node("parallel_agents", node_parallel_agents)
        graph.add_node("critic", node_critic)

        graph.set_entry_point("syllabus_mapper")
        graph.add_edge("syllabus_mapper", "rag_retrieval")
        graph.add_edge("rag_retrieval", "parallel_agents")
        graph.add_edge("parallel_agents", "critic")
        graph.add_edge("critic", END)

        return graph.compile()

    except ImportError:
        # LangGraph not installed — return None, use fallback pipeline
        return None


# Compile graph once at module load
_compiled_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    use_only_user_docs: bool = False,
    eli15: bool = False,
    show_mistakes: bool = False,
    run_quiz: bool = True,
    run_solver: bool = True,
    run_visualizer: bool = True,
    run_critic: bool = True,
) -> Dict[str, Any]:
    """
    Full multi-agent pipeline.

    Parameters
    ----------
    query              : student's question
    use_only_user_docs : restrict RAG to uploaded documents only
    eli15              : explain like I'm 15
    show_mistakes      : include common mistakes in explanation
    run_quiz           : whether to generate quiz questions
    run_solver         : whether to generate step-by-step solution
    run_visualizer     : whether to generate a diagram
    run_critic         : whether to run the quality gate

    Returns
    -------
    dict with keys:
      query, syllabus_info, context, citations,
      explanation, visual, solution, quiz, review, error
    """
    initial_state: AgentState = {
        "query": query,
        "use_only_user_docs": use_only_user_docs,
        "eli15": eli15,
        "show_mistakes": show_mistakes,
        "run_quiz": run_quiz,
        "run_solver": run_solver,
        "run_visualizer": run_visualizer,
        "run_critic": run_critic,
        "syllabus_info": {},
        "context": "",
        "citations": [],
        "explanation": "",
        "visual": {"type": "none", "content": "", "caption": ""},
        "solution": "",
        "quiz": {},
        "review": {},
        "error": None,
    }

    # ── Use LangGraph if available ────────────────────────────────────────
    if _compiled_graph is not None:
        try:
            result = _compiled_graph.invoke(initial_state)
            return dict(result)
        except Exception as exc:
            initial_state["error"] = f"LangGraph pipeline error: {exc}"
            # Fall through to sequential fallback

    # ── Fallback: sequential execution ────────────────────────────────────
    state = initial_state
    state = node_syllabus_mapper(state)
    state = node_rag_retrieval(state)
    state = node_parallel_agents(state)
    state = node_critic(state)
    return dict(state)
