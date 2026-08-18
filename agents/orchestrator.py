"""
agents/orchestrator.py
───────────────────────
Multi-agent orchestration pipeline.

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
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import agents.syllabus_mapper as syllabus_mapper
import agents.explainer as explainer
import agents.visualizer as visualizer
import agents.problem_solver as problem_solver
import agents.quiz_master as quiz_master
import agents.critic as critic
from rag.retriever import retrieve_context


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
      syllabus_info, context, citations,
      explanation, visual, solution, quiz, review
    """
    result: Dict[str, Any] = {
        "query": query,
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

    try:
        # ── Step 1: Syllabus Mapping ──────────────────────────────────────────
        result["syllabus_info"] = syllabus_mapper.run(query)
        syllabus_info = result["syllabus_info"]

        # ── Step 2: RAG Retrieval ─────────────────────────────────────────────
        refined_q = syllabus_info.get("refined_query", query)
        context, citations = retrieve_context(
            refined_q,
            k=6,
            use_only_user_docs=use_only_user_docs,
        )
        result["context"] = context
        result["citations"] = citations

        # ── Step 3: Parallel agent execution ─────────────────────────────────
        tasks = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Always run explainer
            tasks["explanation"] = executor.submit(
                explainer.run,
                query, syllabus_info, context, eli15, show_mistakes
            )

            if run_visualizer:
                tasks["visual"] = executor.submit(
                    visualizer.run, query, syllabus_info, context
                )
            if run_solver:
                tasks["solution"] = executor.submit(
                    problem_solver.run, query, syllabus_info, context
                )
            if run_quiz:
                tasks["quiz"] = executor.submit(
                    quiz_master.run, query, syllabus_info, context
                )

            # Collect results
            for key, future in tasks.items():
                try:
                    result[key] = future.result(timeout=60)
                except Exception as exc:
                    result[key] = f"⚠️ Agent error: {exc}"

        # ── Step 4: Critic (sequential — needs explainer + solver outputs) ───
        if run_critic:
            explanation_str = result.get("explanation", "")
            solution_str = result.get("solution", "")
            if isinstance(explanation_str, str) and isinstance(solution_str, str):
                result["review"] = critic.run(
                    query, explanation_str, solution_str, syllabus_info
                )

    except Exception as exc:
        result["error"] = str(exc)

    return result
