"""
agents/problem_solver.py
─────────────────────────
AGENT 4 – Problem Solver

Solves ACD practice problems step by step.
Handles:
  • Regex / DFA / NFA construction and conversion
  • Grammar construction, First/Follow sets
  • LL(1) / LR parsing table construction
  • 3-address code generation
  • Code optimization and DAG construction
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
PROBLEM_SOLVER_PROMPT = """You are the Problem Solver Agent for an Automata and Compiler Design (ACD) course.
You are a patient, methodical teacher who solves problems step-by-step.

Your approach for EVERY problem:
1. **Understand** – Restate the problem in simple terms.
2. **Identify** – Name the algorithm or technique to use.
3. **Execute** – Walk through each step, numbered, with intermediate states shown.
4. **Verify** – Check the answer / perform a quick sanity test.
5. **Summarise** – One-line answer and key insight.

Formatting rules:
- Number all steps clearly: Step 1, Step 2, etc.
- Show intermediate automata states, sets, or tables explicitly.
- Use code blocks for formal notation (states, transitions, productions).
- For DFA/NFA: explicitly show the transition function δ.
- For parsing: explicitly show stack/input/action traces.
- For 3-address code: show each instruction on a separate line.
- For optimization: show BEFORE and AFTER code blocks.

NEVER skip steps. A student should be able to follow your solution without external help.
If context from study materials is provided, reference it where relevant.

End every solution with:
✅ Final Answer: [concise result]
🧠 Key Insight: [the one thing to remember]"""


def run(
    query: str,
    syllabus_info: dict,
    context: str = "",
) -> str:
    """
    Solve a step-by-step ACD problem.

    Parameters
    ----------
    query         : the problem statement
    syllabus_info : from syllabus_mapper.run()
    context       : retrieved RAG context

    Returns
    -------
    str : markdown-formatted step-by-step solution
    """
    llm = get_llm(temperature=0.1)

    user_msg = f"""Topic: {syllabus_info.get('topic', 'Unknown')} ({syllabus_info.get('unit', 'Unknown')})
Problem: {query}

{"--- Retrieved Context / Definitions ---" + chr(10) + context[:2000] if context else ""}

Solve this step by step."""

    messages = [
        SystemMessage(content=PROBLEM_SOLVER_PROMPT),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    return response.content
