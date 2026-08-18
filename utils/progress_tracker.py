"""
utils/progress_tracker.py
──────────────────────────
Tracks which units / topics the student has interacted with.
State lives in Streamlit session_state so it persists across reruns.
"""

import streamlit as st
from typing import Dict, List


# ── Unit metadata ─────────────────────────────────────────────────────────────
UNIT_MAP: Dict[str, List[str]] = {
    "Unit I – Formal Languages & Automata": [
        "Formal Languages", "Regular Expressions", "DFA", "NFA",
        "RE → NFA", "NFA → DFA", "Minimization", "Pumping Lemma", "Lex",
    ],
    "Unit II – Parsing & Grammars": [
        "Compiler Phases", "Lexical Analysis", "CFG", "Parse Trees",
        "Ambiguity", "LL(1)", "First & Follow", "Bottom-Up Parsing",
        "LR Parsing", "LALR", "YACC",
    ],
    "Unit III – Syntax-Directed Translation": [
        "SDT Overview", "S-Attributed Grammars", "L-Attributed Grammars",
        "Intermediate Code", "AST", "3-Address Code",
        "Translation of Assignments", "Translation of Control Flow",
    ],
    "Unit IV – Runtime & Optimization": [
        "Runtime Environments", "Storage Organization",
        "Static/Dynamic Allocation", "Code Optimization",
        "Peephole Optimization", "Flow Graphs", "Basic Blocks",
    ],
    "Unit V – Code Generation": [
        "Code Generation Overview", "Register Allocation",
        "DAG Representation", "Instruction Selection",
    ],
}


def _init():
    """Initialise progress state in session if not already present."""
    if "progress" not in st.session_state:
        st.session_state.progress = {unit: set() for unit in UNIT_MAP}


def mark_topic(unit: str, topic: str):
    """Mark a topic as practiced."""
    _init()
    if unit in st.session_state.progress:
        st.session_state.progress[unit].add(topic)


def get_progress() -> Dict[str, Dict]:
    """Return completion stats per unit."""
    _init()
    result = {}
    for unit, topics in UNIT_MAP.items():
        done = st.session_state.progress.get(unit, set())
        result[unit] = {
            "total": len(topics),
            "done": len(done),
            "topics": topics,
            "practiced": list(done),
            "pct": int(100 * len(done) / max(len(topics), 1)),
        }
    return result


def reset_progress():
    """Reset all progress."""
    st.session_state.progress = {unit: set() for unit in UNIT_MAP}
