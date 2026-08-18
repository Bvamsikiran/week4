"""
agents/syllabus_mapper.py
──────────────────────────
AGENT 1 – Syllabus Mapper

Maps the user's question to the most relevant ACD unit and topic.
Returns structured JSON so downstream agents know exactly what to focus on.
"""

from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
SYLLABUS_MAPPER_PROMPT = """You are the Syllabus Mapper Agent for an Automata and Compiler Design (ACD) course.

Your ONLY job is to analyse the student's question and identify:
1. The most relevant UNIT from the ACD syllabus.
2. The specific TOPIC within that unit.
3. A brief QUERY REFINEMENT — a cleaner, more specific version of the student's question.
4. DIFFICULTY level: beginner | intermediate | advanced

ACD Syllabus:
UNIT I  – Formal Languages, Regular Expressions, DFA, NFA, RE→NFA→DFA conversions, Pumping Lemma, Lex
UNIT II – Compiler Phases, Lexical Analysis, CFG, Parse Trees, Ambiguity, LL(1), Bottom-Up, LR, LALR, YACC
UNIT III– Syntax-Directed Translation, S-attributed/L-attributed grammars, Intermediate Code, AST, 3-address code
UNIT IV – Runtime Environments, Storage Organization, Code Optimization, Peephole, Flow Graphs, Basic Blocks
UNIT V  – Code Generation, Register Allocation, DAG Representation of Basic Blocks

Output ONLY valid JSON in this exact schema (no markdown, no extra text):
{
  "unit": "Unit I | Unit II | Unit III | Unit IV | Unit V | General",
  "unit_number": 1,
  "topic": "<specific topic name>",
  "refined_query": "<cleaner rephrasing of the question>",
  "difficulty": "beginner | intermediate | advanced",
  "keywords": ["kw1", "kw2", "kw3"]
}"""


def run(query: str) -> dict:
    """
    Map a student query to the ACD syllabus.

    Returns a dict with keys: unit, unit_number, topic, refined_query,
    difficulty, keywords.
    """
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=SYLLABUS_MAPPER_PROMPT),
        HumanMessage(content=f"Student's question: {query}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse JSON — be defensive
    try:
        # Strip possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model didn't return clean JSON
        return {
            "unit": "General",
            "unit_number": 0,
            "topic": "ACD General",
            "refined_query": query,
            "difficulty": "intermediate",
            "keywords": [],
        }
