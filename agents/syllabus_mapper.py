"""
agents/syllabus_mapper.py
──────────────────────────
AGENT 1 – Syllabus Mapper

Maps the user's question to the most relevant ACD unit and topic.
Returns structured output via Pydantic model — no fragile regex/JSON parsing.
"""

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── Pydantic schema ──────────────────────────────────────────────────────────
class SyllabusMapping(BaseModel):
    """Structured output from the Syllabus Mapper agent."""
    unit: str = Field(description="Unit name, e.g. 'Unit I', 'Unit II', ... or 'General'")
    unit_number: int = Field(description="Unit number 1-5, or 0 for General")
    topic: str = Field(description="Specific topic name within the unit")
    refined_query: str = Field(description="Cleaner, more specific rephrasing of the student's question")
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(default="intermediate")
    keywords: List[str] = Field(default_factory=list, description="3-5 relevant search keywords")


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
UNIT V  – Code Generation, Register Allocation, DAG Representation of Basic Blocks"""


def run(query: str) -> dict:
    """
    Map a student query to the ACD syllabus.

    Returns a dict with keys: unit, unit_number, topic, refined_query,
    difficulty, keywords.
    """
    llm = get_llm(temperature=0.0)

    # Try structured output first (requires model support)
    try:
        structured_llm = llm.with_structured_output(SyllabusMapping)
        messages = [
            SystemMessage(content=SYLLABUS_MAPPER_PROMPT),
            HumanMessage(content=f"Student's question: {query}"),
        ]
        result: SyllabusMapping = structured_llm.invoke(messages)
        return result.model_dump()
    except (NotImplementedError, AttributeError, Exception):
        pass

    # Fallback: raw invoke + JSON parse
    import json
    fallback_prompt = SYLLABUS_MAPPER_PROMPT + """

Output ONLY valid JSON in this exact schema (no markdown, no extra text):
{
  "unit": "Unit I | Unit II | Unit III | Unit IV | Unit V | General",
  "unit_number": 1,
  "topic": "<specific topic name>",
  "refined_query": "<cleaner rephrasing of the question>",
  "difficulty": "beginner | intermediate | advanced",
  "keywords": ["kw1", "kw2", "kw3"]
}"""

    messages = [
        SystemMessage(content=fallback_prompt),
        HumanMessage(content=f"Student's question: {query}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "unit": "General",
            "unit_number": 0,
            "topic": "ACD General",
            "refined_query": query,
            "difficulty": "intermediate",
            "keywords": [],
        }
