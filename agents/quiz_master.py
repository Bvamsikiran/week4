"""
agents/quiz_master.py
──────────────────────
AGENT 5 – Quiz Master

Generates:
  • 3–5 MCQs with 4 options each
  • 2 short-answer questions
  • Explanations for all answers

Returns a structured dict for clean UI rendering.
"""

from __future__ import annotations
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
QUIZ_MASTER_PROMPT = """You are the Quiz Master Agent for an Automata and Compiler Design (ACD) course.
Your job is to generate high-quality exam-style questions that test REAL understanding, not memorisation.

Generate a quiz in this EXACT JSON schema (output only JSON, no markdown fences):
{
  "mcqs": [
    {
      "q": "Question text",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "B",
      "explanation": "Why B is correct and others are not."
    }
  ],
  "short_answers": [
    {
      "q": "Question text",
      "answer": "Model answer",
      "hint": "A helpful hint for students"
    }
  ],
  "topic": "Topic name",
  "difficulty": "beginner | intermediate | advanced"
}

Rules:
- MCQs: Generate exactly 4 MCQs. All 4 options must be plausible (no joke options).
- Short answers: Generate exactly 2 questions.
- Questions must test conceptual understanding, NOT just definitions.
- Include at least one tricky question that tests common misconceptions.
- Explanations must be clear enough to teach from.
- Difficulty should match the syllabus_info provided."""


def run(
    query: str,
    syllabus_info: dict,
    context: str = "",
) -> dict:
    """
    Generate a quiz for the given ACD topic.

    Returns
    -------
    dict with keys: mcqs, short_answers, topic, difficulty
    """
    llm = get_llm(temperature=0.5)

    user_msg = f"""Topic: {syllabus_info.get('topic', 'Unknown')} ({syllabus_info.get('unit', 'Unknown')})
Difficulty: {syllabus_info.get('difficulty', 'intermediate')}
Student Question: {query}

{"--- Context ---" + chr(10) + context[:1500] if context else ""}

Generate a quiz for this topic."""

    messages = [
        SystemMessage(content=QUIZ_MASTER_PROMPT),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback: return the raw text wrapped
        return {
            "mcqs": [],
            "short_answers": [{"q": "See raw output below", "answer": raw, "hint": ""}],
            "topic": syllabus_info.get("topic", "ACD"),
            "difficulty": syllabus_info.get("difficulty", "intermediate"),
        }
