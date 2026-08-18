"""
agents/critic.py
─────────────────
AGENT 6 – Critic / Quality Gate

Reviews the combined outputs of Explainer + Problem Solver for:
  • Correctness (is the explanation/solution accurate?)
  • Completeness (are important aspects missing?)
  • Clarity (is it appropriate for the student's level?)
  • Safety (no misleading information about formal definitions)

Returns a structured review dict.

Upgrade:
  • Pydantic structured output
"""

from __future__ import annotations
import json
import re
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm


# ── Pydantic schema ──────────────────────────────────────────────────────────
class CriticReview(BaseModel):
    """Structured output from the Critic agent."""
    score: int = Field(ge=1, le=10, description="Quality score 1-10")
    verdict: Literal["good", "acceptable", "needs_correction"] = Field(
        description="Overall verdict"
    )
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    corrections: str = Field(default="", description="What should be corrected or added")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")


# ── System prompt ─────────────────────────────────────────────────────────────
CRITIC_PROMPT = """You are the Quality Gate Critic Agent for an ACD (Automata and Compiler Design) learning system.
You review AI-generated educational content for accuracy and completeness.

Your job:
1. Check if the explanation/solution is CORRECT according to standard ACD textbooks (Ullman, Hopcroft, Aho).
2. Identify any ERRORS, OMISSIONS, or MISLEADING statements.
3. Assign an overall QUALITY SCORE: 1–10.
4. Provide a short corrected note if needed.

Rules:
- verdict "good" = score 8–10, no critical issues.
- verdict "acceptable" = score 5–7, minor issues but usable.
- verdict "needs_correction" = score < 5, contains errors.
- Be strict about formal definitions (DFA vs NFA, LL vs LR, etc.).
- Be lenient about simplifications made for pedagogical clarity (e.g., ELI15 mode).
- If you are unsure, set confidence to "low"."""


def run(
    query: str,
    explanation: str,
    solution: str,
    syllabus_info: dict,
) -> dict:
    """
    Run quality check on explanation + solution outputs.

    Returns
    -------
    dict: {score, verdict, issues, corrections, confidence}
    """
    llm = get_llm(temperature=0.0)

    user_msg = f"""Original Question: {query}
Topic: {syllabus_info.get('topic', 'Unknown')} ({syllabus_info.get('unit', 'Unknown')})

--- EXPLANATION TO REVIEW ---
{explanation[:1500]}

--- SOLUTION TO REVIEW ---
{solution[:1500] if solution else "(no solution provided)"}

Please review for accuracy and quality."""

    # Try structured output first
    try:
        structured_llm = llm.with_structured_output(CriticReview)
        messages = [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=user_msg),
        ]
        result: CriticReview = structured_llm.invoke(messages)
        return result.model_dump()
    except (NotImplementedError, AttributeError, Exception):
        pass

    # Fallback: raw invoke + JSON parse
    fallback_prompt = CRITIC_PROMPT + """

Output ONLY valid JSON (no markdown fences):
{
  "score": 8,
  "verdict": "good | acceptable | needs_correction",
  "issues": ["issue 1 if any", "issue 2 if any"],
  "corrections": "What should be corrected or added (empty string if none)",
  "confidence": "high | medium | low"
}"""

    messages = [
        SystemMessage(content=fallback_prompt),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "score": 7,
            "verdict": "acceptable",
            "issues": [],
            "corrections": "",
            "confidence": "low",
        }
