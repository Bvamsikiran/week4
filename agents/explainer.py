"""
agents/explainer.py
────────────────────
AGENT 2 – Explainer

Provides clear, analogy-rich explanations of ACD concepts.
Supports "Explain like I'm 15" (ELI15) mode and "Show common mistakes".
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
EXPLAINER_PROMPT = """You are the Explainer Agent for an Automata and Compiler Design (ACD) course.
Your superpower is making complex CS theory feel intuitive and approachable.

Guidelines:
- Start with a one-sentence plain-English summary.
- Use real-world analogies (traffic lights, vending machines, recipes, etc.).
- Build from simple to complex — never assume prior knowledge.
- Use bullet points and short paragraphs (not walls of text).
- Highlight KEY TERMS in **bold**.
- End with a "💡 Quick Memory Hook" — a memorable trick or phrase.

If context from study materials is provided, weave it in naturally and cite it.

If the ELI15 flag is set, imagine you are explaining to a curious 15-year-old:
- Use even simpler language, relatable analogies (games, daily life).
- Avoid jargon entirely; introduce technical terms gently with definitions.

If "show common mistakes" flag is set, add a section:
⚠️ Common Mistakes Students Make:
- [mistake 1 and correction]
- [mistake 2 and correction]

Always end with: 📌 Key Takeaway: [one sentence]"""


def run(
    query: str,
    syllabus_info: dict,
    context: str = "",
    eli15: bool = False,
    show_mistakes: bool = False,
) -> str:
    """
    Generate a clear explanation of an ACD topic.

    Parameters
    ----------
    query         : student's original question
    syllabus_info : output from syllabus_mapper.run()
    context       : retrieved RAG context (may be empty)
    eli15         : if True, use simplified "explain to a 15 y/o" mode
    show_mistakes : if True, append common mistakes section

    Returns
    -------
    str : markdown-formatted explanation
    """
    llm = get_llm(temperature=0.3)

    # Build user message
    flags = []
    if eli15:
        flags.append("🔵 ELI15 MODE: Use very simple language and relatable analogies.")
    if show_mistakes:
        flags.append("🔴 Include a 'Common Mistakes' section at the end.")

    user_msg = f"""Topic: {syllabus_info.get('topic', 'Unknown')} ({syllabus_info.get('unit', 'Unknown')})
Difficulty: {syllabus_info.get('difficulty', 'intermediate')}
Question: {query}

{"Flags: " + " | ".join(flags) if flags else ""}

{"--- Retrieved Context ---" + chr(10) + context if context else ""}

Please explain this concept clearly."""

    messages = [
        SystemMessage(content=EXPLAINER_PROMPT),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    return response.content
