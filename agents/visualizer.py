"""
agents/visualizer.py
─────────────────────
AGENT 3 – Visualizer

Generates:
  • Mermaid diagrams (state machines, parse trees, flow graphs)
  • ASCII conversion tables (NFA → DFA, First/Follow, etc.)
  • Step-by-step structured output for complex constructions

Returns a dict with keys:
  "type"    : "mermaid" | "table" | "ascii" | "steps"
  "content" : the raw diagram / table / steps string
  "caption" : brief description of the visual
"""

from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
VISUALIZER_PROMPT = """You are the Visualizer Agent for an Automata and Compiler Design (ACD) course.
Your job is to generate CLEAR, ACCURATE visual representations of ACD concepts.

Choose the best visual format:

1. **Mermaid Diagram** — for DFA/NFA state machines, parse trees, syntax trees, flow graphs, compiler phase diagrams.
   Use valid Mermaid syntax. For state machines use `stateDiagram-v2` or `graph LR`. 
   Example DFA: 
   ```mermaid
   graph LR
       q0 -->|a| q1
       q1 -->|b| q2
       q2((q2))
   ```

2. **Markdown Table** — for transition tables (DFA/NFA), First/Follow sets, parsing tables.
   Use proper markdown table syntax with headers.

3. **ASCII / Code Block** — for 3-address code, parse trees in text form, DAG, stack traces.

4. **Numbered Steps** — for conversion algorithms (RE→NFA, NFA→DFA, minimization, LL(1) table construction).

Output ONLY valid JSON:
{
  "type": "mermaid | table | ascii | steps",
  "content": "the diagram / table / steps here",
  "caption": "Brief description, e.g., 'DFA for language L = a*b*'"
}

Rules:
- Mermaid MUST be syntactically valid.
- Tables MUST have proper markdown | headers |.
- Steps MUST be numbered and self-explanatory.
- Do NOT add explanation text — just the visual + caption.
- If the topic does not lend itself to a visual, output: {"type": "none", "content": "", "caption": "No diagram needed for this topic."}"""


def run(
    query: str,
    syllabus_info: dict,
    context: str = "",
) -> dict:
    """
    Generate a diagram, table, or step list for the given ACD topic.

    Returns
    -------
    dict: {"type": str, "content": str, "caption": str}
    """
    llm = get_llm(temperature=0.1)

    user_msg = f"""Topic: {syllabus_info.get('topic', 'Unknown')} ({syllabus_info.get('unit', 'Unknown')})
Question: {query}

{"--- Retrieved Context ---" + chr(10) + context[:1500] if context else ""}

Generate the best visual representation for this topic/question."""

    messages = [
        SystemMessage(content=VISUALIZER_PROMPT),
        HumanMessage(content=user_msg),
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
        # Fallback: return as an ASCII block
        return {
            "type": "ascii",
            "content": raw,
            "caption": f"Diagram for: {syllabus_info.get('topic', query)}",
        }
