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

Upgrade:
  • Pydantic structured output
  • Mermaid sanitizer: strips markdown fences, validates basic syntax
  • ASCII fallback on invalid mermaid
"""

from __future__ import annotations
import json
import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm_factory import get_llm


# ── Pydantic schema ──────────────────────────────────────────────────────────
class VisualOutput(BaseModel):
    """Structured output from the Visualizer agent."""
    type: Literal["mermaid", "table", "ascii", "steps", "none"] = Field(
        description="Type of visual: mermaid, table, ascii, steps, or none"
    )
    content: str = Field(description="The diagram / table / steps content")
    caption: str = Field(description="Brief description of the visual")


# ── Mermaid Sanitizer ─────────────────────────────────────────────────────────
def clean_mermaid_code(raw_text: str) -> str:
    """
    Strip markdown code fences, clean whitespace, and validate
    that the result looks like valid Mermaid syntax.

    Returns clean Mermaid code or raises ValueError if invalid.
    """
    text = raw_text.strip()

    # Strip opening fence: ```mermaid, ```
    text = re.sub(r"^```(?:mermaid)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Strip closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    if not text:
        raise ValueError("Empty mermaid content after sanitisation")

    # Validate: must start with a known mermaid directive
    valid_starts = (
        "graph ", "graph\n",
        "flowchart ", "flowchart\n",
        "stateDiagram", "sequenceDiagram", "classDiagram",
        "gantt", "pie", "erDiagram", "journey",
        "gitgraph", "mindmap", "timeline",
    )
    if not any(text.lower().startswith(s.lower()) for s in valid_starts):
        # Might be wrapped in extra text — try to extract
        for start in valid_starts:
            idx = text.lower().find(start.lower())
            if idx != -1:
                text = text[idx:]
                break
        else:
            raise ValueError(f"Mermaid syntax not detected in: {text[:100]}...")

    # Remove any remaining markdown artifacts
    text = re.sub(r"```\w*", "", text)

    return text


# ── System prompt ─────────────────────────────────────────────────────────────
VISUALIZER_PROMPT = """You are the Visualizer Agent for an Automata and Compiler Design (ACD) course.
Your job is to generate CLEAR, ACCURATE visual representations of ACD concepts.

Choose the best visual format:

1. **Mermaid Diagram** — for DFA/NFA state machines, parse trees, syntax trees, flow graphs, compiler phase diagrams.
   Use valid Mermaid syntax. For state machines use `stateDiagram-v2` or `graph LR`. 
   Example DFA: 
   graph LR
       q0 -->|a| q1
       q1 -->|b| q2
       q2((q2))

2. **Markdown Table** — for transition tables (DFA/NFA), First/Follow sets, parsing tables.
   Use proper markdown table syntax with headers.

3. **ASCII / Code Block** — for 3-address code, parse trees in text form, DAG, stack traces.

4. **Numbered Steps** — for conversion algorithms (RE→NFA, NFA→DFA, minimization, LL(1) table construction).

IMPORTANT: Do NOT wrap mermaid code in triple backticks. Output raw mermaid syntax in the content field.

Rules:
- Mermaid MUST be syntactically valid (no backtick fences).
- Tables MUST have proper markdown | headers |.
- Steps MUST be numbered and self-explanatory.
- Do NOT add explanation text — just the visual + caption.
- If the topic does not lend itself to a visual, set type to "none"."""


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

    # Try structured output first
    try:
        structured_llm = llm.with_structured_output(VisualOutput)
        messages = [
            SystemMessage(content=VISUALIZER_PROMPT),
            HumanMessage(content=user_msg),
        ]
        result: VisualOutput = structured_llm.invoke(messages)
        output = result.model_dump()

        # Sanitize mermaid if needed
        if output["type"] == "mermaid" and output["content"]:
            try:
                output["content"] = clean_mermaid_code(output["content"])
            except ValueError:
                output["type"] = "ascii"

        return output
    except (NotImplementedError, AttributeError, Exception):
        pass

    # Fallback: raw invoke + JSON parse
    fallback_prompt = VISUALIZER_PROMPT + """

Output ONLY valid JSON:
{
  "type": "mermaid | table | ascii | steps | none",
  "content": "the diagram / table / steps here",
  "caption": "Brief description"
}"""

    messages = [
        SystemMessage(content=fallback_prompt),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)

        # Sanitize mermaid
        if parsed.get("type") == "mermaid" and parsed.get("content"):
            try:
                parsed["content"] = clean_mermaid_code(parsed["content"])
            except ValueError:
                parsed["type"] = "ascii"

        return parsed
    except (json.JSONDecodeError, KeyError):
        return {
            "type": "ascii",
            "content": raw,
            "caption": f"Diagram for: {syllabus_info.get('topic', query)}",
        }
