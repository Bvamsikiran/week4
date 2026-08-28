"""
agents/memory_agent.py
───────────────────────
Mem0 Memory Agent — persists conversation facts across sessions.

Uses the Mem0 Python SDK (mem0ai) to:
  • add()    — store key facts from user messages
  • search() — retrieve relevant memories before answering

Fails silently when MEM0_API_KEY is not set or network is unreachable.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any

_mem_client = None  # lazy init


def _get_client():
    """Lazy-initialise the Mem0 MemoryClient (cached globally)."""
    global _mem_client
    if _mem_client is not None:
        return _mem_client

    api_key = os.getenv("MEM0_API_KEY", "").strip()
    if not api_key:
        return None  # no key → silent no-op

    try:
        from mem0 import MemoryClient  # type: ignore
        _mem_client = MemoryClient(api_key=api_key)
    except Exception as exc:
        print(f"[MemoryAgent] Init failed: {exc}")
        _mem_client = None

    return _mem_client


# ── Public helpers ────────────────────────────────────────────────────────────

def add_memory(user_id: str, message: str, role: str = "user") -> bool:
    """
    Store a message (or fact) in Mem0 for the given user.

    Parameters
    ----------
    user_id : str  — unique session/user identifier
    message : str  — text to remember
    role    : str  — "user" | "assistant"

    Returns True on success, False on failure / no-op.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        client.add(
            messages=[{"role": role, "content": message}],
            user_id=user_id,
        )
        return True
    except Exception as exc:
        print(f"[MemoryAgent] add failed: {exc}")
        return False


def search_memory(user_id: str, query: str, limit: int = 5) -> str:
    """
    Retrieve relevant past memories for a user given a query.

    Returns formatted string ready for LLM context, or "" if nothing found.
    """
    client = _get_client()
    if client is None:
        return ""
    try:
        results: List[Dict[str, Any]] = client.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        if not results:
            return ""
        lines = []
        for r in results:
            mem_text = r.get("memory", "")
            if mem_text:
                lines.append(f"• {mem_text}")
        return "Previous context:\n" + "\n".join(lines) if lines else ""
    except Exception as exc:
        print(f"[MemoryAgent] search failed: {exc}")
        return ""


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node: called on every turn.

    1. Searches Mem0 for relevant memories and injects them as `memory_context`.
    2. After the pipeline, call add_memory() with the user query.

    State keys consumed : query, user_id
    State keys produced : memory_context
    """
    query   = state.get("query", "")
    user_id = state.get("user_id", "anonymous")

    # Inject relevant memories into state
    memory_ctx = search_memory(user_id, query)
    state["memory_context"] = memory_ctx

    # Store the current query (facts stored after answering — cheap)
    add_memory(user_id, query, role="user")

    return state
