"""
rag/retriever.py
─────────────────
Hybrid retrieval:
  • User-uploaded documents
  • GFG external knowledge (if enabled)

Returns a formatted context string + list of source citations.
"""

from __future__ import annotations
from typing import Tuple, List

from langchain_core.documents import Document
from rag.vector_store import similarity_search


def retrieve_context(
    query: str,
    k: int = 6,
    use_only_user_docs: bool = False,
) -> Tuple[str, List[str]]:
    """
    Retrieve relevant chunks and format them as a context block.

    Returns
    -------
    context_text : str   — formatted text ready to insert into a prompt
    citations    : list  — unique source names for display
    """
    docs: List[Document] = similarity_search(query, k=k)

    if use_only_user_docs:
        docs = [d for d in docs if d.metadata.get("type") in ("pdf", "pptx", "text")]

    if not docs:
        return "", []

    parts = []
    citations = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", doc.metadata.get("slide", ""))
        page_str = f", page {page}" if page else ""
        parts.append(f"[{i}] ({src}{page_str}):\n{doc.page_content}")
        label = f"{src}{page_str}"
        if label not in citations:
            citations.append(label)

    context_text = "\n\n---\n\n".join(parts)
    return context_text, citations
