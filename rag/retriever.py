"""
rag/retriever.py
─────────────────
Hybrid retrieval:
  • Dense vector search (Chroma/FAISS)
  • Sparse BM25 keyword search
  • Reciprocal Rank Fusion (RRF) to merge results

Returns a formatted context string + list of source citations.
"""

from __future__ import annotations
from typing import Tuple, List, Dict
from collections import defaultdict

from langchain_core.documents import Document
from rag.vector_store import similarity_search, get_all_documents


def _build_bm25_results(query: str, k: int = 10) -> List[Document]:
    """
    Simple BM25-style keyword retrieval over all stored documents.
    Uses rank_bm25 if available, otherwise falls back to naive TF matching.
    """
    all_docs = get_all_documents()
    if not all_docs:
        return []

    try:
        from rank_bm25 import BM25Okapi
        corpus = [doc.page_content.lower().split() for doc in all_docs]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        scored = list(zip(scores, all_docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
    except ImportError:
        # Fallback: naive keyword overlap scoring
        query_terms = set(query.lower().split())
        scored = []
        for doc in all_docs:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


def _reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)  for each list where the doc appears.
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # Use page_content hash as key for deduplication
            key = hash(doc.page_content[:200])
            doc_scores[key] += 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys]


def retrieve_context(
    query: str,
    k: int = 6,
    use_only_user_docs: bool = False,
) -> Tuple[str, List[str]]:
    """
    Retrieve relevant chunks using hybrid search (dense + BM25 + RRF)
    and format them as a context block.

    Returns
    -------
    context_text : str   — formatted text ready to insert into a prompt
    citations    : list  — unique source names for display
    """
    # Dense vector search
    dense_results: List[Document] = similarity_search(query, k=k * 2)

    # Sparse BM25 search
    bm25_results: List[Document] = _build_bm25_results(query, k=k * 2)

    # Reciprocal Rank Fusion
    if bm25_results:
        docs = _reciprocal_rank_fusion([dense_results, bm25_results])[:k]
    else:
        docs = dense_results[:k]

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
