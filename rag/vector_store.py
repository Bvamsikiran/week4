"""
rag/vector_store.py
────────────────────
Thin wrapper around ChromaDB (default) or FAISS.
Provides:
  • build_or_load()     – create or reload persisted store
  • add_documents()     – ingest new documents
  • similarity_search() – k-NN retrieval

Upgrade:
  • @st.cache_resource for vector store initialisation
  • Deprecated .persist() call guarded properly
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document
from rag.embedder import get_embeddings, chunk_documents
from utils.config import VECTOR_STORE, CHROMA_PERSIST_DIR

# Globals — reuse across Streamlit reruns
_vectorstore = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


@st.cache_resource(show_spinner="Initialising vector store…")
def _create_chroma(collection_name: str):
    """Cached Chroma initialisation."""
    from langchain_community.vectorstores import Chroma
    persist_dir = str(Path(CHROMA_PERSIST_DIR))
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=persist_dir,
    )


def build_or_load(collection_name: str = "acd_mentor") -> object:
    """
    Build a new vector store or load an existing persisted one.
    Returns the vectorstore object.
    """
    global _vectorstore
    emb = _get_embeddings()

    if VECTOR_STORE == "chroma":
        _vectorstore = _create_chroma(collection_name)
    else:
        # FAISS — in-memory (no automatic persistence across runs)
        from langchain_community.vectorstores import FAISS
        # FAISS needs at least one document to initialise; use a placeholder
        placeholder = Document(page_content="ACD Mentor knowledge base initialised.")
        _vectorstore = FAISS.from_documents([placeholder], emb)

    return _vectorstore


def get_vectorstore():
    """Return current vectorstore, initialising if needed."""
    if _vectorstore is None:
        return build_or_load()
    return _vectorstore


def add_documents(docs: List[Document], pre_chunked: bool = False) -> int:
    """
    Chunk (if needed) and add documents to the store.
    Returns number of chunks added.
    """
    vs = get_vectorstore()
    chunks = docs if pre_chunked else chunk_documents(docs)
    if not chunks:
        return 0
    vs.add_documents(chunks)
    return len(chunks)


def similarity_search(
    query: str,
    k: int = 5,
    filter_source: str | None = None,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks.

    Parameters
    ----------
    query         : user query string
    k             : number of results
    filter_source : optional source file name filter (Chroma only)
    """
    vs = get_vectorstore()
    if filter_source and VECTOR_STORE == "chroma":
        return vs.similarity_search(
            query, k=k, filter={"source": filter_source}
        )
    return vs.similarity_search(query, k=k)


def get_all_documents() -> List[Document]:
    """
    Return all documents in the vector store (for BM25 index building).
    Works with Chroma; returns empty list for FAISS.
    """
    vs = get_vectorstore()
    if VECTOR_STORE == "chroma":
        try:
            collection = vs._collection
            result = collection.get(include=["documents", "metadatas"])
            docs = []
            for text, meta in zip(result["documents"], result["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta or {}))
            return docs
        except Exception:
            return []
    return []


def reset_vectorstore():
    """Clear the in-memory reference (force reload on next call)."""
    global _vectorstore
    _vectorstore = None
    _create_chroma.clear()
