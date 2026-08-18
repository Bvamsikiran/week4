"""
rag/embedder.py
────────────────
Chunking + embedding pipeline.
Supports:
  • Local embeddings via sentence-transformers (free, no API needed)
  • OpenAI text-embedding-3-small (requires OPENAI_API_KEY)

Upgrade:
  • Wrapped with @st.cache_resource to avoid reloading weights on reruns.
"""

from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import EMBEDDING_PROVIDER, OPENAI_API_KEY

import streamlit as st


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    """Return a LangChain embeddings instance based on config (cached)."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
    else:
        # Free local embeddings — no API key needed
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)
