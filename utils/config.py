"""
utils/config.py
───────────────
Central configuration loader.
Reads from .env file (local) or Streamlit secrets (cloud).
"""

import os
from pathlib import Path

# Try to load .env file if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed — rely on environment variables

# Try to pull from Streamlit secrets (cloud deploy)
try:
    import streamlit as st
    _secrets = st.secrets
    def _get(key: str, default: str = "") -> str:
        try:
            return _secrets[key]
        except (KeyError, Exception):
            return os.getenv(key, default)
except Exception:
    def _get(key: str, default: str = "") -> str:
        return os.getenv(key, default)


# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = _get("GROQ_API_KEY", "")
OPENAI_API_KEY: str = _get("OPENAI_API_KEY", "")
LLM_PROVIDER: str = _get("LLM_PROVIDER", "groq")   # "groq" | "openai"
GROQ_MODEL: str = _get("GROQ_MODEL", "qwen/qwen3.6-27b")
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o-mini")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = _get("EMBEDDING_PROVIDER", "local")  # "local" | "openai"

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTOR_STORE: str = _get("VECTOR_STORE", "chroma")  # "chroma" | "faiss"
CHROMA_PERSIST_DIR: str = _get("CHROMA_PERSIST_DIR", "./data/chroma_db")

# ── Misc ─────────────────────────────────────────────────────────────────────
ENABLE_WEB_KNOWLEDGE: bool = _get("ENABLE_WEB_KNOWLEDGE", "true").lower() == "true"
UPLOADED_DOCS_DIR: str = "./data/uploaded_docs"

# ── Derived: active model name ────────────────────────────────────────────────
def active_model() -> str:
    return GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL

def active_api_key() -> str:
    return GROQ_API_KEY if LLM_PROVIDER == "groq" else OPENAI_API_KEY

def has_valid_api_key() -> bool:
    return bool(active_api_key().strip())
