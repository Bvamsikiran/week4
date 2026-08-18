"""
utils/llm_factory.py
────────────────────
Returns a LangChain-compatible LLM based on configured provider.
Supports Groq and OpenAI with automatic fallback chains.

Upgrade:
  • Primary model → fallback model via .with_fallbacks()
  • If Groq 70B hits rate limits, falls back to Groq 8B or OpenAI mini.
"""

from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from utils.config import (
    LLM_PROVIDER, GROQ_API_KEY, OPENAI_API_KEY,
    GROQ_MODEL, OPENAI_MODEL,
)

# Groq fallback model (smaller, higher rate limit)
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"


def _build_groq(model: str, temperature: float) -> BaseChatModel:
    """Build a ChatGroq instance."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model,
        temperature=temperature,
    )


def _build_openai(model: str, temperature: float) -> BaseChatModel:
    """Build a ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model,
        temperature=temperature,
    )


def get_llm(temperature: float = 0.2, *, with_fallback: bool = True) -> BaseChatModel:
    """
    Factory function.  Returns a configured LangChain chat model
    with automatic fallback chain.

    Parameters
    ----------
    temperature   : float – Sampling temperature (0 = deterministic, 1 = creative).
    with_fallback : bool  – Attach a fallback model chain for resilience.
    """
    # ── Build primary model ───────────────────────────────────────────────
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        primary = _build_groq(GROQ_MODEL, temperature)

    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        primary = _build_openai(OPENAI_MODEL, temperature)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Choose 'groq' or 'openai' in your .env file."
        )

    # ── Build fallback chain ──────────────────────────────────────────────
    if not with_fallback:
        return primary

    fallbacks = []

    if LLM_PROVIDER == "groq" and GROQ_MODEL != GROQ_FALLBACK_MODEL:
        # Groq primary → smaller Groq model
        fallbacks.append(_build_groq(GROQ_FALLBACK_MODEL, temperature))

    if OPENAI_API_KEY and LLM_PROVIDER != "openai":
        # Cross-provider fallback: Groq → OpenAI
        fallbacks.append(_build_openai(OPENAI_MODEL, temperature))

    if GROQ_API_KEY and LLM_PROVIDER != "groq":
        # Cross-provider fallback: OpenAI → Groq
        fallbacks.append(_build_groq(GROQ_MODEL, temperature))

    if fallbacks:
        return primary.with_fallbacks(fallbacks)

    return primary
