"""
utils/llm_factory.py
────────────────────
Returns a LangChain-compatible LLM based on configured provider.
Supports Groq and OpenAI.  Easily extendable to Anthropic / Ollama.
"""

from langchain_core.language_models import BaseChatModel
from utils.config import LLM_PROVIDER, GROQ_API_KEY, OPENAI_API_KEY, GROQ_MODEL, OPENAI_MODEL


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """
    Factory function.  Returns a configured LangChain chat model.

    Parameters
    ----------
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature,
        )

    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Choose 'groq' or 'openai' in your .env file."
        )
