"""Local Ollama model construction."""

from langchain_ollama import ChatOllama

from .settings import settings


def get_chat_model() -> ChatOllama:
    """Return the application's local model without making a request yet."""
    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
