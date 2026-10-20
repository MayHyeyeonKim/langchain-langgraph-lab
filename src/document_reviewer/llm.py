"""Local Ollama model construction."""

from langchain_ollama import ChatOllama

from .settings import settings


def get_chat_model() -> ChatOllama:
    """Return the application's local model without making a request yet."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        reasoning=False,
        num_predict=512, #LLM이 한 번에 생성할 수 있는 최대 토큰 수를 절반으로 줄임, 장:응답이 빨라질 수 있음, 단:응답이 끊길 수 있음
    )
