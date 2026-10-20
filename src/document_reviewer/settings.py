"""Central configuration. Environment variables keep local settings out of source control."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    max_retries: int = 2
    rag_persist_directory: str = "chroma_db"
    rag_collection_name: str = "technical_references"


settings = Settings()
