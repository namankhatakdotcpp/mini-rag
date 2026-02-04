from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized application configuration.
    All tunable parameters live here.
    """

    openai_api_key: Optional[str] = None
    supabase_db_url: str

    # Provider selection
    embedding_provider: str = "ollama"
    chat_provider: str = "ollama"

    # Ollama (local) settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3"

    # RAG parameters
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5

    # LLM parameters (OpenAI)
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0.2
    chat_max_tokens: int = 500

    # Prompting
    rag_system_prompt: str = (
        "You are a careful assistant for a retrieval-augmented system. "
        "Use only the provided context to answer. "
        "If the answer is not contained in the context, say you don't know. "
        "Cite sources using the tags like [S1], [S2]."
    )

    class Config:
        env_file = ".env"


settings = Settings()
