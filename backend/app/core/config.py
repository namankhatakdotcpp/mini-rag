from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized application configuration.
    All tunable parameters live here.
    """

    openai_api_key: str
    supabase_db_url: str

    # RAG parameters
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5

    # LLM parameters
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
