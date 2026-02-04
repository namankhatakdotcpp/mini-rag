from typing import List
from openai import OpenAI
from app.core.config import settings


class Embedder:
    """
    Responsible for converting text chunks into vector embeddings.
    Thin abstraction to allow swapping providers later.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Backwards-compatible wrapper used by older ingestion code.
        """
        return self.embed(texts)
