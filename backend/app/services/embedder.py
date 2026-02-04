from typing import List

import httpx
from openai import OpenAI

from app.core.config import settings


class Embedder:
    """
    Responsible for converting text chunks into vector embeddings.
    Thin abstraction to allow swapping providers later.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        url = f"{settings.ollama_base_url.rstrip('/')}/api/embeddings"

        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                resp = client.post(
                    url,
                    json={
                        "model": settings.ollama_embed_model,
                        "prompt": text,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embedding"])

        return embeddings

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("OpenAI API key not configured.")
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        """
        provider = settings.embedding_provider.lower()
        if provider == "ollama":
            return self._embed_ollama(texts)
        if provider == "openai":
            return self._embed_openai(texts)
        raise RuntimeError(f"Unknown embedding_provider: {settings.embedding_provider}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Backwards-compatible wrapper used by older ingestion code.
        """
        return self.embed(texts)
