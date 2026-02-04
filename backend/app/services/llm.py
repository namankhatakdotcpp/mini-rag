from typing import List, Dict

import httpx
from openai import OpenAI

from app.core.config import settings
from app.services.prompts import build_rag_messages


class LLMClient:
    """
    Thin wrapper around LLM chat completion for RAG answering.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.chat_model
        self.temperature = settings.chat_temperature
        self.max_tokens = settings.chat_max_tokens
        self.system_prompt = settings.rag_system_prompt

    def _chat_ollama(self, messages: List[Dict]) -> str:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()

    def _chat_openai(self, messages: List[Dict]) -> str:
        if not self.client:
            raise RuntimeError("OpenAI API key not configured.")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Build a grounded prompt from retrieved context and get a response.
        """
        if not context_chunks:
            return (
                "I don't know based on the provided documents. "
                "Please ingest relevant content and try again."
            )

        messages = build_rag_messages(question, context_chunks, self.system_prompt)

        provider = settings.chat_provider.lower()
        if provider == "ollama":
            return self._chat_ollama(messages)
        if provider == "openai":
            return self._chat_openai(messages)
        raise RuntimeError(f"Unknown chat_provider: {settings.chat_provider}")
