from typing import List, Dict
from openai import OpenAI
from app.core.config import settings
from app.services.prompts import build_rag_messages


class LLMClient:
    """
    Thin wrapper around OpenAI Chat Completions for RAG answering.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model
        self.temperature = settings.chat_temperature
        self.max_tokens = settings.chat_max_tokens
        self.system_prompt = settings.rag_system_prompt

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

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        return response.choices[0].message.content.strip()
