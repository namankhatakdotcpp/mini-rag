from typing import List
import tiktoken
from app.core.config import settings


class TextChunker:
    """
    Responsible for splitting raw text into overlapping chunks
    suitable for embedding and retrieval.
    """

    def __init__(self):
        # Using OpenAI-compatible tokenizer for accurate token counts
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count.
        """
        tokens = self.tokenizer.encode(text)
        chunks: List[str] = []

        start = 0
        while start < len(tokens):
            end = start + settings.chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))

            # Move forward with overlap
            start += settings.chunk_size - settings.chunk_overlap

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Backwards-compatible wrapper used by older ingestion code.
        """
        return self.chunk(text)
