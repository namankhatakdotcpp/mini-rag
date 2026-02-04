from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request payload for RAG queries.
    """

    question: str = Field(..., min_length=1, description="User question")


class RetrievedChunk(BaseModel):
    """
    A single retrieved chunk with optional metadata.
    """

    id: str
    content: str
    source: Optional[str] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    created_at: Optional[str] = None
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    """
    Response payload for RAG queries.
    """

    answer: str
    sources: List[RetrievedChunk]
