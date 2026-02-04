from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.client import get_db
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk
from app.services.embedder import Embedder
from app.services.retriever import Retriever
from app.services.llm import LLMClient

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest, db: Session = Depends(get_db)):
    """
    Run a full RAG query:
    1) Embed the user question
    2) Retrieve top-k similar chunks
    3) Construct a grounded prompt
    4) Generate an answer with citations
    """

    question = payload.question.strip() if payload.question else ""
    if not question:
        return {
            "answer": "No question provided.",
            "sources": []
        }

    try:
        # Step 1: Embed the query
        embedder = Embedder()
        query_embedding = embedder.embed([question])[0]

        # Step 2: Retrieve similar chunks
        retriever = Retriever(db)
        retrieved = retriever.search(query_embedding, settings.top_k)

        # Step 3: Generate grounded answer
        llm = LLMClient()
        answer = llm.generate_answer(question, retrieved)

        # Step 4: Build response
        sources = [RetrievedChunk(**chunk) for chunk in retrieved]

        return QueryResponse(answer=answer, sources=sources)
    except Exception:
        return {
            "answer": "Query failed.",
            "sources": []
        }
