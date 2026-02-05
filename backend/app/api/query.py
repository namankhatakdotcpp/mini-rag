from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import os
import requests
from typing import List

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for the given text using Hugging Face Inference API.
    """
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token is not set.")
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        embedding = response.json()
        if isinstance(embedding, list) and len(embedding) > 0:
            return embedding[0]  # Return the first embedding vector
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from embedding service.")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=500, detail="Embedding service unavailable.")

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Generate an answer using the Hugging Face Inference API.
    """
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token is not set.")
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    context = "\n".join(context_chunks)
    prompt = f"Answer the question using ONLY the context below:\n\n{context}\n\nQuestion: {question}"
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_LLM_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from answer generation service.")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=500, detail="Answer generation service unavailable.")

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
