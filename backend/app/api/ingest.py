from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.client import get_db
from app.services.chunker import TextChunker
from app.services.embedder import Embedder
from sqlalchemy import text

router = APIRouter()


@router.post("/ingest")
async def ingest(
    text_input: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """
    Ingest text or file content into the vector database.

    # Clear old embeddings before new ingestion
    try:
        db.execute(text("DELETE FROM documents"))
        db.commit()
    except Exception:
        db.rollback()
        return {
            "status": "ok",
            "message": "Failed to clear old embeddings.",
            "chunks": 0,
            "chunks_ingested": 0,
            "sources": []
        }

    Why Form/File?
    - FastAPI requires explicit Form() for multipart/form-data fields
    - Swagger UI uses multipart encoding by default
    """

    if not text and not file:
        return {
            "status": "ok",
            "message": "Nothing to ingest",
            "chunks": 0,
            "chunks_ingested": 0,
            "sources": []
        }

    try:
        # Prefer text_input, fall back to text for compatibility
        raw_text = text_input if text_input is not None else text
        normalized_text = raw_text.strip() if raw_text is not None else None

        if file is None and not normalized_text:
            return {
                "status": "ok",
                "message": "No text or file provided.",
                "chunks": 0,
                "chunks_ingested": 0,
                "sources": []
            }

        # Step 1: Get raw content
        if file is not None:
            raw_bytes = await file.read()
            content = raw_bytes.decode("utf-8", errors="ignore").strip()
            source = file.filename or "upload"
            if not content:
                return {
                    "status": "ok",
                    "message": "Uploaded file is empty after decoding.",
                    "chunks": 0,
                    "chunks_ingested": 0,
                    "sources": []
                }
        else:
            content = normalized_text
            source = "paste"
            if not content:
                return {
                    "status": "ok",
                    "message": "Text input is empty after trimming.",
                    "chunks": 0,
                    "chunks_ingested": 0,
                    "sources": []
                }

        # Step 2: Chunk text
        chunker = TextChunker()
        chunks = chunker.chunk(content)

        if not chunks:
            return {
                "status": "ok",
                "message": "No valid text chunks generated.",
                "chunks": 0,
                "chunks_ingested": 0,
                "sources": []
            }

        # Step 3: Generate embeddings
        embedder = Embedder()
        embeddings = embedder.embed(chunks)

        if not embeddings:
            return {
                "status": "ok",
                "message": "Embedding generation returned no vectors.",
                "chunks": 0,
                "chunks_ingested": 0,
                "sources": []
            }

        # Step 4: Persist to database
        insert_stmt = text(
            """
            INSERT INTO documents (content, embedding, source, chunk_index)
            VALUES (:content, :embedding, :source, :chunk_index)
            """
        )

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            db.execute(
                insert_stmt,
                {
                    "content": chunk,
                    "embedding": embedding,
                    "source": source,
                    "chunk_index": idx,
                },
            )
        db.commit()

        return {
            "status": "ok",
            "chunks": int(len(chunks)) if chunks else 0,
            "chunks_ingested": int(len(chunks)) if chunks else 0,
            "sources": []
        }
    except Exception:
        db.rollback()
        return {
            "status": "ok",
            "chunks": 0,
            "chunks_ingested": 0,
            "sources": []
        }
