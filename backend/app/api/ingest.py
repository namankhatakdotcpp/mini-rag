from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.client import get_db
from app.services.chunker import TextChunker
from app.services.embedder import Embedder

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

    Why Form/File?
    - FastAPI requires explicit Form() for multipart/form-data fields
    - Swagger UI uses multipart encoding by default
    """

    # Prefer text_input, fall back to text for compatibility
    raw_text = text_input if text_input is not None else text
    normalized_text = raw_text.strip() if raw_text is not None else None

    if file is None and not normalized_text:
        raise HTTPException(
            status_code=400,
            detail="Either text or file must be provided.",
        )

    # Step 1: Get raw content
    if file is not None:
        raw_bytes = await file.read()
        content = raw_bytes.decode("utf-8", errors="ignore").strip()
        source = file.filename or "upload"
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty after decoding.",
            )
    else:
        content = normalized_text
        source = "paste"
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Text input is empty after trimming.",
            )

    # Step 2: Chunk text
    chunker = TextChunker()
    chunks = chunker.chunk(content)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid text chunks generated.",
        )

    # Step 3: Generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    if not embeddings:
        raise HTTPException(
            status_code=500,
            detail="Embedding generation returned no vectors.",
        )

    # Step 4: Persist to database
    insert_stmt = text(
        """
        INSERT INTO documents (content, embedding, source, chunk_index)
        VALUES (:content, :embedding, :source, :chunk_index)
        """
    )

    try:
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
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to ingest content.") from exc

    return {
        "status": "success",
        "chunks_ingested": len(chunks),
        "source": source,
    }
