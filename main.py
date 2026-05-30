"""
FastAPI server for the Mini-RAG system.

Endpoints:
  GET  /          → serve index.html
  GET  /health    → liveness probe
  POST /upload    → ingest a PDF or plain-text file (or pasted text)
  POST /query     → RAG query, returns answer + structured citations
"""

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_engine import ingest_pdf, ingest_text, get_answer

app = FastAPI(title="Mini RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve static assets if the directory exists
_static = Path("static")
if _static.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse("index.html")


@app.get("/health")
async def health() -> dict:
    """Liveness probe — useful for Render / Railway keep-alive pings."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/upload")
async def handle_upload(
    text: Optional[str]       = Form(default=None),
    file: List[UploadFile]    = File(default=None),
) -> dict:
    """
    Ingest one or more files (PDF or plain-text) and/or pasted text.

    PDFs are processed page-by-page with OCR fallback for scanned pages.
    Plain-text files are split and embedded directly.
    Re-uploading the same file is idempotent (chunk IDs are deterministic).
    """
    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide a file or text to ingest.")

    total_chunks = 0
    last_source  = "manual_input"

    if file:
        for f in file:
            if not f.filename:
                continue
            dest = UPLOAD_DIR / Path(f.filename).name
            with open(dest, "wb") as buf:
                shutil.copyfileobj(f.file, buf)

            source_name = Path(f.filename).stem
            ext         = Path(f.filename).suffix.lower()

            if ext == ".pdf":
                total_chunks += ingest_pdf(str(dest), source_name)
            else:
                # Graceful fallback for .txt, .md, etc.
                with open(dest, "r", errors="replace") as fh:
                    content = fh.read()
                total_chunks += ingest_text(content, source_name)

            last_source = source_name

    if text and text.strip():
        total_chunks += ingest_text(text.strip(), "manual_input")
        last_source = "manual_input"

    return {"status": "success", "chunks": total_chunks, "source": last_source}


@app.post("/query")
async def handle_query(question: str = Form(...)) -> dict:
    """
    Full RAG pipeline: hybrid retrieval → rerank → grounded generation → cite.

    Response fields (backward-compatible with existing frontend):
      answer   — LLM-generated answer grounded in retrieved context
      context  — human-readable citation context string for display
      citations — structured list of {book_name, page_number, chunk_id, preview, score}
      meta     — latency, token counts, model name, chunks retrieved
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_answer(question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Query processing failed: {str(exc)}",
        ) from exc

    # Build a readable context string for the frontend source card
    context_parts = []
    for c in result["citations"]:
        context_parts.append(
            f"[{c['book_name']} — Page {c['page_number']}]\n{c['preview']}"
        )

    meta = result["metadata"]

    return {
        "answer":    result["answer"],
        # Readable context block — shown in the frontend source card
        "context":   "\n\n---\n\n".join(context_parts),
        # Structured citations — available for programmatic use / debugging
        "citations": result["citations"],
        # Metadata matching frontend's data.meta shape
        "meta": {
            "time":             f"{meta['latency']}s",
            "tokens":           (meta.get("input_tokens") or 0) + (meta.get("output_tokens") or 0),
            "cost":             "local embeddings + reranker",
            "model":            meta.get("model", ""),
            "chunks_retrieved": meta.get("chunks_retrieved", 0),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
