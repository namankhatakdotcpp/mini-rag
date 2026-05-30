"""
RAG Engine — production-grade Mini-RAG.

Pipeline:
  PDF Loader → OCR Fallback → Metadata Chunking → BGE Embeddings
  → ChromaDB + BM25 → Hybrid Retrieval → Cross-Encoder Rerank
  → Grounded Generation → Structured Citations → Query Logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import chromadb

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (all values overridable via .env)
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE: int      = int(os.getenv("CHUNK_SIZE",    500))
CHUNK_OVERLAP: int   = int(os.getenv("CHUNK_OVERLAP", 100))
DENSE_TOP_K: int     = int(os.getenv("DENSE_TOP_K",   10))
BM25_TOP_K: int      = int(os.getenv("BM25_TOP_K",    10))
RERANK_TOP_N: int    = int(os.getenv("RERANK_TOP_N",  5))
OCR_MIN_CHARS: int   = int(os.getenv("OCR_MIN_CHARS", 50))

EMBED_MODEL: str      = os.getenv("EMBED_MODEL",    "BAAI/bge-small-en-v1.5")
RERANK_MODEL: str     = os.getenv("RERANK_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_PROVIDER: str     = os.getenv("LLM_PROVIDER",   "gemini").lower()
DISABLE_RERANKER: bool = os.getenv("DISABLE_RERANKER", "").lower() in ("1", "true", "yes")

CHROMA_DIR: str      = os.getenv("CHROMA_DIR",     "./chroma_db")
LOGS_DIR: str        = os.getenv("LOGS_DIR",        "./outputs")
COLLECTION_NAME: str = "rag_chunks"

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
QUERY_LOG_PATH = Path(LOGS_DIR) / "query_logs.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: OCR dependencies
# ──────────────────────────────────────────────────────────────────────────────

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract/Pillow not installed — OCR fallback disabled.")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL INITIALIZATION  (lazy — loaded on first use so port binds before download)
# ──────────────────────────────────────────────────────────────────────────────

_model_lock: threading.Lock = threading.Lock()
_embed_model: Optional[SentenceTransformer] = None
_cross_encoder_model: Optional[CrossEncoder] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        with _model_lock:
            if _embed_model is None:
                logger.info(f"Loading embedding model: {EMBED_MODEL}")
                _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        with _model_lock:
            if _cross_encoder_model is None:
                logger.info(f"Loading cross-encoder: {RERANK_MODEL}")
                _cross_encoder_model = CrossEncoder(RERANK_MODEL)
    return _cross_encoder_model

# ──────────────────────────────────────────────────────────────────────────────
# CHROMADB
# ──────────────────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# ──────────────────────────────────────────────────────────────────────────────
# BM25 INDEX  (rebuilt from ChromaDB after every ingest)
# ──────────────────────────────────────────────────────────────────────────────

_bm25: Optional[BM25Okapi] = None
_bm25_chunks: list[dict[str, Any]] = []


def _rebuild_bm25() -> None:
    """Sync BM25 index with the current ChromaDB collection."""
    global _bm25, _bm25_chunks
    result = collection.get(include=["documents", "metadatas"])
    docs  = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not docs:
        _bm25, _bm25_chunks = None, []
        return
    _bm25_chunks = [{"text": d, **m} for d, m in zip(docs, metas)]
    tokenized    = [c["text"].lower().split() for c in _bm25_chunks]
    _bm25        = BM25Okapi(tokenized)
    logger.info(f"BM25 index rebuilt: {len(_bm25_chunks)} chunks")


_rebuild_bm25()  # warm up on import

# ──────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_page(page: fitz.Page, page_num: int) -> str:
    """Render a PDF page to a 200 DPI image and run Tesseract OCR on it.

    Returns empty string on any failure — OCR is a fallback and must never
    crash the ingestion pipeline. Common failure: binary stderr from Tesseract
    on corrupt/pure-image pages causes UnicodeDecodeError in pytesseract.
    """
    if not OCR_AVAILABLE:
        return ""
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except Exception as exc:
        logger.warning(f"OCR failed p{page_num} — skipping: {type(exc).__name__}: {exc}")
        return ""


def extract_pdf(
    pdf_path: str,
    book_name: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Extract per-page text from a PDF file.

    Pages with fewer than OCR_MIN_CHARS characters automatically fall back to
    Tesseract OCR. Every page record carries book_name and page_number so
    downstream citations are exact.

    Returns
    -------
    pages : list of {book_name, page_number, text}
    stats : {total, extracted, ocr_attempted, ocr_succeeded, ocr_failed, skipped}
    """
    book_name = book_name or Path(pdf_path).stem
    pages: list[dict[str, Any]] = []

    ocr_attempted = 0
    ocr_succeeded = 0
    ocr_failed    = 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{pdf_path}': {exc}") from exc

    with doc:
        total = len(doc)
        for i in range(total):
            page_num = i + 1
            page     = doc[i]
            text     = page.get_text("text").strip()

            if len(text) < OCR_MIN_CHARS:
                ocr_attempted += 1
                recovered = _ocr_page(page, page_num).strip()
                if recovered:
                    ocr_succeeded += 1
                    logger.info(
                        f"OCR p{page_num}: '{book_name}' "
                        f"({len(text)} → {len(recovered)} chars)"
                    )
                    text = recovered
                else:
                    ocr_failed += 1

            if text:
                pages.append({
                    "book_name":   book_name,
                    "page_number": page_num,
                    "text":        text,
                })

    stats: dict[str, int] = {
        "total":         total,
        "extracted":     len(pages),
        "ocr_attempted": ocr_attempted,
        "ocr_succeeded": ocr_succeeded,
        "ocr_failed":    ocr_failed,
        "skipped":       total - len(pages),
    }
    return pages, stats


# ──────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split pages into overlapping chunks while propagating book_name and page_number.

    chunk_id is a deterministic MD5 of (book_name + page_number + position index)
    so re-ingesting the same document is idempotent (ChromaDB upsert deduplicates).
    """
    chunks: list[dict[str, Any]] = []
    for page in pages:
        splits = _splitter.split_text(page["text"])
        for idx, split in enumerate(splits):
            cid = hashlib.md5(
                f"{page['book_name']}_{page['page_number']}_{idx}".encode()
            ).hexdigest()[:12]
            chunks.append({
                "chunk_id":    cid,
                "book_name":   page["book_name"],
                "page_number": page["page_number"],
                "text":        split.strip(),
            })
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# INGESTION
# ──────────────────────────────────────────────────────────────────────────────

_CHROMA_BATCH = 5000  # ChromaDB hard limit is 5461; stay safely below it


def get_document_id(source_name: str) -> str:
    """Deterministic, stable document ID derived from the source name.

    Using MD5 of the source name keeps re-ingestion idempotent: the same PDF
    always maps to the same document_id, so ChromaDB upserts update rather than
    duplicate existing chunks.
    """
    return hashlib.md5(source_name.encode()).hexdigest()[:16]


def _upsert_chunks(chunks: list[dict[str, Any]], document_id: str) -> None:
    """Embed and upsert chunks into ChromaDB in batches, then rebuild BM25.

    ChromaDB enforces a maximum batch size of 5461. Large books (1000+ pages)
    exceed this in a single call, so we split into _CHROMA_BATCH-sized batches.
    """
    texts = [c["text"] for c in chunks]
    embeddings = _get_embed_model().encode(
        texts,
        batch_size=64,
        show_progress_bar=len(chunks) > 50,
        normalize_embeddings=True,
    ).tolist()

    for start in range(0, len(chunks), _CHROMA_BATCH):
        batch   = chunks[start : start + _CHROMA_BATCH]
        batch_e = embeddings[start : start + _CHROMA_BATCH]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=batch_e,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "book_name":   c["book_name"],
                    "page_number": c["page_number"],
                    "chunk_id":    c["chunk_id"],
                    "document_id": document_id,
                }
                for c in batch
            ],
        )

    _rebuild_bm25()


def ingest_pdf(pdf_path: str, book_name: Optional[str] = None) -> int:
    """
    Full PDF ingestion: extract → chunk → embed → store in ChromaDB + BM25.

    Prints a summary after every ingest and verifies the collection is non-empty.
    Returns the number of chunks stored for this document.
    """
    name = book_name or Path(pdf_path).stem
    pages, stats = extract_pdf(pdf_path, name)

    if not pages:
        raise ValueError(
            f"No text extracted from '{pdf_path}'. "
            f"OCR attempted {stats['ocr_attempted']} pages, "
            f"all failed or empty."
        )

    chunks = chunk_pages(pages)
    doc_id = get_document_id(name)
    _upsert_chunks(chunks, doc_id)

    col_count = collection.count()

    logger.info(
        f"\n"
        f"  ┌─ Ingestion: '{name}'\n"
        f"  │  Pages in PDF      : {stats['total']}\n"
        f"  │  Pages extracted   : {stats['extracted']}\n"
        f"  │  Pages via OCR     : {stats['ocr_succeeded']}\n"
        f"  │  Pages skipped     : {stats['skipped']}"
        + (f" ({stats['ocr_failed']} OCR failures)" if stats['ocr_failed'] else "") + "\n"
        f"  │  Chunks created    : {len(chunks)}\n"
        f"  │  Collection total  : {col_count}\n"
        f"  └─ {'OK' if col_count > 0 else 'WARNING: collection still empty'}"
    )

    return len(chunks)


def ingest_text(text: str, source_name: str) -> int:
    """Ingest raw text (no PDF). Preserves backward compatibility with text uploads."""
    pages  = [{"book_name": source_name, "page_number": 1, "text": text}]
    chunks = chunk_pages(pages)
    _upsert_chunks(chunks, get_document_id(source_name))
    logger.info(f"Ingested {len(chunks)} chunks from text source '{source_name}'")
    return len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def _dense_search(
    query:       str,
    top_k:       int,
    document_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """ChromaDB cosine-similarity search. Optionally scoped to one document."""
    n = collection.count()
    if n == 0:
        return []
    q_emb  = _get_embed_model().encode(query, normalize_embeddings=True).tolist()
    kwargs: dict[str, Any] = {
        "query_embeddings": [q_emb],
        "n_results":        min(top_k, n),
        "include":          ["documents", "metadatas", "distances"],
    }
    if document_id:
        kwargs["where"] = {"document_id": document_id}
    try:
        result = collection.query(**kwargs)
    except Exception as exc:
        logger.warning(f"Dense search error (document_id={document_id}): {exc}")
        return []
    chunks = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "book_name":   meta["book_name"],
            "page_number": meta["page_number"],
            "chunk_id":    meta["chunk_id"],
            "document_id": meta.get("document_id", ""),
            "dense_score": round(float(1.0 - dist), 4),
        })
    return chunks


def _bm25_search(
    query:       str,
    top_k:       int,
    document_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """BM25 keyword search. Optionally scoped to one document."""
    if _bm25 is None or not _bm25_chunks:
        return []
    scores  = _bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        chunk = _bm25_chunks[i]
        if document_id and chunk.get("document_id") != document_id:
            continue
        results.append({**chunk, "bm25_score": round(float(scores[i]), 4)})
        if len(results) >= top_k:
            break
    return results


def hybrid_retrieve(
    query:       str,
    dense_k:     int = DENSE_TOP_K,
    bm25_k:      int = BM25_TOP_K,
    document_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Merge dense and BM25 results, deduplicate by chunk_id.

    Pass document_id to restrict retrieval to a single uploaded document.
    Omit it (or pass None) to search across the entire collection.
    """
    dense  = _dense_search(query, dense_k, document_id)
    sparse = _bm25_search(query, bm25_k, document_id)

    merged: dict[str, dict[str, Any]] = {}
    for c in dense:
        merged[c["chunk_id"]] = c
    for c in sparse:
        if c["chunk_id"] in merged:
            merged[c["chunk_id"]]["bm25_score"] = c.get("bm25_score", 0.0)
        else:
            merged[c["chunk_id"]] = c

    return list(merged.values())


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-ENCODER RERANKING
# ──────────────────────────────────────────────────────────────────────────────

def rerank_chunks(
    query:  str,
    chunks: list[dict[str, Any]],
    top_n:  int = RERANK_TOP_N,
) -> list[dict[str, Any]]:
    """
    Score every (query, chunk) pair with the cross-encoder and return top_n.

    Cross-encoders are more accurate than bi-encoders for relevance because
    they see both texts jointly — but are too slow to run on the full index,
    hence the two-stage retrieve-then-rerank design.
    """
    if not chunks:
        return []
    if DISABLE_RERANKER:
        for c in chunks:
            c["rerank_score"] = c.get("dense_score", 0.0)
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    pairs  = [[query, c["text"]] for c in chunks]
    scores = _get_cross_encoder().predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = round(float(s), 4)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# CITATION FORMATTER
# ──────────────────────────────────────────────────────────────────────────────

def format_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build structured citations from reranked chunks.

    Deduplicates by (book_name, page_number) — multiple chunks from the same
    page produce a single citation entry, preserving the highest-scored preview.
    """
    seen:      set[tuple[str, int]]     = set()
    citations: list[dict[str, Any]]     = []
    for c in chunks:
        key = (c["book_name"], c["page_number"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "book_name":    c["book_name"],
                "page_number":  c["page_number"],
                "chunk_id":     c.get("chunk_id", ""),
                "preview":      c["text"][:200],
                "rerank_score": c.get("rerank_score", 0.0),
            })
    return citations


# ──────────────────────────────────────────────────────────────────────────────
# LLM PROVIDERS  (configurable via LLM_PROVIDER env var)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise research assistant.\n"
    "You must answer ONLY from the provided context. "
    "Never use prior knowledge or invent information.\n"
    "If the answer is not in the context, respond exactly: "
    "'I could not find the answer in the supplied documents.'\n"
    "Always cite sources using the [N] notation shown in the context."
)


def _context_block(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['book_name']} — Page {c['page_number']}:\n{c['text']}")
    return "\n\n".join(parts)


def _generate_gemini(user_prompt: str) -> tuple[str, dict[str, Any]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )
    usage = response.usage_metadata
    return response.text, {
        "model":         model,
        "input_tokens":  getattr(usage, "prompt_token_count",     0),
        "output_tokens": getattr(usage, "candidates_token_count", 0),
    }


def _generate_openai(user_prompt: str) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp   = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content, {
        "model":         model,
        "input_tokens":  resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def _generate_local(user_prompt: str) -> tuple[str, dict[str, Any]]:
    """Ollama-compatible local model endpoint."""
    import requests

    model = os.getenv("LOCAL_MODEL",   "llama3")
    url   = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
    resp  = requests.post(
        url,
        json={"model": model, "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}", "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"], {"model": model}


_LLM_REGISTRY = {
    "gemini": _generate_gemini,
    "openai": _generate_openai,
    "local":  _generate_local,
}


def generate_answer(
    query:  str,
    chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    Build a grounded prompt from reranked chunks and call the configured LLM.
    Retries up to 3 times on 429 rate-limit errors with exponential back-off.
    """
    context     = _context_block(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using ONLY the context above. "
        "Use [N] to cite specific sources. "
        "If the answer is not present, respond: "
        "'I could not find the answer in the supplied documents.'"
    )
    gen_fn = _LLM_REGISTRY.get(LLM_PROVIDER, _generate_gemini)

    for attempt in range(3):
        try:
            return gen_fn(user_prompt)
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                wait = (attempt + 1) * 5
                logger.warning(f"Rate limited. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


# ──────────────────────────────────────────────────────────────────────────────
# QUERY LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def _log_query(
    query:     str,
    chunks:    list[dict[str, Any]],
    answer:    str,
    citations: list[dict[str, Any]],
    latency:   float,
    llm_meta:  dict[str, Any],
) -> None:
    """Append a full query trace to outputs/query_logs.jsonl (one JSON object per line)."""
    entry = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "query":           query,
        "latency_seconds": round(latency, 3),
        "answer_preview":  answer[:300],
        "citations":       citations,
        "llm_meta":        llm_meta,
        "chunks": [
            {
                "chunk_id":    c.get("chunk_id"),
                "book_name":   c.get("book_name"),
                "page_number": c.get("page_number"),
                "rerank_score": c.get("rerank_score"),
                "dense_score":  c.get("dense_score"),
                "bm25_score":   c.get("bm25_score"),
                "preview":      c["text"][:120],
            }
            for c in chunks
        ],
    }
    with open(QUERY_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def get_answer(query: str, document_id: Optional[str] = None) -> dict[str, Any]:
    """
    Full RAG pipeline:
      Query → Hybrid Retrieve → Cross-Encoder Rerank → Grounded LLM → Cite → Log

    Pass document_id to restrict retrieval to a single uploaded document.
    """
    start = time.time()

    candidates = hybrid_retrieve(query, document_id=document_id)
    if not candidates:
        return {
            "answer":       "I could not find the answer in the supplied documents.",
            "citations":    [],
            "sources_text": "",
            "metadata":     {"latency": 0.0, "model": LLM_PROVIDER, "chunks_retrieved": 0},
        }

    reranked     = rerank_chunks(query, candidates)
    answer_text, llm_meta = generate_answer(query, reranked)
    citations    = format_citations(reranked)
    latency      = time.time() - start

    _log_query(query, reranked, answer_text, citations, latency, llm_meta)

    return {
        "answer":    answer_text,
        "citations": citations,
        "sources_text": " | ".join(
            f"{c['book_name']} p.{c['page_number']}" for c in citations
        ),
        "metadata": {
            "latency":         round(latency, 3),
            "model":           llm_meta.get("model", LLM_PROVIDER),
            "chunks_retrieved": len(reranked),
            "input_tokens":    llm_meta.get("input_tokens"),
            "output_tokens":   llm_meta.get("output_tokens"),
        },
    }
