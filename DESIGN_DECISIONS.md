# Design Decisions Report

## What I Built

A retrieve-then-cite system over two sports medicine textbooks. Given a natural language question, the system retrieves the most relevant passages, reranks them with a cross-encoder, generates a grounded answer using only the retrieved context, and returns structured citations (book name + page number) alongside the answer.

The pipeline:

```
PDF → OCR Fallback → Metadata Chunking → BGE Embeddings
    → ChromaDB + BM25 → Hybrid Retrieval → Cross-Encoder Rerank
    → Grounded LLM → Structured Citations
```

**Corpus:** Clinical Sports Medicine (Brukner & Khan, 1069 pages, 8,708 chunks) and Essential Sports Medicine (Cooper & Herrera, 203 pages, 1,169 chunks). Total: 9,877 chunks stored in ChromaDB.

---

## Why I Chose These Components

**BAAI/bge-small-en-v1.5 over cloud embeddings**

The original system used Gemini's embedding API. I replaced it with a local model for three reasons: (1) no API quota or rate limiting during demo, (2) fully reproducible without credentials, (3) comparable retrieval quality. BGE consistently ranks in the top tier of the MTEB benchmark for its size. Local inference on Apple Silicon (MPS) takes ~2ms per chunk.

**ChromaDB over Pinecone**

Pinecone requires an account, has a cold-start period on the free tier, and fails without internet access. ChromaDB stores everything on disk locally. For a two-book corpus, a local HNSW index is more than sufficient and eliminates a cloud dependency that would prevent a reviewer from running the system offline.

**Hybrid retrieval (dense + BM25)**

Dense retrieval misses exact keyword matches — acronyms like TUBS or AMBRI, drug names like erythropoietin, and proper nouns. BM25 fills this gap. I built the BM25 index in-memory from ChromaDB contents and rebuild it after every ingest. The two results are merged by chunk ID before reranking.

**Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)**

Bi-encoder embeddings cannot see the query and document together, so they miss fine-grained relevance signals. The cross-encoder reads each (query, chunk) pair jointly and re-scores them. On the 14-question evaluation set, this delivered +21.4 percentage points on Hit@1 (from 71.4% to 92.9%). The two-stage design — fast bi-encoder for recall, accurate cross-encoder for precision — is the standard production RAG pattern.

**500-character chunks, 100 overlap**

Chunk size directly controls citation precision. A 500-character chunk almost always fits within one book page, keeping `page_number` citations exact. Larger chunks (1000+) span page boundaries and produce ambiguous citations ("page 42 or 43?"). The 100-character overlap prevents content at boundaries from being split mid-sentence and losing embedding context.

**JSONL for query logs**

The original system used a single JSON array file, reading and rewriting the entire file on every query. Under concurrent requests, two writes can race and one overwrites the other. JSONL uses append-mode writes — each query is one line — making logging atomic at typical log sizes.

---

## Evaluation Results

I wrote 14 evaluation questions from the two textbooks, validated each question to confirm the answer is unique to the expected page (anchor phrase confirmed absent from all other pages), then ran three retrieval configurations to measure the incremental contribution of each component.

| Configuration | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| Dense Only | 71.4% | 100.0% | 100.0% |
| Dense + BM25 | 71.4% | 100.0% | 100.0% |
| Dense + BM25 + Reranker | **92.9%** | **100.0%** | **100.0%** |

The perfect Hit@5 across all configurations confirms the retrieval pipeline reliably finds the correct page — the reranker's job is to rank it first.

---

## Failure Analysis

Four questions missed rank 1 in at least one configuration. Three patterns emerged:

**Cross-book interference.** Brukner & Khan's cardiovascular exercise testing chapter outranked Cooper's PPE cardiac contraindications list in dense retrieval. Both pages discuss "cardiac conditions + sports/exercise," embedding into the same semantic region. The reranker resolved this by reading the specific list against the specific question.

**Adjacent-page clustering.** The page immediately before the WADA prohibited substance list (Cooper p193, a historical summary) scored nearly as high as p194 (the actual list) because both pages share the same vocabulary. The reranker identified the list header and ranked p194 first.

**Same-topic competing chapter.** Myositis ossificans appears in two Brukner chapters: the definitions chapter (p51, X-ray timing: 10–14 days) and the treatment chapter (p472, detailed mechanistic description). Both the dense model and the reranker preferred p472's richer content. This is the one case the reranker did not fix — the correct page hit rank 2 in the full pipeline. The solution is sentence-window retrieval: store 100-character sub-chunks, expand to the full page for the LLM. The specific `"10 to 14 days"` sub-chunk would then anchor retrieval to p51.

---

## What Broke and How I Fixed It

**OCR crash during ingestion**

When PyMuPDF found a page with fewer than 50 characters of text, it triggered Tesseract OCR. On certain pages (pure figures or cover images), Tesseract produced binary data in its stderr. pytesseract attempted to decode this as UTF-8 and raised `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89`. This crashed the entire ingestion pipeline.

Fix: wrapped `_ocr_page` in a blanket `try/except`, returning an empty string on any Tesseract failure with a per-page warning. OCR is a fallback — a failure on one page must never abort ingestion of 1,000 others.

**ChromaDB batch size exceeded**

Brukner & Khan (1069 pages) produces 8,708 chunks. ChromaDB enforces a hard maximum batch size of 5,461 per `upsert` call. Attempting to insert 8,708 chunks in one call raised `InternalError: Batch size of 8708 is greater than max batch size of 5461`.

Fix: split the upsert loop into batches of 5,000. The embedding step still runs in one vectorized call; only the ChromaDB write is batched.

**Python 3.14 compatibility warning**

The project runs on Python 3.14. langchain-text-splitters internally imports pydantic v1 compatibility shims, which print a startup warning on Python 3.14: `UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater`. The splitter itself works correctly; the warning is cosmetic. Python 3.11 or 3.12 is recommended for clean output.
