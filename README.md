# Mini RAG — Retrieve → Rerank → Generate → Cite

A production-oriented retrieval-augmented generation system built for a RAG Engineer internship assignment. Ingests medical-textbook PDFs, retrieves relevant passages with **hybrid dense + BM25 search**, reranks with a **cross-encoder**, generates grounded answers via Gemini, and evaluates retrieval quality with **Hit@k metrics**.

**Evaluation results (14 questions, two textbooks):**

| Configuration | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| Dense Only | 71.4% | 100.0% | 100.0% |
| Dense + BM25 | 71.4% | 100.0% | 100.0% |
| **Dense + BM25 + Reranker** | **92.9%** | **100.0%** | **100.0%** |

---

## Quick Start

```bash
# 1. Install
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # add your GEMINI_API_KEY

# 3. Run
uvicorn main:app --reload --port 8000
# Open http://localhost:8000 — upload PDFs, ask questions
```

Optional: pre-ingest PDFs from the command line before starting the server:

```python
from rag_engine import ingest_pdf
ingest_pdf("data/pdfs/textbook.pdf", "Book Name")
```

---

## Architecture

```
PDF Textbooks
      │
      ▼
┌──────────────────────────────────────────┐
│  PDF Loader  (PyMuPDF)                   │  Per-page text extraction
│  OCR Fallback (Tesseract)                │  Auto-triggered when page text < 50 chars
│  Metadata: {book_name, page_number}      │  Attached to every page before chunking
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  Chunking  (RecursiveCharacterSplitter)  │  chunk_size=500, overlap=100
│  Deterministic chunk_id  (MD5)           │  Re-ingestion is idempotent
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  Embeddings  (BAAI/bge-small-en-v1.5)    │  Local, no API key, 384-dim
│  ChromaDB  (persistent, cosine)          │  Local vector store, no cloud account
│  BM25 index  (rank-bm25)                 │  Rebuilt in-memory after every ingest
└──────────┬────────────────┬──────────────┘
           │                │
    Dense Top-10       BM25 Top-10
           │                │
           └────────┬───────┘
                    │  Merge + Deduplicate by chunk_id
                    ▼
     ┌──────────────────────────────┐
     │  Cross-Encoder Reranker      │  ms-marco-MiniLM-L-6-v2, local
     │  Top-20 candidates → Top-5   │  Scores (query, chunk) pairs jointly
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Grounded LLM Generation     │  Gemini 2.5 Flash (configurable)
     │  System prompt enforces:     │  Answer ONLY from context
     │    - No hallucination        │  Cite with [N] notation
     │    - Explicit fallback phrase│
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Structured Citations        │  {book_name, page_number, chunk_id, preview}
     │  Query Logger  (JSONL)       │  outputs/query_logs.jsonl per query
     └──────────────────────────────┘
```

---

## Setup

### Prerequisites

```bash
brew install tesseract          # macOS — required for OCR fallback on scanned pages
sudo apt install tesseract-ocr  # Ubuntu
```

### Install

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Python 3.11 or 3.12 recommended. Python 3.14 works but produces a harmless startup warning from langchain's pydantic-v1 compatibility layer.

### Configure

```bash
cp .env.example .env
```

Set at minimum:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (if using Gemini) | — | Gemini API key |
| `LLM_PROVIDER` | No | `gemini` | `gemini` / `openai` / `local` |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | If `LLM_PROVIDER=openai` | — | OpenAI API key |
| `CHUNK_SIZE` | No | `500` | Characters per chunk |
| `DENSE_TOP_K` | No | `10` | Dense retrieval candidate count |
| `BM25_TOP_K` | No | `10` | BM25 retrieval candidate count |
| `RERANK_TOP_N` | No | `5` | Final chunks passed to the LLM |

---

## API

The server exposes three endpoints. Interactive docs at `http://localhost:8000/docs`.

### `GET /health`
Liveness probe. Returns `{"status": "ok", "version": "2.0.0"}`.

### `POST /upload`
Ingest one or more PDFs or plain-text files, and/or pasted text.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `file` | File (repeatable) | PDF or `.txt`/`.md` file |
| `text` | string | Pasted plain text |

**Response:**
```json
{ "status": "success", "chunks": 42, "source": "textbook_name" }
```

Re-uploading the same file is idempotent — chunk IDs are deterministic MD5 hashes.

### `POST /query`
Full RAG pipeline: retrieve → rerank → generate → cite.

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `question` | string | The question to answer |

**Response:**
```json
{
  "answer": "Second impact syndrome occurs when...",
  "context": "[Book — Page 31]\n...",
  "citations": [
    {
      "book_name": "ESSNTIAL SPORTS MEDICINE - GRANT COOPER",
      "page_number": 31,
      "chunk_id": "235d1a7a7576",
      "preview": "Second Impact Syndrome...",
      "rerank_score": 6.99
    }
  ],
  "meta": {
    "time": "4.2s",
    "tokens": 1840,
    "model": "gemini-2.5-flash",
    "chunks_retrieved": 5
  }
}
```

---

## Evaluation

### Run

```bash
python evaluate.py                  # full pipeline — saves to outputs/eval_results.json
python evaluate.py --ablation       # all three configurations — saves ablation_results.json
python evaluate.py --questions path/to/questions.json   # custom question set
```

### Questions format

```json
[
  {
    "question": "What is second impact syndrome?",
    "expected_page": 31,
    "book": "GRANT COOPER"
  }
]
```

`book` is matched as a case-insensitive substring against the stored `book_name`, so `"GRANT COOPER"` correctly matches `"ESSNTIAL SPORTS MEDICINE - GRANT COOPER"`.

### Key findings

The cross-encoder reranker delivers **+21.4 pp on Hit@1** over dense-only retrieval. BM25 shows no marginal gain on this 14-question dataset because BGE already achieves perfect recall (Hit@5 = 100%) — the correct page always enters the candidate pool. BM25 would show gains on larger corpora where dense recall degrades.

---

## Failure Analysis

**1 question ranks at Hit@2. 3 additional questions miss Hit@1 in dense-only but are corrected by the reranker. Zero complete misses at Hit@5 across all configurations.**

### Pattern A — Cross-book interference
*Q: Cardiac contraindications (dense-only miss, fixed by reranker)*

A Brukner & Khan chapter on cardiovascular stress-test contraindications outranks the Cooper PPE chapter in the bi-encoder pass because both embed into overlapping semantic space. The cross-encoder reads each chunk with the query jointly and correctly identifies the Cooper passage as the direct answer to a sports-participation evaluation question.

### Pattern B — Adjacent-page clustering
*Q: WADA prohibited substances (dense-only miss, fixed by reranker)*

The page before the WADA list (Cooper p193, history of substance scheduling) scores nearly as high as p194 (the list itself) because both pages are dense with banned-substance vocabulary. The reranker identifies the explicit section header on p194 and ranks it first with score 9.06 vs 5.18 for the competing Brukner page. Tighter chunking at section boundaries would partially address this.

### Pattern C — Same-topic competing chapter
*Q: Myositis ossificans — only miss not corrected by reranker (Hit@2)*

The same condition appears in two Brukner & Khan chapters: p51 (definition + X-ray timing) and p472 (detailed treatment). The reranker prefers p472 (3.782 vs 3.622) because the query asks three things and p472 answers two of them with more detail. The "10 to 14 days" X-ray timing detail on p51 is the discriminating answer, but the cross-encoder cannot identify it as the *intended* answer without additional context. Sentence-window retrieval or rephrasing the question to anchor on the unique timing detail would resolve this.

---

## Design Decisions

### Embeddings — BAAI/bge-small-en-v1.5
The original system used Gemini's cloud embedding API, which adds network latency, rate limits, and cost on every ingest and query. `bge-small-en-v1.5` runs fully locally (MPS/CUDA/CPU), ranks in the top tier of the MTEB retrieval benchmark for its size class, and makes the system reproducible without credentials. Embedding dimension is 384; ~2 ms per chunk on Apple Silicon.

### Vector Store — ChromaDB
Replaced Pinecone (cloud, requires account, cold-start latency) with ChromaDB (local, persistent on disk, zero cost, no API key). For a two-textbook corpus, a local cosine-HNSW store is orders of magnitude simpler to reproduce and demo.

### Hybrid Retrieval — Dense + BM25
Dense retrieval finds semantically similar chunks but misses exact keyword matches for acronyms, proper nouns, and technical terms. BM25 fills this gap precisely. The two candidate sets are merged by `chunk_id` and deduplicated before reranking.

### Reranker — cross-encoder/ms-marco-MiniLM-L-6-v2
Bi-encoder embeddings compress each text independently, losing cross-text interaction signals. A cross-encoder sees the query and chunk together, producing substantially more accurate relevance scores. The retrieve-then-rerank pattern is standard production RAG design: fast bi-encoders for recall, accurate cross-encoders for precision.

### Chunk Size — 500 chars, 100 overlap
Smaller chunks improve citation precision. A 500-character chunk fits comfortably within a single book page, making `page_number` citations exact. Larger chunks (1000+) risk spanning page boundaries and producing ambiguous citations. The 100-character overlap prevents boundary content from losing embedding context.

---

## Project Structure

```
├── rag_engine.py        — Core pipeline: PDF → chunk → embed → retrieve → rerank → cite
├── main.py              — FastAPI server (POST /upload, POST /query, GET /health)
├── evaluate.py          — Hit@k evaluation CLI with ablation study support
├── index.html           — Frontend UI (served by FastAPI at /)
├── requirements.txt
├── .env.example         — Copy to .env and fill in API keys
│
├── data/
│   ├── pdfs/            — Place PDF textbooks here for ingestion
│   └── questions/
│       └── questions.json   — 14 validated evaluation questions
│
├── outputs/             — query_logs.jsonl, eval_results.json (auto-generated)
└── chroma_db/           — Persistent ChromaDB vector store (auto-generated)
```

> `chroma_db/` and `outputs/` are generated at runtime and excluded from version control.

---

## Future Improvements

| Improvement | Expected impact |
|---|---|
| Reciprocal Rank Fusion (RRF) | Better score merging for hybrid retrieval; reduces Pattern B failures |
| Sentence-window retrieval | Store at sentence level, expand to page for LLM context; resolves Pattern C |
| Larger embedding model (bge-base or bge-large) | Higher semantic discrimination; reduces Pattern A cross-book interference |
| OCR improvement (surya or doctr) | Better handling of complex layouts and tables in medical textbooks |
| RAGAS evaluation | Measure answer faithfulness and relevance in addition to retrieval accuracy |
| Query rewriting / HyDE | Improve recall for ambiguous or multi-part questions |
