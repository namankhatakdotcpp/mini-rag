# RAG Retrieve-Then-Cite System

A production-oriented retrieval-augmented generation system built for a RAG Engineer internship assignment. Ingests textbook PDFs, retrieves relevant passages using **hybrid dense + BM25 search**, reranks with a **cross-encoder**, generates grounded answers with page-level citations, and evaluates retrieval quality with **measured Hit@k metrics**.

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
     │  Grounded LLM Generation     │  Gemini / OpenAI / Ollama (configurable)
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

## Design Decisions

### Embeddings — BAAI/bge-small-en-v1.5

The original system used Gemini's cloud embedding API, which introduces network latency, rate limits, and cost on every ingest and query. `bge-small-en-v1.5` runs fully locally using MPS/CUDA/CPU, ranks in the top tier of the MTEB retrieval benchmark for its size class, and makes the system reproducible without credentials. Embedding dimension is 384; inference takes ~2ms per chunk on Apple Silicon.

### Vector Store — ChromaDB

Replaced Pinecone (cloud, requires account, cold-start latency) with ChromaDB (local, persistent on disk, zero cost, no API key). For a two-textbook corpus, a local store with cosine HNSW is orders of magnitude simpler to reproduce and demo.

### Hybrid Retrieval — Dense + BM25

Dense retrieval finds semantically similar chunks but misses exact keyword matches for acronyms, proper nouns, and technical terms. BM25 fills this gap precisely. The two sources are merged by chunk_id and deduplicated before reranking. In the ablation study, BM25 did not improve Hit@1 on this specific dataset (BGE already achieved perfect recall within top-5), but it matters at scale with larger corpora where dense recall degrades.

### Reranker — cross-encoder/ms-marco-MiniLM-L-6-v2

Bi-encoder embeddings (used for retrieval speed) compress each text independently, losing cross-text interaction signals. A cross-encoder sees the query and chunk together, producing substantially more accurate relevance scores. The retrieve-then-rerank pattern is standard production RAG design: fast bi-encoders for recall, accurate cross-encoders for precision. On this dataset the reranker delivers **+21.4 percentage points on Hit@1**.

### Chunk Size — 500 chars, 100 overlap

Smaller chunks improve citation precision. A 500-character chunk fits comfortably on a single book page, making `page_number` citations exact. Larger chunks (1000+) risk spanning page boundaries and producing ambiguous citations. Overlap of 100 characters prevents content at chunk boundaries from losing context for embedding.

---

## Setup

### Prerequisites
```bash
# System dependency for OCR fallback (scanned pages)
brew install tesseract        # macOS
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
# Set GEMINI_API_KEY (or OPENAI_API_KEY if using OpenAI)
```

### Run
```bash
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
# Upload PDFs via the web UI, then ask questions
```

### Ingest PDFs directly
```python
from rag_engine import ingest_pdf
ingest_pdf("data/pdfs/textbook1.pdf", "Textbook 1")
ingest_pdf("data/pdfs/textbook2.pdf", "Textbook 2")
```

---

## Evaluation

Evaluated on 14 questions drawn from both textbooks. Each question was manually validated: the anchor phrase was confirmed to exist on the expected page, and confirmed to be absent on all other pages (or present only in the book index). See `data/questions/questions.json`.

### Ablation Results

| Configuration | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| Dense Only | 71.4% | 100.0% | 100.0% |
| Dense + BM25 | 71.4% | 100.0% | 100.0% |
| **Dense + BM25 + Reranker** | **92.9%** | **100.0%** | **100.0%** |

**Key finding:** The cross-encoder reranker delivers +21.4 percentage points on Hit@1 over dense-only retrieval. BM25 shows no marginal gain on this 14-question set because BGE already achieves perfect recall (Hit@5 = 100%) without it — the correct page always enters the candidate pool. BM25 would show gains on larger corpora where dense recall degrades.

### Run the evaluation
```bash
# Full pipeline (Dense + BM25 + Reranker)
python evaluate.py

# Ablation study (all three configurations)
python evaluate.py --ablation
```

---

## Failure Analysis

**1 question hits rank 2 in the full pipeline. 3 additional questions miss rank 1 in dense-only retrieval but are corrected by the reranker. Zero complete misses at Hit@5 across all configurations.**

Three recurring patterns were identified across all four ranking failures:

### Pattern A — Cross-book interference
*Affected: Q2 (Cardiac contraindications)*

A chapter in Brukner & Khan on cardiovascular exercise testing contraindications outranks the Cooper PPE chapter on cardiac sports participation contraindications. The bi-encoder cannot distinguish between "contraindications to a stress test" and "contraindications to sports participation in a preparticipation evaluation" — both embed into overlapping semantic space. The reranker reads both chunks jointly with the query and correctly identifies the Cooper list as the direct answer.

### Pattern B — Adjacent-page clustering
*Affected: Q9 (WADA prohibited substances)*

The page immediately before the WADA list (Cooper p193, a history of substance scheduling) scores nearly as high as p194 (the actual list) because both pages are dense with banned substance vocabulary. The correct page ranks 3rd in dense retrieval. The reranker identifies the explicit section header `"Substances Prohibited at All Times"` on p194 and ranks it first with a score of 9.06 vs 5.18 for the competing Brukner page. This failure pattern would be partially addressed by tighter chunking at section boundaries.

### Pattern C — Same-topic competing chapter
*Affected: Q10 (Myositis ossificans) — only case not fixed by reranker*

The same clinical condition appears in two chapters of Brukner & Khan: Chapter 2 (p51, definition + X-ray timing) and Chapter 25 (p472, detailed treatment). The dense model prefers p472 (score 0.8610 vs 0.8319) because it contains richer mechanistic content. The reranker also prefers p472 (3.782 vs 3.622) because the query asks three things — definition, mechanism, and X-ray timing — and p472 answers the first two with more detail. The specific "10 to 14 days" X-ray detail is on p51, but the reranker cannot determine that it is the *discriminating* answer without additional context. This case would be resolved by sentence-window retrieval (store small sub-chunks, expand to page context for the LLM) or by rephrasing the question to anchor on the unique timing detail.

---

## Project Structure

```
├── rag_engine.py        — Core pipeline: PDF → chunk → embed → retrieve → rerank → cite
├── main.py              — FastAPI server (POST /upload, POST /query, GET /health)
├── evaluate.py          — Hit@k evaluation CLI with ablation study support
├── requirements.txt
├── .env.example
├── index.html           — Frontend UI
│
├── data/
│   ├── pdfs/            — Place PDF textbooks here
│   └── questions/
│       └── questions.json   — 14 validated evaluation questions
│
├── outputs/             — query_logs.jsonl, eval_results.json (generated at runtime)
└── chroma_db/           — Persistent ChromaDB storage (generated at runtime)
```

---

## Future Improvements

| Improvement | Expected impact |
|---|---|
| Reciprocal Rank Fusion (RRF) | Better score merging for hybrid retrieval; would help Pattern B failures |
| Sentence-window retrieval | Retrieve at sentence level, expand to page for LLM context; fixes Pattern C |
| Larger embedding model (bge-base or bge-large) | Higher semantic discrimination; reduces Pattern A cross-book interference |
| OCR improvement (surya or doctr) | Replace Tesseract; handles complex layouts and tables in medical textbooks |
| RAGAS evaluation | Measure faithfulness and answer relevance in addition to retrieval accuracy |
| Query rewriting / HyDE | Improve recall for ambiguous or multi-part questions |
