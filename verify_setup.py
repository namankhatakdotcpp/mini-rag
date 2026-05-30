"""
Pre-flight check for the RAG system.

Verifies every component before running ingestion or evaluation.
Run this first if something isn't working as expected.

Usage:
    python verify_setup.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_chromadb() -> bool:
    print("Checking ChromaDB...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_or_create_collection("rag_chunks")
        count = col.count()
        print(f"  OK — ChromaDB {chromadb.__version__}, collection has {count} chunks")
        if count == 0:
            print("  NOTE: collection is empty — run ingestion before evaluation")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def check_embeddings() -> bool:
    print("Checking embedding model (BAAI/bge-small-en-v1.5)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        emb = model.encode("test sentence", normalize_embeddings=True)
        assert emb.shape == (384,), f"unexpected shape {emb.shape}"
        print(f"  OK — 384-dim embedding produced")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def check_reranker() -> bool:
    print("Checking cross-encoder (ms-marco-MiniLM-L-6-v2)...")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.predict([["what is fever?", "Fever is high body temperature."]])
        print(f"  OK — score: {float(scores[0]):.4f}")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def check_bm25() -> bool:
    print("Checking BM25 (rank-bm25)...")
    try:
        from rank_bm25 import BM25Okapi
        # Use enough documents so IDF is non-trivial
        corpus = [
            ["myositis", "ossificans", "x-ray", "days"],
            ["cardiac", "hypertrophic", "cardiomyopathy"],
            ["tendinitis", "inflammation", "tendon"],
            ["stress", "fracture", "bone", "scan"],
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["myositis", "ossificans"])
        assert float(scores[0]) > 0, "expected positive score for matching document"
        print(f"  OK — top score: {float(scores[0]):.4f}")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def check_pymupdf() -> bool:
    print("Checking PyMuPDF...")
    try:
        import fitz
        print(f"  OK — PyMuPDF {fitz.__version__}")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def check_tesseract() -> bool:
    print("Checking Tesseract OCR (optional)...")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"  OK — Tesseract {version}")
        return True
    except Exception as exc:
        print(f"  SKIP — pytesseract not available or Tesseract not installed: {exc}")
        print("         OCR fallback disabled; digital PDFs will still work")
        return True  # OCR is optional


def check_llm() -> bool:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    print(f"Checking LLM provider ({provider})...")
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            print("  FAIL — GEMINI_API_KEY not set in .env")
            return False
        try:
            from google import genai
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Say 'OK' and nothing else."
            )
            print(f"  OK — Gemini responded: {resp.text.strip()[:40]}")
            return True
        except Exception as exc:
            print(f"  FAIL: {exc}")
            return False
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            print("  FAIL — OPENAI_API_KEY not set in .env")
            return False
        try:
            import importlib
            OpenAI = importlib.import_module("openai").OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            print(f"  OK — OpenAI responded")
            return True
        except Exception as exc:
            print(f"  FAIL: {exc}")
            return False
    else:
        print(f"  SKIP — provider '{provider}' (local/custom), skipping API check")
        return True


def main() -> None:
    print()
    print("=" * 50)
    print("  RAG SYSTEM PRE-FLIGHT CHECK")
    print("=" * 50)
    print()

    results = {
        "ChromaDB":     check_chromadb(),
        "Embeddings":   check_embeddings(),
        "Reranker":     check_reranker(),
        "BM25":         check_bm25(),
        "PyMuPDF":      check_pymupdf(),
        "Tesseract":    check_tesseract(),
        "LLM":          check_llm(),
    }

    print()
    print("=" * 50)
    all_ok = all(results.values())
    if all_ok:
        print("  ALL CHECKS PASSED — system ready")
        print()
        print("  Next steps:")
        print("    1. Add PDFs to data/pdfs/")
        print("    2. uvicorn main:app --reload  (then upload via UI)")
        print("    3. python evaluate.py --ablation")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED: {', '.join(failed)}")
        print("  Fix the above before running ingestion or evaluation.")
    print("=" * 50)
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
