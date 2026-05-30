#!/usr/bin/env python3
"""
Retrieval evaluation script.

Measures Hit@1, Hit@3, and Hit@5 — the fraction of questions where the
expected source (book + page) appears in the top-k reranked results.

Usage:
    python evaluate.py
    python evaluate.py --questions data/questions/questions.json
    python evaluate.py --ablation          ← runs 3-config comparison table

questions.json format:
    [
      {
        "question": "What is gradient descent?",
        "expected_page": 42,
        "book": "deep_learning_textbook"
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rag_engine import DENSE_TOP_K, BM25_TOP_K, collection, hybrid_retrieve, rerank_chunks


# ──────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _is_match(chunk: dict, expected_page: int, expected_book: str) -> bool:
    """True if this chunk's source matches the expected page and book."""
    page_match = chunk["page_number"] == expected_page
    book_match = (
        expected_book.lower() in chunk["book_name"].lower()
        if expected_book else True
    )
    return page_match and book_match


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-CONFIG EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_config(
    questions:    list[dict],
    config_name:  str,
    dense_k:      int,
    bm25_k:       int,
    use_reranker: bool,
    verbose:      bool = False,
) -> dict:
    """
    Evaluate one retrieval configuration against the full question set.

    Parameters
    ----------
    dense_k      : how many dense (ChromaDB cosine) results to retrieve
    bm25_k       : how many BM25 results to retrieve (0 = disable BM25)
    use_reranker : whether to apply cross-encoder reranking
    """
    hits_at = {1: 0, 3: 0, 5: 0}
    per_question = []

    for item in tqdm(questions, desc=config_name, unit="q", leave=False):
        query         = item["question"]
        expected_page = int(item["expected_page"])
        expected_book = item.get("book", "")

        candidates = hybrid_retrieve(query, dense_k=dense_k, bm25_k=bm25_k)

        if use_reranker:
            top5 = rerank_chunks(query, candidates, top_n=5)
        else:
            # Without reranker: rank by dense similarity, BM25-only hits fall last
            top5 = sorted(
                candidates,
                key=lambda c: c.get("dense_score", 0.0),
                reverse=True,
            )[:5]

        hit_rank = None
        for rank, chunk in enumerate(top5, start=1):
            if _is_match(chunk, expected_page, expected_book):
                hit_rank = rank
                break

        for k in (1, 3, 5):
            if hit_rank is not None and hit_rank <= k:
                hits_at[k] += 1

        if verbose:
            per_question.append({
                "question":      query,
                "expected_book": expected_book,
                "expected_page": expected_page,
                "hit_rank":      hit_rank,
                "top5_sources": [
                    f"{c['book_name']} p{c['page_number']}"
                    f" (score={c.get('rerank_score') or c.get('dense_score', 0):.3f})"
                    for c in top5
                ],
            })

    n = len(questions)
    result = {
        "config":   config_name,
        "hit_at_1": round(hits_at[1] / n, 4) if n else 0,
        "hit_at_3": round(hits_at[3] / n, 4) if n else 0,
        "hit_at_5": round(hits_at[5] / n, 4) if n else 0,
    }
    if verbose:
        result["per_question"] = per_question
    return result


# ──────────────────────────────────────────────────────────────────────────────
# FULL-PIPELINE EVALUATION  (default mode)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval(questions: list[dict]) -> dict:
    """
    Run the full pipeline (hybrid + reranker) and return metrics + per-question detail.
    This is the primary evaluation used for reporting.
    """
    return _evaluate_config(
        questions,
        config_name="Dense + BM25 + Reranker",
        dense_k=DENSE_TOP_K,
        bm25_k=BM25_TOP_K,
        use_reranker=True,
        verbose=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY  (--ablation flag)
# ──────────────────────────────────────────────────────────────────────────────

_ABLATION_CONFIGS = [
    {
        "config_name":  "Dense Only",
        "dense_k":      5,
        "bm25_k":       0,
        "use_reranker": False,
    },
    {
        "config_name":  "Dense + BM25",
        "dense_k":      DENSE_TOP_K,
        "bm25_k":       BM25_TOP_K,
        "use_reranker": False,
    },
    {
        "config_name":  "Dense + BM25 + Reranker",
        "dense_k":      DENSE_TOP_K,
        "bm25_k":       BM25_TOP_K,
        "use_reranker": True,
    },
]


def ablation_study(questions: list[dict]) -> list[dict]:
    """
    Run three configurations in sequence and return their metrics.

    Config 1 — Dense Only      : pure vector search, no BM25, no reranker
    Config 2 — Dense + BM25    : hybrid retrieval, no reranker
    Config 3 — Full Pipeline   : hybrid + cross-encoder reranker

    Each row in the returned list is directly comparable, showing the
    incremental contribution of each retrieval component.
    """
    return [_evaluate_config(questions, **cfg) for cfg in _ABLATION_CONFIGS]


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS
# ──────────────────────────────────────────────────────────────────────────────

def _print_report(metrics: dict) -> None:
    n = metrics.get("total_questions") or len(metrics.get("per_question", []))
    print()
    print("=" * 52)
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Config          : {metrics.get('config', 'Full Pipeline')}")
    print(f"  Total questions : {n}")
    print(f"  Hit@1           : {metrics['hit_at_1'] * 100:.1f}%")
    print(f"  Hit@3           : {metrics['hit_at_3'] * 100:.1f}%")
    print(f"  Hit@5           : {metrics['hit_at_5'] * 100:.1f}%")
    print("=" * 52)

    misses = [q for q in metrics.get("per_question", []) if q["hit_rank"] is None]
    if misses:
        print(f"\n  Missed ({len(misses)} questions):")
        for m in misses:
            print(f"    ✗  \"{m['question'][:60]}\"")
            print(f"       Expected : {m['expected_book']} p{m['expected_page']}")
            print(f"       Got      : {', '.join(m['top5_sources'])}")
    print()


def _print_ablation_table(rows: list[dict]) -> None:
    print()
    print("=" * 64)
    print("  ABLATION STUDY — Incremental Retrieval Improvement")
    print("=" * 64)
    print(f"  {'Configuration':<30}  {'Hit@1':>6}  {'Hit@3':>6}  {'Hit@5':>6}")
    print("  " + "─" * 58)
    baseline_h1 = rows[0]["hit_at_1"] if rows else 0
    for row in rows:
        delta = ""
        if row["hit_at_1"] > baseline_h1:
            delta = f"  (+{(row['hit_at_1'] - baseline_h1)*100:.1f}pp vs baseline)"
        print(
            f"  {row['config']:<30}  "
            f"{row['hit_at_1']*100:>5.1f}%  "
            f"{row['hit_at_3']*100:>5.1f}%  "
            f"{row['hit_at_5']*100:>5.1f}%"
            f"{delta}"
        )
    print("=" * 64)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality (Hit@k)")
    parser.add_argument(
        "--questions",
        default="data/questions/questions.json",
        help="Path to questions.json (default: data/questions/questions.json)",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval_results.json",
        help="Path to save detailed results (default: outputs/eval_results.json)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run 3-config ablation study: Dense → Hybrid → Full Pipeline",
    )
    args = parser.parse_args()

    # Guard: refuse to evaluate against an empty collection.
    col_count = collection.count()
    print(f"\nCollection count: {col_count}")
    if col_count == 0:
        print("ERROR: ChromaDB collection is empty.")
        print("Ingest your PDFs first:")
        print("  uvicorn main:app --reload  (then upload via the web UI)")
        print("  OR: from rag_engine import ingest_pdf; ingest_pdf('path/to/book.pdf')")
        return

    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"ERROR: '{q_path}' not found.")
        print()
        print("Create a questions.json file in this format:")
        print(json.dumps(
            [{"question": "What is X?", "expected_page": 1, "book": "my_book"}],
            indent=2,
        ))
        return

    questions = json.loads(q_path.read_text())
    if not questions:
        print("ERROR: questions.json is empty.")
        return

    print(f"Loaded {len(questions)} questions from '{q_path}'")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        rows = ablation_study(questions)
        _print_ablation_table(rows)
        out_path = out_path.with_name("ablation_results.json")
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"Ablation results saved → {out_path}")
    else:
        metrics = evaluate_retrieval(questions)
        _print_report(metrics)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Detailed results saved → {out_path}")


if __name__ == "__main__":
    main()
