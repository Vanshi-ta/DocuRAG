"""
Empirical similarity-threshold inspection for DocuRAG.

SIMILARITY_THRESHOLD in config.py is a heuristic, not a calibrated
probability -- the right value depends on your embedding model and your
documents' actual content. This script does NOT recommend a number; it
prints raw cosine-similarity scores for real queries against your
CURRENTLY INDEXED documents, split into "questions you expect to be
answerable" and "questions you expect to be unsupported", so you can read
the actual score gap on YOUR corpus and set SIMILARITY_THRESHOLD in .env
accordingly.

Usage:
    python scripts/tune_threshold.py \\
        --answerable "What is Vanshita's CGPA?" "What are Manas's skills?" \\
        --unsupported "Who is the CEO of Google?" "What is the revenue in 2019?"

Or edit DEFAULT_ANSWERABLE / DEFAULT_UNSUPPORTED below and run with no args.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.embedder import Embedder
from src.logging_config import configure_logging
from src.pipeline import load_vector_store
from src.retrieval.retriever import Retriever

DEFAULT_ANSWERABLE = [
    "What is the candidate's CGPA?",
    "What programming languages does the candidate know?",
]
DEFAULT_UNSUPPORTED = [
    "Who is the CEO of Google?",
    "What was the company's revenue in 2019?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real similarity scores on your indexed corpus.")
    parser.add_argument("--answerable", nargs="*", default=DEFAULT_ANSWERABLE)
    parser.add_argument("--unsupported", nargs="*", default=DEFAULT_UNSUPPORTED)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    configure_logging()

    embedder = Embedder()
    try:
        faiss_store, metadata_store, _registry = load_vector_store(embedder.embedding_dimension)
    except FileNotFoundError:
        print("No persisted vector store found. Index your documents first.")
        return
    if faiss_store.ntotal == 0:
        print("Vector store is empty. Index your documents first.")
        return

    retriever = Retriever(embedder, faiss_store, metadata_store)

    def run(label: str, questions: list[str]) -> None:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        for q in questions:
            chunks = retriever.retrieve(q, top_k=args.top_k, similarity_threshold=None)
            print(f"\nQ: {q}")
            if not chunks:
                print("  (index returned nothing at all -- unexpected unless the index is empty)")
                continue
            for rank, c in enumerate(chunks, start=1):
                print(f"  #{rank}  score={c.similarity_score:.4f}  {c.source_filename} p.{c.page_number}")

    run("ANSWERABLE QUESTIONS (expect high top-1 scores)", args.answerable)
    run("UNSUPPORTED QUESTIONS (expect a visible score drop vs. answerable)", args.unsupported)

    print(
        "\n" + "=" * 70 +
        "\nHOW TO READ THIS:\n"
        "Look at the top-1 score for each answerable question vs. each\n"
        "unsupported question. A good SIMILARITY_THRESHOLD sits in the gap\n"
        "between the lowest answerable top-1 score and the highest unsupported\n"
        "top-1 score, if such a gap exists. If the two groups' scores overlap\n"
        "heavily, thresholding alone won't cleanly separate them for your\n"
        "corpus/model -- that's a real finding, not a bug, and worth noting as\n"
        "a limitation rather than papering over with an arbitrary cutoff.\n"
        + "=" * 70
    )


if __name__ == "__main__":
    main()
