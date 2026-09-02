"""
Retrieval score diagnostic for DocuRAG.

This script does NOT recommend a threshold value on its own — it prints
RAW cosine-similarity scores (inner product on L2-normalized MiniLM
vectors — see embedder.py / faiss_store.py) for real queries against your
CURRENTLY INDEXED documents, so you can read the actual numbers on YOUR
corpus and decide.

For each query it prints the top-N raw candidates straight from FAISS
(single-stage `retriever.retrieve(..., similarity_threshold=None)` — NOT
`retrieve_diverse`, deliberately: this script is for inspecting the raw
ranking before any threshold or diversity logic touches it), with:
    rank | score | PASS/FAIL vs --threshold | source filename | page

Usage:
    # Diagnostic mode: threshold=0 disables filtering entirely, so you see
    # every candidate FAISS actually returned, ranked.
    python scripts/tune_threshold.py --threshold 0 --top-k 15 \\
        --query "what is vanshita marks"

    # Run the standard 5-query regression set from the CGPA bug report:
    python scripts/tune_threshold.py --preset vanshita --top-k 15 --threshold 0

    # Check where the CURRENT configured threshold (config.SIMILARITY_THRESHOLD)
    # would have cut off, without overriding it:
    python scripts/tune_threshold.py --preset vanshita --top-k 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import SIMILARITY_THRESHOLD
from src.ingestion.embedder import Embedder
from src.logging_config import configure_logging
from src.pipeline import load_vector_store
from src.retrieval.retriever import Retriever

# The exact regression set from the "Vanshita marks" bug report: the first
# four are expected to retrieve real evidence, the last is expected to
# demonstrate correct unsupported-question rejection.
PRESET_VANSHITA = [
    "what is vanshita marks",
    "What is Vanshita's CGPA?",
    "What are Vanshita's Class 10 marks?",
    "What are Vanshita's Class 12 marks?",
    "What is Vanshita's education?",
    "What is Vanshita's favorite animal?",  # expected unsupported
]


def run_query(retriever: Retriever, question: str, top_k: int, threshold: float | None) -> None:
    print(f"\n{'=' * 78}\nQ: {question!r}")
    print(f"threshold used for PASS/FAIL below: {threshold}")
    print("-" * 78)

    # retrieve_raw: unfiltered, uncapped against MAX_TOP_K -- this is a
    # diagnostic-only method (see retriever.py) specifically so --top-k 15
    # actually shows 15 candidates instead of silently clamping to
    # MAX_TOP_K=10 the way the production retrieve()/retrieve_diverse()
    # methods correctly do for the app-facing path.
    chunks = retriever.retrieve_raw(question, search_width=top_k)

    if not chunks:
        print("  FAISS returned ZERO candidates — the index itself may be empty.")
        return

    print(f"  {'rank':>4} | {'score':>8} | {'pass?':>5} | {'source':<32} | page | content preview")
    for rank, c in enumerate(chunks, start=1):
        passed = "PASS" if (threshold is None or c.similarity_score >= threshold) else "FAIL"
        preview = c.chunk_text.replace("\n", " ").strip()[:80]
        print(f"  {rank:>4} | {c.similarity_score:>8.4f} | {passed:>5} | {c.source_filename:<32} | {c.page_number:>4} | {preview}...")

    n_pass = sum(1 for c in chunks if threshold is None or c.similarity_score >= threshold)
    print(f"  -> {n_pass}/{len(chunks)} candidates pass threshold={threshold}")

    sources_seen = {c.source_filename for c in chunks}
    print(f"  -> distinct source documents in top-{top_k}: {sorted(sources_seen)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw cosine-similarity scores on your indexed corpus.")
    parser.add_argument("--query", nargs="*", default=None, help="One or more ad-hoc queries.")
    parser.add_argument(
        "--preset", choices=["vanshita"], default=None,
        help="Use a built-in regression query set instead of --query.",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            "Threshold to check candidates against for the PASS/FAIL column. "
            f"Defaults to the currently configured SIMILARITY_THRESHOLD ({SIMILARITY_THRESHOLD}). "
            "Pass 0 to see every candidate marked PASS (fully permissive, for diagnosis)."
        ),
    )
    args = parser.parse_args()

    if args.preset == "vanshita":
        questions = PRESET_VANSHITA
    elif args.query:
        questions = args.query
    else:
        parser.error("Provide --query \"...\" (one or more) or --preset vanshita")
        return

    threshold = args.threshold if args.threshold is not None else SIMILARITY_THRESHOLD

    configure_logging()

    embedder = Embedder()
    try:
        faiss_store, metadata_store, registry = load_vector_store(embedder.embedding_dimension)
    except FileNotFoundError:
        print("No persisted vector store found. Index your documents first, then re-run this script.")
        return
    if faiss_store.ntotal == 0:
        print("Vector store is empty (0 chunks indexed). Index your documents first.")
        return

    print(f"Indexed documents ({len(registry.documents)}):")
    for entry in registry.list_documents():
        print(f"  - {entry.source_filename} ({entry.chunk_count} chunks, {entry.page_count} pages)")
    if not any(e.source_filename.lower().startswith("vanshita") for e in registry.documents.values()):
        print(
            "\n  NOTE: no file starting with 'vanshita' is currently in the registry above. "
            "If you expect Vanshita_Suryavanshi.pdf to be indexed, this is very likely your "
            "real problem — re-check that it was uploaded and processed successfully, not "
            "just the threshold value."
        )

    retriever = Retriever(embedder, faiss_store, metadata_store)
    for q in questions:
        run_query(retriever, q, args.top_k, threshold)

    print(
        "\n" + "=" * 78 +
        "\nHOW TO READ THIS:\n"
        "1. Check whether Vanshita_Suryavanshi.pdf appears in the candidate list at\n"
        "   all for the first four queries. If it's ABSENT even at rank 15, the\n"
        "   problem is retrieval/indexing, not the threshold, and no threshold\n"
        "   value will fix it — re-check ingestion.\n"
        "2. If it IS present, look at its score. Compare the top score for the four\n"
        "   'should answer' queries against the top score for 'favorite animal'\n"
        "   (expected unsupported). A workable threshold sits in the gap between\n"
        "   them, if one exists.\n"
        "3. If scores for genuinely relevant chunks sit BELOW your current\n"
        "   threshold, that's real evidence the threshold is too high for this\n"
        "   query phrasing/corpus — lower SIMILARITY_THRESHOLD in .env accordingly.\n"
        "   Do not guess a number without this evidence.\n"
        + "=" * 78
    )


if __name__ == "__main__":
    main()
