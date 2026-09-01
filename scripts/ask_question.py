"""
Interactive retrieval test for DocuRAG — Phase 5.

Loads the persisted vector store (built in Phase 4) and lets you type
questions in the terminal, printing the top-k retrieved chunks for each one.
No LLM involved yet — this only tests retrieval quality in isolation.

Run from the project root with:
    python scripts/ask_question.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_TOP_K
from src.ingestion.embedder import Embedder
from src.pipeline import load_vector_store
from src.retrieval.retriever import RetrievedChunk, Retriever


def print_results(question: str, results: list[RetrievedChunk]) -> None:
    print(f"\nQuestion: {question!r}")
    if not results:
        print("  No results — the index may be empty.")
        return

    for rank, r in enumerate(results, start=1):
        preview = r.chunk_text[:220].replace("\n", " ").strip()
        print(
            f"\n  #{rank}  similarity={r.similarity_score:.3f}  "
            f"source={r.source_filename}  page={r.page_number}"
        )
        print(f"       {preview}...")


def main() -> None:
    print("Loading embedding model and vector store (one-time cost)...")
    embedder = Embedder()
    try:
        faiss_store, metadata_store = load_vector_store(embedder.embedding_dimension)
    except FileNotFoundError:
        print(
            "No persisted vector store found. Run `python -m src.pipeline` "
            "first to build and save one from your PDFs in data/uploads/."
        )
        return

    retriever = Retriever(embedder, faiss_store, metadata_store)
    print(
        f"Ready — {faiss_store.index.ntotal} chunks indexed. "
        f"top_k={DEFAULT_TOP_K}. Type 'quit' to exit.\n"
    )

    while True:
        try:
            question = input("Ask a question about your documents: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        results = retriever.retrieve(question, top_k=DEFAULT_TOP_K)
        print_results(question, results)


if __name__ == "__main__":
    main()
