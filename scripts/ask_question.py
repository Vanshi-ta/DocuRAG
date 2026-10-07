"""
Interactive retrieval test for DocuRAG.

Loads the persisted vector store and lets you type questions in the
terminal, printing the top-k retrieved chunks for each one. No LLM is
involved — this only tests retrieval quality in isolation, using the plain
single-stage `Retriever.retrieve()` (no diversity cap or entity fan-out, so it
is not identical to what the app retrieves).

Run from the project root with:
    python scripts/ask_question.py
"""

import _common  # noqa: F401  (makes the project root importable)

from config import DEFAULT_TOP_K
from src.retrieval.retriever import RetrievedChunk


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
    loaded = _common.load_retriever(
        "No persisted vector store found. Run `python -m src.pipeline` "
        "first to build and save one from your PDFs in data/uploads/."
    )
    if loaded is None:
        return
    retriever, faiss_store = loaded

    print(
        f"Ready — {faiss_store.ntotal} chunks indexed. "
        f"top_k={DEFAULT_TOP_K}. Type 'quit' to exit.\n"
    )

    for question in _common.prompt_questions():
        results = retriever.retrieve(question, top_k=DEFAULT_TOP_K)
        print_results(question, results)


if __name__ == "__main__":
    main()
