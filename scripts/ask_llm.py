"""
Interactive end-to-end RAG test for DocuRAG — Phase 6.

Loads the persisted vector store, connects to your local Ollama model, and
lets you type questions in the terminal — printing the generated answer
plus the sources it was grounded in for each one.

Run from the project root with:
    python scripts/ask_llm.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_TOP_K
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import RAGAnswer, answer_question
from src.ingestion.embedder import Embedder
from src.pipeline import load_vector_store
from src.retrieval.retriever import Retriever


def print_answer(result: RAGAnswer) -> None:
    print(f"\nQuestion: {result.question!r}")
    print(f"\nAnswer:\n{result.answer}")

    print("\nSources:")
    if not result.sources:
        print("  (none retrieved)")
    for i, s in enumerate(result.sources, start=1):
        print(
            f"  [{i}] {s.source_filename}, page {s.page_number}  "
            f"(similarity={s.similarity_score:.3f})"
        )


def main() -> None:
    print("Loading embedding model and vector store...")
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
    llm_client = OllamaClient()

    print(
        f"Ready — {faiss_store.index.ntotal} chunks indexed. "
        f"Model: {llm_client.model}. Type 'quit' to exit.\n"
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

        try:
            result = answer_question(question, retriever, llm_client, top_k=DEFAULT_TOP_K)
        except (ConnectionError, TimeoutError) as exc:
            print(f"\nLLM error: {exc}")
            continue

        print_answer(result)


if __name__ == "__main__":
    main()
