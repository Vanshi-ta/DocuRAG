"""
Interactive end-to-end RAG test for DocuRAG.

Loads the persisted vector store, connects to your local Ollama model, and
lets you type questions in the terminal — printing the generated answer
plus the sources it was grounded in for each one.

Run from the project root with:
    python scripts/ask_llm.py
"""

import _common  # noqa: F401  (makes the project root importable)

from config import DEFAULT_TOP_K
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import RAGAnswer, answer_question


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
    loaded = _common.load_retriever(
        "No persisted vector store found. Run `python -m src.pipeline` "
        "first to build and save one from your PDFs in data/uploads/."
    )
    if loaded is None:
        return
    retriever, faiss_store = loaded
    llm_client = OllamaClient()

    print(
        f"Ready — {faiss_store.ntotal} chunks indexed. "
        f"Model: {llm_client.model}. Type 'quit' to exit.\n"
    )

    for question in _common.prompt_questions():
        try:
            result = answer_question(question, retriever, llm_client, top_k=DEFAULT_TOP_K)
        except (ConnectionError, TimeoutError) as exc:
            print(f"\nLLM error: {exc}")
            continue

        print_answer(result)


if __name__ == "__main__":
    main()
