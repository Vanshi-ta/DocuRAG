"""
Shared bootstrap for the command-line scripts in this folder.

Importing this module (as `import _common`, which works because Python puts
the script's own directory on sys.path) makes the project root importable, so
`config` and `src` can be imported when a script is run as
`python scripts/<name>.py`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.ingestion.embedder import Embedder
    from src.retrieval.retriever import Retriever
    from src.vectorstore.document_registry import DocumentRegistry
    from src.vectorstore.faiss_store import FaissVectorStore
    from src.vectorstore.metadata_store import MetadataStore

QUIT_WORDS = {"quit", "exit"}


def load_embedder_and_store(
    missing_store_message: str,
) -> tuple[Embedder, FaissVectorStore, MetadataStore, DocumentRegistry] | None:
    """Load the embedding model and the persisted vector store.

    Prints `missing_store_message` and returns None when nothing has been
    persisted yet, so callers can simply `return`.
    """
    from src.ingestion.embedder import Embedder  # lazy: loads the embedding stack
    from src.pipeline import load_vector_store

    embedder = Embedder()
    try:
        faiss_store, metadata_store, registry = load_vector_store(embedder.embedding_dimension)
    except FileNotFoundError:
        print(missing_store_message)
        return None
    return embedder, faiss_store, metadata_store, registry


def load_retriever(missing_store_message: str) -> tuple[Retriever, FaissVectorStore] | None:
    """Like `load_embedder_and_store`, but returns a ready Retriever plus the FAISS store."""
    loaded = load_embedder_and_store(missing_store_message)
    if loaded is None:
        return None
    embedder, faiss_store, metadata_store, _registry = loaded
    from src.retrieval.retriever import Retriever

    return Retriever(embedder, faiss_store, metadata_store), faiss_store


def prompt_questions(prompt: str = "Ask a question about your documents: ") -> Iterator[str]:
    """Yield non-empty questions typed in the terminal until 'quit'/'exit', EOF or Ctrl+C."""
    while True:
        try:
            question = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if question.lower() in QUIT_WORDS:
            return
        if question:
            yield question
