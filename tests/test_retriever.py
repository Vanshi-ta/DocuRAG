"""
Automated tests for src/retrieval/retriever.py.

Uses a FakeEmbedder (no real Sentence Transformers model download needed)
so these tests run fast and offline, while still exercising the real
FaissVectorStore, MetadataStore, and Retriever classes.

Run from the project root with:
    pytest tests/test_retriever.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore


class FakeEmbedder:
    """
    A stand-in for the real Embedder that returns pre-registered vectors
    for known text, instead of running an actual model. This lets retrieval
    logic be tested in complete isolation from Sentence Transformers.
    """

    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension
        self._registry: dict[str, np.ndarray] = {}

    def register(self, text: str, vector: np.ndarray) -> None:
        norm = vector / np.linalg.norm(vector)
        self._registry[text] = norm.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._registry:
            raise KeyError(f"FakeEmbedder has no registered vector for: {text!r}")
        return self._registry[text].reshape(1, -1)


def make_normalized_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def populated_store():
    """A FAISS index + metadata store with 3 known chunks, plus a matching FakeEmbedder."""
    dim = 8
    faiss_store = FaissVectorStore(embedding_dimension=dim)
    metadata_store = MetadataStore()
    embedder = FakeEmbedder(dimension=dim)

    chunk_vectors = [make_normalized_vector(seed=i, dim=dim) for i in range(3)]
    faiss_store.add_vectors(np.vstack(chunk_vectors))

    records = [
        ChunkRecord(
            chunk_id=f"chunk-{i}",
            chunk_text=f"This is chunk {i} about topic {i}.",
            source_filename="handbook.pdf",
            doc_id="doc-1",
            page_number=i + 1,
            chunk_index=i,
        )
        for i in range(3)
    ]
    metadata_store.add_records(records)

    # Register a query whose embedding is IDENTICAL to chunk 1's vector,
    # so we know exactly which chunk should rank #1.
    embedder.register("question about topic 1", chunk_vectors[1])

    # Register a query embedding far from all stored chunks (orthogonal-ish),
    # simulating an unrelated question.
    unrelated_vector = make_normalized_vector(seed=999, dim=dim)
    embedder.register("completely unrelated question", unrelated_vector)

    return embedder, faiss_store, metadata_store


def test_retrieve_returns_top_k_results(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve("question about topic 1", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"  # exact match ranks first
    assert results[0].similarity_score > 0.999


def test_retrieve_respects_configurable_top_k(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results_k1 = retriever.retrieve("question about topic 1", top_k=1)
    results_k3 = retriever.retrieve("question about topic 1", top_k=3)

    assert len(results_k1) == 1
    assert len(results_k3) == 3


def test_retrieve_caps_top_k_at_index_size(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    # Only 3 chunks exist — asking for 10 should not crash, just return 3.
    results = retriever.retrieve("question about topic 1", top_k=10)
    assert len(results) == 3


def test_retrieve_returns_full_metadata(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve("question about topic 1", top_k=1)
    top = results[0]

    assert top.source_filename == "handbook.pdf"
    assert top.page_number == 2  # chunk 1 -> page_number = 1 + 1 = 2
    assert "chunk 1" in top.chunk_text
    assert isinstance(top.similarity_score, float)


def test_retrieve_rejects_empty_question(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("   ", top_k=2)


def test_unrelated_question_still_returns_results_but_with_lower_scores(populated_store):
    """
    FAISS's IndexFlatIP always returns the k CLOSEST vectors it has, even if
    none of them are a good match — it never returns "nothing." This test
    documents that behavior: an unrelated question still gets results back,
    but with a visibly worse similarity score than a genuine match.
    """
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    good_match = retriever.retrieve("question about topic 1", top_k=1)
    unrelated = retriever.retrieve("completely unrelated question", top_k=1)

    assert len(unrelated) == 1  # still returns something — FAISS always does
    assert unrelated[0].similarity_score < good_match[0].similarity_score
