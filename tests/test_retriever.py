import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore


class FakeEmbedder:
    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension
        self._registry = {}

    def register(self, text, vector):
        norm = vector / np.linalg.norm(vector)
        self._registry[text] = norm.astype("float32")

    def embed_query(self, text):
        return self._registry[text].reshape(1, -1)


def make_normalized_vector(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.random(dim, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def populated_store():
    dim = 8
    faiss_store = FaissVectorStore(embedding_dimension=dim)
    metadata_store = MetadataStore()
    embedder = FakeEmbedder(dimension=dim)

    chunk_vectors = [make_normalized_vector(seed=i, dim=dim) for i in range(3)]
    ids = [10, 11, 12]
    faiss_store.add_vectors(np.vstack(chunk_vectors), ids)

    records = [
        ChunkRecord(vector_id=ids[i], chunk_id=f"chunk-{i}", chunk_text=f"This is chunk {i}.",
                    source_filename="handbook.pdf", doc_id="doc-1", page_number=i + 1, chunk_index=i)
        for i in range(3)
    ]
    metadata_store.add_records(records)

    embedder.register("question about topic 1", chunk_vectors[1])
    unrelated_vector = make_normalized_vector(seed=999, dim=dim)
    embedder.register("completely unrelated question", unrelated_vector)

    return embedder, faiss_store, metadata_store


def test_retrieve_returns_top_k_results(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve("question about topic 1", top_k=2, similarity_threshold=None)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].similarity_score > 0.999


def test_retrieve_raw_is_not_clamped_by_max_top_k(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    # This fixture only has 3 chunks total, so asking for search_width=100
    # should return all 3 without raising or clamping against MAX_TOP_K,
    # unlike retrieve()/retrieve_diverse() which cap their top_k argument.
    results = retriever.retrieve_raw("question about topic 1", search_width=100)
    assert len(results) == 3


def test_retrieve_raw_returns_unfiltered_scores_including_low_ones(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve_raw("completely unrelated question", search_width=3)
    assert len(results) == 3  # no threshold applied -- everything comes back


def test_retrieve_rejects_empty_question(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("   ", top_k=2)


def test_retrieve_rejects_invalid_top_k(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("question about topic 1", top_k=0)


def test_similarity_threshold_filters_low_scoring_chunks(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    # An unrelated query still gets *something* back from a flat index —
    # but a high threshold should filter it all out.
    permissive = retriever.retrieve("completely unrelated question", top_k=3, similarity_threshold=None)
    strict = retriever.retrieve("completely unrelated question", top_k=3, similarity_threshold=0.99)

    assert len(permissive) == 3
    assert len(strict) == 0


def test_similarity_threshold_keeps_genuine_match(populated_store):
    embedder, faiss_store, metadata_store = populated_store
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve("question about topic 1", top_k=1, similarity_threshold=0.9)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
