"""
Automated tests for src/vectorstore/faiss_store.py and metadata_store.py.

These tests use synthetic random vectors (no Sentence Transformers model
download required), so they run fast and don't need internet access.

Run from the project root with:
    pytest tests/test_vectorstore.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore


def make_random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim), dtype=np.float32)
    # L2-normalize, same as our real Embedder does, so IndexFlatIP behaves
    # like cosine similarity in these tests too.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype("float32")


def test_add_vectors_rejects_wrong_dtype():
    store = FaissVectorStore(embedding_dimension=8)
    wrong_dtype_vectors = np.random.random((3, 8)).astype("float64")
    with pytest.raises(ValueError, match="float32"):
        store.add_vectors(wrong_dtype_vectors)


def test_add_vectors_rejects_wrong_dimension():
    store = FaissVectorStore(embedding_dimension=8)
    wrong_dim_vectors = make_random_vectors(3, dim=16)
    with pytest.raises(ValueError, match="dimension"):
        store.add_vectors(wrong_dim_vectors)


def test_add_vectors_returns_correct_id_range():
    store = FaissVectorStore(embedding_dimension=8)
    first_batch = make_random_vectors(5, dim=8, seed=1)
    start, end = store.add_vectors(first_batch)
    assert (start, end) == (0, 5)

    second_batch = make_random_vectors(3, dim=8, seed=2)
    start, end = store.add_vectors(second_batch)
    assert (start, end) == (5, 8)
    assert store.index.ntotal == 8


def test_search_returns_the_exact_match_as_top_result():
    store = FaissVectorStore(embedding_dimension=16)
    vectors = make_random_vectors(20, dim=16, seed=3)
    store.add_vectors(vectors)

    # Querying with a vector identical to one already stored should return
    # that vector itself as the #1 result, with similarity ~1.0.
    query = vectors[7:8]  # keep as (1, 16) shape
    scores, indices = store.search(query, top_k=3)

    assert indices[0][0] == 7
    assert scores[0][0] > 0.999


def test_search_on_empty_index_raises():
    store = FaissVectorStore(embedding_dimension=8)
    query = make_random_vectors(1, dim=8)
    with pytest.raises(ValueError, match="empty index"):
        store.search(query, top_k=3)


def test_save_and_load_roundtrip(tmp_path):
    store = FaissVectorStore(embedding_dimension=8)
    vectors = make_random_vectors(10, dim=8, seed=4)
    store.add_vectors(vectors)

    index_path = tmp_path / "index.faiss"
    store.save(index_path)

    reloaded = FaissVectorStore.load(index_path, embedding_dimension=8)
    assert reloaded.index.ntotal == 10

    # A search against the reloaded index should behave identically.
    query = vectors[3:4]
    scores, indices = reloaded.search(query, top_k=1)
    assert indices[0][0] == 3


def test_load_missing_index_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.faiss"
    with pytest.raises(FileNotFoundError):
        FaissVectorStore.load(missing_path, embedding_dimension=8)


def test_metadata_store_get_returns_correct_record():
    store = MetadataStore()
    records = [
        ChunkRecord(
            chunk_id=f"chunk-{i}",
            chunk_text=f"text {i}",
            source_filename="doc.pdf",
            doc_id="doc-1",
            page_number=1,
            chunk_index=i,
        )
        for i in range(5)
    ]
    store.add_records(records)

    assert len(store) == 5
    assert store.get(3).chunk_id == "chunk-3"


def test_metadata_store_get_out_of_range_raises():
    store = MetadataStore()
    store.add_records([
        ChunkRecord("c1", "text", "doc.pdf", "doc-1", 1, 0),
    ])
    with pytest.raises(IndexError, match="out of sync"):
        store.get(5)


def test_metadata_store_save_and_load_roundtrip(tmp_path):
    store = MetadataStore()
    store.add_records([
        ChunkRecord("c1", "some text", "doc.pdf", "doc-1", 2, 0),
        ChunkRecord("c2", "more text", "doc.pdf", "doc-1", 2, 1),
    ])

    path = tmp_path / "metadata.json"
    store.save(path)

    reloaded = MetadataStore.load(path)
    assert len(reloaded) == 2
    assert reloaded.get(1).chunk_text == "more text"
    assert reloaded.get(1).page_number == 2


def test_faiss_and_metadata_store_stay_in_sync_after_two_batches():
    """
    Simulates adding two separate documents' worth of chunks in two
    batches, and confirms FAISS position i still matches metadata
    record i for every i, across both batches.
    """
    faiss_store = FaissVectorStore(embedding_dimension=8)
    metadata_store = MetadataStore()

    batch_1_vectors = make_random_vectors(4, dim=8, seed=10)
    batch_1_records = [
        ChunkRecord(f"c{i}", f"batch1-text-{i}", "doc1.pdf", "doc-1", 1, i)
        for i in range(4)
    ]
    faiss_store.add_vectors(batch_1_vectors)
    metadata_store.add_records(batch_1_records)

    batch_2_vectors = make_random_vectors(3, dim=8, seed=20)
    batch_2_records = [
        ChunkRecord(f"c{i}", f"batch2-text-{i}", "doc2.pdf", "doc-2", 1, i)
        for i in range(3)
    ]
    faiss_store.add_vectors(batch_2_vectors)
    metadata_store.add_records(batch_2_records)

    assert faiss_store.index.ntotal == len(metadata_store) == 7

    # Position 5 should be the 2nd vector of batch 2 (positions 4,5,6 = batch 2).
    query = batch_2_vectors[1:2]
    scores, indices = faiss_store.search(query, top_k=1)
    matched_position = indices[0][0]
    assert matched_position == 5
    assert metadata_store.get(matched_position).chunk_text == "batch2-text-1"
