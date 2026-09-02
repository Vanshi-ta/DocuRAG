import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore
from src.vectorstore.document_registry import DocumentRegistry


def make_random_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype("float32")


def test_add_and_search_returns_caller_assigned_ids():
    store = FaissVectorStore(embedding_dimension=8)
    vectors = make_random_vectors(5, 8, seed=1)
    ids = [100, 101, 102, 103, 104]
    store.add_vectors(vectors, ids)

    scores, returned_ids = store.search(vectors[2:3], top_k=1)
    assert returned_ids[0][0] == 102
    assert scores[0][0] > 0.999


def test_remove_ids_deletes_only_targeted_vectors():
    store = FaissVectorStore(embedding_dimension=8)
    vectors = make_random_vectors(5, 8, seed=2)
    ids = [10, 11, 12, 13, 14]
    store.add_vectors(vectors, ids)

    removed = store.remove_ids([12])
    assert removed == 1
    assert store.ntotal == 4

    # Searching for the removed vector's content should NOT return id 12 as an exact hit.
    scores, returned_ids = store.search(vectors[2:3], top_k=4)
    assert 12 not in returned_ids[0]

    # Other ids remain searchable.
    scores, returned_ids = store.search(vectors[0:1], top_k=1)
    assert returned_ids[0][0] == 10


def test_save_and_load_roundtrip_preserves_ids(tmp_path):
    store = FaissVectorStore(embedding_dimension=8)
    vectors = make_random_vectors(3, 8, seed=3)
    store.add_vectors(vectors, [500, 501, 502])

    path = tmp_path / "index.faiss"
    store.save(path)

    reloaded = FaissVectorStore.load(path, embedding_dimension=8)
    scores, ids = reloaded.search(vectors[1:2], top_k=1)
    assert ids[0][0] == 501


def test_metadata_store_keyed_by_vector_id_not_position():
    store = MetadataStore()
    store.add_records([
        ChunkRecord(vector_id=50, chunk_id="c1", chunk_text="a", source_filename="x.pdf",
                    doc_id="d1", page_number=1, chunk_index=0),
        ChunkRecord(vector_id=51, chunk_id="c2", chunk_text="b", source_filename="x.pdf",
                    doc_id="d1", page_number=2, chunk_index=1),
    ])
    assert store.get(51).chunk_text == "b"
    with pytest.raises(KeyError):
        store.get(999)


def test_metadata_store_remove_and_survives_gap():
    store = MetadataStore()
    store.add_records([
        ChunkRecord(vector_id=1, chunk_id="c1", chunk_text="a", source_filename="x.pdf",
                    doc_id="d1", page_number=1, chunk_index=0),
        ChunkRecord(vector_id=2, chunk_id="c2", chunk_text="b", source_filename="x.pdf",
                    doc_id="d1", page_number=1, chunk_index=1),
    ])
    removed = store.remove([1])
    assert removed == 1
    assert len(store) == 1
    assert store.get(2).chunk_text == "b"
    with pytest.raises(KeyError):
        store.get(1)


def test_document_registry_dedup_and_id_allocation():
    registry = DocumentRegistry()
    ids_a = registry.allocate_vector_ids(3)
    assert ids_a == [0, 1, 2]
    registry.add_document("doc-a", "a.pdf", "hash-a", page_count=2, chunk_count=3, vector_ids=ids_a)

    assert registry.find_by_hash("hash-a").source_filename == "a.pdf"
    assert registry.find_by_hash("hash-nonexistent") is None

    ids_b = registry.allocate_vector_ids(2)
    assert ids_b == [3, 4]  # never reuses ids, even before any deletion


def test_document_registry_remove_frees_correct_vector_ids():
    registry = DocumentRegistry()
    ids_a = registry.allocate_vector_ids(2)
    registry.add_document("doc-a", "a.pdf", "hash-a", 1, 2, ids_a)
    ids_b = registry.allocate_vector_ids(2)
    registry.add_document("doc-b", "b.pdf", "hash-b", 1, 2, ids_b)

    freed = registry.remove_document("doc-a")
    assert freed == ids_a
    assert registry.find_by_filename("a.pdf") is None
    assert registry.find_by_filename("b.pdf") is not None


def test_document_registry_ids_never_reused_after_deletion():
    registry = DocumentRegistry()
    ids_a = registry.allocate_vector_ids(3)
    registry.add_document("doc-a", "a.pdf", "hash-a", 1, 3, ids_a)
    registry.remove_document("doc-a")

    ids_c = registry.allocate_vector_ids(2)
    assert ids_c == [3, 4]  # continues from next_vector_id, not from freed ids


def test_document_registry_save_and_load_roundtrip(tmp_path):
    registry = DocumentRegistry()
    ids_a = registry.allocate_vector_ids(2)
    registry.add_document("doc-a", "a.pdf", "hash-a", 1, 2, ids_a)

    path = tmp_path / "documents.json"
    registry.save(path)

    reloaded = DocumentRegistry.load(path)
    assert reloaded.next_vector_id == 2
    assert reloaded.find_by_filename("a.pdf").content_hash == "hash-a"
