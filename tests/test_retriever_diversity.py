"""
Regression test for the exact failure mode reported against real usage:

  "What are the skills of Manas and Vanshita?" -> incomplete answer
  "What skill does Vanshita have that Manas does not?" -> false
      "I could not find this information", even though both resumes
      were indexed.

Root cause (see docs/RETRIEVAL.md): a single top-k FAISS search can let
one document's chunks fill every slot, starving other relevant documents
of any representation in the context handed to the LLM.

This test builds a synthetic corpus that reproduces that exact geometry
(one document's 4 chunks score higher than another document's 1 chunk for
a shared query embedding) using a FakeEmbedder with hand-placed vectors --
not a real model, since no real PDFs are available in this environment --
and verifies:

  1. The OLD single-stage `retriever.retrieve()` reproduces the bug (one
     document dominates, the other is fully absent).
  2. The NEW `retriever.retrieve_diverse()` fixes it (both documents
     represented, capped per source).
  3. The NEW `retrieve_for_question()` entity fan-out guarantees BOTH
     named entities appear even when a single query embedding drifts
     toward one of them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import Retriever, retrieve_for_question
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore


class FakeEmbedder:
    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension
        self._registry = {}

    def register(self, text: str, vector: np.ndarray) -> None:
        norm = vector / np.linalg.norm(vector)
        self._registry[text] = norm.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._registry:
            raise KeyError(
                f"FakeEmbedder has no vector registered for query: {text!r}. "
                f"Registered: {list(self._registry.keys())}"
            )
        return self._registry[text].reshape(1, -1)


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def dominated_corpus():
    """
    4 chunks from vanshita.pdf, all clustered near the query vector
    (scores ~0.95-0.99), and 1 chunk from manas.pdf placed further away
    (score ~0.55) but still well above a reasonable relevance threshold --
    exactly the geometry that makes a plain top_k=4 search return 100%
    Vanshita and 0% Manas, reproducing the reported bug.
    """
    dim = 8
    faiss_store = FaissVectorStore(embedding_dimension=dim)
    metadata_store = MetadataStore()
    embedder = FakeEmbedder(dimension=dim)

    base = unit_vector(seed=1, dim=dim)

    def near(vec, noise_seed, noise_scale):
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, noise_scale, size=dim).astype("float32")
        v = vec + noise
        return (v / np.linalg.norm(v)).astype("float32")

    vanshita_vectors = [near(base, seed, 0.02) for seed in range(10, 14)]  # very close to query
    manas_vector = near(base, 900, 0.55)  # further, but still plausibly relevant

    vector_ids = [1, 2, 3, 4, 5]
    faiss_store.add_vectors(np.vstack(vanshita_vectors + [manas_vector]), vector_ids)

    records = [
        ChunkRecord(vector_id=1, chunk_id="v1", chunk_text="Vanshita skills: Python, SQL, Java.",
                    source_filename="Vanshita_Suryavanshi.pdf", doc_id="d1", page_number=1, chunk_index=0),
        ChunkRecord(vector_id=2, chunk_id="v2", chunk_text="Vanshita education: B.Tech CS, CGPA 9.0.",
                    source_filename="Vanshita_Suryavanshi.pdf", doc_id="d1", page_number=1, chunk_index=1),
        ChunkRecord(vector_id=3, chunk_id="v3", chunk_text="Vanshita internship: Data Analyst at Acme.",
                    source_filename="Vanshita_Suryavanshi.pdf", doc_id="d1", page_number=2, chunk_index=2),
        ChunkRecord(vector_id=4, chunk_id="v4", chunk_text="Vanshita projects: ML pipeline, web app.",
                    source_filename="Vanshita_Suryavanshi.pdf", doc_id="d1", page_number=2, chunk_index=3),
        ChunkRecord(vector_id=5, chunk_id="m1", chunk_text="Manas skills: Python, Java, JavaScript.",
                    source_filename="Manas_Resume.pdf", doc_id="d2", page_number=1, chunk_index=0),
    ]
    metadata_store.add_records(records)

    combined_query = "What are the skills of Manas and Vanshita?"
    embedder.register(combined_query, base)
    embedder.register(f"Vanshita {combined_query}", vanshita_vectors[0])
    embedder.register(f"Manas {combined_query}", manas_vector)

    return embedder, faiss_store, metadata_store, combined_query


def test_old_single_stage_retrieve_reproduces_the_reported_bug(dominated_corpus):
    embedder, faiss_store, metadata_store, query = dominated_corpus
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve(query, top_k=4, similarity_threshold=None)

    sources = {r.source_filename for r in results}
    assert sources == {"Vanshita_Suryavanshi.pdf"}  # bug reproduced: Manas is fully absent
    assert len(results) == 4


def test_retrieve_diverse_includes_both_documents(dominated_corpus):
    embedder, faiss_store, metadata_store, query = dominated_corpus
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve_diverse(
        query, top_k=4, candidate_pool_size=5, max_chunks_per_source=2, similarity_threshold=None
    )

    sources = {r.source_filename for r in results}
    assert "Manas_Resume.pdf" in sources  # fixed: Manas now represented
    assert "Vanshita_Suryavanshi.pdf" in sources
    # Note: this corpus only has ONE Manas chunk, so filling top_k=4 with a
    # strict 2-per-source cap is mathematically impossible (2 Vanshita + 1
    # Manas = 3 < 4) -- the backfill step correctly takes a 3rd Vanshita
    # chunk to reach top_k rather than returning fewer than requested. The
    # cap is still doing its job: Manas is present at all, which the old
    # single-stage retrieve() completely failed to guarantee.
    manas_count = sum(1 for r in results if r.source_filename == "Manas_Resume.pdf")
    assert manas_count == 1
    assert len(results) == 4


def test_retrieve_diverse_strictly_respects_cap_when_enough_diverse_candidates_exist():
    """
    Cleaner test of the cap itself: 4 Vanshita chunks + 2 Manas chunks (now
    enough Manas chunks to fill the cap without backfilling into Vanshita),
    top_k=4, cap=2 -> expect EXACTLY 2 from each source, no backfill needed.
    """
    dim = 8
    faiss_store = FaissVectorStore(embedding_dimension=dim)
    metadata_store = MetadataStore()
    embedder = FakeEmbedder(dimension=dim)

    base = unit_vector(seed=1, dim=dim)

    def near(vec, noise_seed, noise_scale):
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, noise_scale, size=dim).astype("float32")
        v = vec + noise
        return (v / np.linalg.norm(v)).astype("float32")

    vanshita_vectors = [near(base, seed, 0.02) for seed in range(10, 14)]
    manas_vectors = [near(base, seed, 0.4) for seed in (901, 902)]

    ids = [1, 2, 3, 4, 5, 6]
    faiss_store.add_vectors(np.vstack(vanshita_vectors + manas_vectors), ids)

    records = [
        ChunkRecord(vector_id=1, chunk_id="v1", chunk_text="Vanshita chunk 1", source_filename="Vanshita.pdf",
                    doc_id="d1", page_number=1, chunk_index=0),
        ChunkRecord(vector_id=2, chunk_id="v2", chunk_text="Vanshita chunk 2", source_filename="Vanshita.pdf",
                    doc_id="d1", page_number=1, chunk_index=1),
        ChunkRecord(vector_id=3, chunk_id="v3", chunk_text="Vanshita chunk 3", source_filename="Vanshita.pdf",
                    doc_id="d1", page_number=2, chunk_index=2),
        ChunkRecord(vector_id=4, chunk_id="v4", chunk_text="Vanshita chunk 4", source_filename="Vanshita.pdf",
                    doc_id="d1", page_number=2, chunk_index=3),
        ChunkRecord(vector_id=5, chunk_id="m1", chunk_text="Manas chunk 1", source_filename="Manas.pdf",
                    doc_id="d2", page_number=1, chunk_index=0),
        ChunkRecord(vector_id=6, chunk_id="m2", chunk_text="Manas chunk 2", source_filename="Manas.pdf",
                    doc_id="d2", page_number=1, chunk_index=1),
    ]
    metadata_store.add_records(records)

    query = "skills query"
    embedder.register(query, base)
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retriever.retrieve_diverse(
        query, top_k=4, candidate_pool_size=6, max_chunks_per_source=2, similarity_threshold=None
    )

    counts = {}
    for r in results:
        counts[r.source_filename] = counts.get(r.source_filename, 0) + 1

    assert counts == {"Vanshita.pdf": 2, "Manas.pdf": 2}  # strict cap held exactly, no backfill triggered
    assert len(results) == 4


def test_entity_fanout_guarantees_both_named_entities_present(dominated_corpus):
    embedder, faiss_store, metadata_store, query = dominated_corpus
    retriever = Retriever(embedder, faiss_store, metadata_store)

    results = retrieve_for_question(
        query, retriever, top_k=4, candidate_pool_size=5,
        max_chunks_per_source=2, similarity_threshold=None, entity_fanout_enabled=True,
    )

    sources = {r.source_filename for r in results}
    assert sources == {"Vanshita_Suryavanshi.pdf", "Manas_Resume.pdf"}
    manas_chunks = [r for r in results if r.source_filename == "Manas_Resume.pdf"]
    assert len(manas_chunks) >= 1
    assert "Manas skills" in manas_chunks[0].chunk_text


def test_entity_fanout_disabled_falls_back_to_old_behavior(dominated_corpus):
    embedder, faiss_store, metadata_store, query = dominated_corpus
    retriever = Retriever(embedder, faiss_store, metadata_store)

    # With fan-out off, retrieve_for_question degrades to a single
    # retrieve_diverse call -- diversity capping still helps, but the
    # guarantee both entities appear is weaker (no dedicated sub-query for
    # each). Included so the fan-out's marginal contribution is visible.
    results = retrieve_for_question(
        query, retriever, top_k=4, candidate_pool_size=5,
        max_chunks_per_source=2, similarity_threshold=None, entity_fanout_enabled=False,
    )
    sources = {r.source_filename for r in results}
    # retrieve_diverse alone still recovers Manas here because he's within
    # the candidate pool (pool_size=5 covers all 5 chunks in this corpus).
    assert "Manas_Resume.pdf" in sources
