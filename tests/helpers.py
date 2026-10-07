"""Shared fakes and builders for the test suite (`from tests.helpers import ...`)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas

from src.retrieval.retriever import RetrievedChunk


class DeterministicFakeEmbedder:
    """Turns text into a reproducible pseudo-random unit vector via a hash
    seed, so the same text always embeds to the same vector without loading
    any real model."""

    embedding_dimension = 16

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).random(self.embedding_dimension, dtype=np.float64)
            vectors.append((v / np.linalg.norm(v)).astype("float32"))
        return np.vstack(vectors) if vectors else np.empty((0, self.embedding_dimension), dtype="float32")

    def embed_query(self, text):
        return self.embed_texts([text])


class RegisteredEmbedder:
    """Query embedder where each test registers the exact vector a query maps to."""

    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension
        self._registry: dict[str, np.ndarray] = {}

    def register(self, text: str, vector: np.ndarray) -> None:
        norm = vector / np.linalg.norm(vector)
        self._registry[text] = norm.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._registry:
            raise KeyError(
                f"RegisteredEmbedder has no vector registered for query: {text!r}. "
                f"Registered: {list(self._registry.keys())}"
            )
        return self._registry[text].reshape(1, -1)


class FakeRetriever:
    """Stands in for Retriever in rag_engine tests; records the last retrieve_diverse call."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_call = None

    def retrieve_diverse(self, question, top_k, candidate_pool_size=None,
                         max_chunks_per_source=None, similarity_threshold=None):
        self.last_call = (question, top_k, similarity_threshold)
        return self._chunks


def make_pdf(path: Path, lines) -> None:
    """Write a one-page PDF with each string in `lines` on its own row."""
    c = canvas.Canvas(str(path))
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    v = np.random.default_rng(seed).random(dim, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype("float32")


def near_vector(vec: np.ndarray, noise_seed: int, noise_scale: float) -> np.ndarray:
    """A unit vector close to `vec`; larger `noise_scale` means further away."""
    noise = np.random.default_rng(noise_seed).normal(0, noise_scale, size=len(vec)).astype("float32")
    v = vec + noise
    return (v / np.linalg.norm(v)).astype("float32")


def make_chunk(text="content", filename="doc.pdf", page=1, score=0.9, chunk_id="c1") -> RetrievedChunk:
    return RetrievedChunk(chunk_text=text, source_filename=filename, page_number=page,
                          similarity_score=score, chunk_id=chunk_id)
