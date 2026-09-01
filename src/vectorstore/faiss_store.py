"""
FAISS vector store module for DocuRAG — Phase 4.

Responsible ONLY for:
    - building a FAISS index from embedding vectors
    - adding new vectors to it
    - searching for the top-k nearest vectors to a query vector
    - persisting the index to disk, and reloading it

FAISS itself only knows about vectors and sequential integer positions — it
has NO concept of "documents," "chunks," "filenames," or "page numbers."
That's exactly why this module has a partner, metadata_store.py: FAISS
hands back integer positions from a search, and metadata_store.py maps
those same positions back to the actual chunk text and metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class FaissVectorStore:
    """
    A thin, explicit wrapper around a single faiss.IndexFlatIP.

    IndexFlatIP performs *exact* (brute-force) nearest-neighbor search using
    inner product. "Flat" means no approximation and no clustering — every
    query is compared against every stored vector. For a student-project
    scale (hundreds to low tens-of-thousands of chunks), this is both fast
    enough and simplest to reason about; approximate indexes (IVF, HNSW)
    trade a small amount of accuracy for speed at a much larger scale than
    this project needs.
    """

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)

    def add_vectors(self, vectors: np.ndarray) -> Tuple[int, int]:
        """
        Add vectors to the index. Returns (start_id, end_id): the FAISS
        position range these vectors now occupy.

        IndexFlatIP assigns positions sequentially, starting at 0, in the
        exact order vectors are added — so start_id is simply the index's
        size immediately before this call. This sequential-by-insertion-
        order guarantee is the entire foundation of how we map FAISS
        positions to metadata records (Section 14 of the guide).
        """
        if vectors.dtype != np.float32:
            raise ValueError(f"FAISS requires float32 vectors, got {vectors.dtype}")
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dimension:
            got_dim = vectors.shape[1] if vectors.ndim == 2 else vectors.shape
            raise ValueError(
                f"Vector dimension mismatch: index expects dimension "
                f"{self.embedding_dimension}, got dimension {got_dim} "
                f"(full shape {vectors.shape})"
            )

        start_id = self.index.ntotal
        self.index.add(vectors)
        end_id = self.index.ntotal
        logger.info("Added %d vectors to FAISS index (now %d total)", vectors.shape[0], end_id)
        return start_id, end_id

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top_k nearest vectors to query_vector.

        query_vector must be shape (1, D). Returns (scores, indices), both
        shape (1, top_k): `scores` are inner-product similarity scores
        (== cosine similarity, since vectors are pre-normalized), and
        `indices` are the FAISS positions — pass these directly into
        MetadataStore.get() to retrieve the matching chunk text/metadata.
        """
        if self.index.ntotal == 0:
            raise ValueError("Cannot search an empty index — add vectors first")
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index (%d vectors) to %s", self.index.ntotal, path)

    @classmethod
    def load(cls, path: Path, embedding_dimension: int) -> "FaissVectorStore":
        """
        Reload a previously saved index. `embedding_dimension` must match
        the dimension the index was originally built with — we don't infer
        it from the file to keep this explicit and fail loudly (via
        FAISS's own internal check) on a mismatch rather than silently
        loading an incompatible index.
        """
        if not path.exists():
            raise FileNotFoundError(f"No FAISS index found at {path}")
        store = cls(embedding_dimension)
        store.index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index (%d vectors) from %s", store.index.ntotal, path)
        return store
