"""
FAISS vector store module for DocuRAG.

Wraps a single `faiss.IndexIDMap2(faiss.IndexFlatIP(dim))`.

IndexFlatIP alone assigns vectors sequential positions (0, 1, 2, ...) with
no way to remove one without shifting every later position — which would
silently desync the metadata store on every deletion. Wrapping it in
IndexIDMap2 lets every vector carry an explicit, caller-assigned int64 ID
that never changes and can be individually removed via `remove_ids`. This
is what makes document deletion and re-indexing safe: removing a document
just removes its specific vector IDs, and every other vector's ID (and its
corresponding MetadataStore entry) is untouched.

IndexFlatIP itself performs exact (brute-force) nearest-neighbor search via
inner product; combined with the Embedder's L2-normalized vectors, inner
product is mathematically equivalent to cosine similarity. "Flat" (no
approximation/clustering) is appropriate at this project's scale (hundreds
to low tens-of-thousands of chunks) — approximate indexes (IVF, HNSW) trade
a bit of accuracy for speed at a much larger scale than this needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dimension))

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add_vectors(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        """
        Add vectors under caller-supplied int64 IDs (see DocumentRegistry
        for ID allocation). len(ids) must equal vectors.shape[0].
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
        ids_arr = np.asarray(ids, dtype="int64")
        if ids_arr.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"ids length ({ids_arr.shape[0]}) must match vectors count ({vectors.shape[0]})"
            )

        self.index.add_with_ids(vectors, ids_arr)
        logger.info("Added %d vectors to FAISS index (now %d total)", vectors.shape[0], self.ntotal)

    def remove_ids(self, ids: Sequence[int]) -> int:
        """Remove the given vector IDs. Returns the number actually removed."""
        if not ids:
            return 0
        ids_arr = np.asarray(ids, dtype="int64")
        selector = faiss.IDSelectorBatch(ids_arr)
        n_removed = self.index.remove_ids(selector)
        logger.info("Removed %d vectors from FAISS index (now %d total)", n_removed, self.ntotal)
        return n_removed

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top_k nearest vectors. Returns (scores, ids), both
        shape (1, top_k). `ids` are the caller-assigned vector IDs (NOT
        raw FAISS positions) — pass directly into MetadataStore.get().
        A padding value of -1 appears if fewer than top_k vectors exist.
        """
        if self.index.ntotal == 0:
            raise ValueError("Cannot search an empty index — add vectors first")
        top_k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(query_vector, top_k)
        return scores, ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index (%d vectors) to %s", self.ntotal, path)

    @classmethod
    def load(cls, path: Path, embedding_dimension: int) -> "FaissVectorStore":
        if not path.exists():
            raise FileNotFoundError(f"No FAISS index found at {path}")
        store = cls(embedding_dimension)
        store.index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index (%d vectors) from %s", store.ntotal, path)
        return store
