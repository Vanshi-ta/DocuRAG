

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

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
        
        if not ids:
            return 0
        ids_arr = np.asarray(ids, dtype="int64")
        selector = faiss.IDSelectorBatch(ids_arr)
        n_removed = self.index.remove_ids(selector)
        logger.info("Removed %d vectors from FAISS index (now %d total)", n_removed, self.ntotal)
        return n_removed

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        
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
    def load(cls, path: Path, embedding_dimension: int) -> FaissVectorStore:
        if not path.exists():
            raise FileNotFoundError(f"No FAISS index found at {path}")
        store = cls(embedding_dimension)
        store.index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index (%d vectors) from %s", store.ntotal, path)
        return store
