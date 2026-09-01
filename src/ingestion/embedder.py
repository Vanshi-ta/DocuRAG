"""
Embedding module for DocuRAG — Phase 4.

Responsible ONLY for:
    - loading a local Sentence Transformers model ONCE
    - converting chunk text (or a query string) into vectors

No FAISS, no metadata management, no persistence — this module's entire
job is text -> vector. Everything downstream (FAISS storage, retrieval)
depends on this module producing vectors in a consistent, known dimension.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.

    IMPORTANT: construct ONE Embedder and reuse it for every chunk and every
    query in a run. Loading a model reads its weights from disk into memory
    and is relatively expensive (tens to hundreds of milliseconds); doing
    that once per query instead of once per session would make every single
    question slow for no benefit — see Phase 4 guide Section 19.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(
            "Loading embedding model '%s' (first run downloads it once, "
            "then it's cached locally)...",
            model_name,
        )
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self.embedding_dimension)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings. Returns a (N, D) float32 numpy array.

        normalize_embeddings=True L2-normalizes every vector (length 1) at
        embedding time. Combined with FAISS's inner-product index
        (IndexFlatIP), this makes inner product mathematically equivalent
        to cosine similarity — see Phase 4 guide Section 8. We also cast to
        float32 explicitly because FAISS requires it and NumPy/PyTorch
        sometimes default to float64.
        """
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a (1, D) float32 array."""
        return self.embed_texts([query])
