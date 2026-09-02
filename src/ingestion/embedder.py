"""
Embedding module for DocuRAG.

Responsible ONLY for loading a local Sentence Transformers model once and
converting text into vectors. No FAISS, no metadata management here.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model. Construct ONE Embedder
    and reuse it for every chunk and every query in a run — loading model
    weights from disk is relatively expensive and should happen once per
    process, not once per query.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info("Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self.embedding_dimension)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings into a (N, D) float32 array, L2-normalized
        so FAISS inner-product search behaves as cosine similarity.
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
