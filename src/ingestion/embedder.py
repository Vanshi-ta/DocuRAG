

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class Embedder:
    

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info("Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension: %d", self.embedding_dimension)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        
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
        
        return self.embed_texts([query])
