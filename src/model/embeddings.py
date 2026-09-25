"""
Konkan Vani — Embedding Model
==============================
Wraps sentence-transformers to produce multilingual embeddings for Konkani text.
Model: paraphrase-multilingual-MiniLM-L12-v2
  - 384-dimensional embeddings
  - Trained on 50+ languages including Indic (Devanagari, Kannada, Latin scripts)
  - Optimized for semantic similarity (paraphrase detection)
"""

import os
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default model — multilingual, lightweight, good for Indic scripts
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Sentence-transformer embedding model for Konkani text."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        self.model_name = model_name
        self.model = None
        self.device = device
        self._loaded = False

    def load(self):
        """Load the sentence-transformer model (lazy loading)."""
        if self._loaded:
            return

        logger.info(f"Loading embedding model: {self.model_name} ...")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._loaded = True
        logger.info(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding
            show_progress: Show tqdm progress bar
            normalize: L2-normalize embeddings (for cosine similarity via dot product)

        Returns:
            np.ndarray of shape (len(texts), EMBEDDING_DIM)
        """
        self.load()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM


# Global singleton — lazy loaded on first use
_model_instance: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Get or create the global embedding model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel(model_name)
    return _model_instance
