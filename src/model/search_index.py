"""
Konkan Vani — FAISS Search Index
=================================
Builds and queries a FAISS index over precomputed sentence embeddings.
Uses IndexFlatIP (inner product) with L2-normalized vectors = cosine similarity.
"""

import os
import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple

from .embeddings import get_embedding_model, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Directory for persisted index
INDEX_DIR = os.path.join("data", "processed", "index")


class SearchIndex:
    """
    FAISS-based semantic search index over the Konkani dataset.

    Stores embeddings + metadata so we can retrieve full idiom entries
    by vector similarity.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []  # Parallel to index vectors
        self._built = False

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ):
        """
        Build the index from precomputed embeddings and metadata.

        Args:
            embeddings: np.ndarray of shape (N, dim), L2-normalized
            metadata: List of N dicts, each with at least 'text', 'script',
                      'source_dataset', 'type', 'phonetic_key'
        """
        assert embeddings.shape[1] == self.dim, (
            f"Expected dim {self.dim}, got {embeddings.shape[1]}"
        )
        assert len(embeddings) == len(metadata), (
            f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) must match"
        )

        logger.info(f"Building FAISS index with {len(embeddings)} vectors (dim={self.dim})...")

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
        self._built = True

        logger.info(f"Index built. Total vectors: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Search the index for nearest neighbors.

        Args:
            query_embedding: (dim,) query vector, L2-normalized
            top_k: Number of results to return
            score_threshold: Minimum cosine similarity score

        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        if not self._built or self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            if score >= score_threshold:
                results.append((int(idx), float(score)))

        return results

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a given index position."""
        return self.metadata[idx]

    def save(self, directory: Optional[str] = None):
        """Persist index and metadata to disk."""
        directory = directory or INDEX_DIR
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "konkani.index")
        meta_path = os.path.join(directory, "konkani_meta.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=None)

        logger.info(f"Index saved to {directory} ({self.index.ntotal} vectors)")

    def load(self, directory: Optional[str] = None):
        """Load index and metadata from disk."""
        directory = directory or INDEX_DIR
        index_path = os.path.join(directory, "konkani.index")
        meta_path = os.path.join(directory, "konkani_meta.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.dim = self.index.d
        self._built = True
        logger.info(f"Index loaded: {self.index.ntotal} vectors from {directory}")

    @property
    def size(self) -> int:
        return self.index.ntotal if self._built and self.index else 0


# Global singleton
_index_instance: Optional[SearchIndex] = None


def get_search_index() -> SearchIndex:
    """Get or create the global search index instance."""
    global _index_instance
    if _index_instance is None:
        _index_instance = SearchIndex()
    return _index_instance
