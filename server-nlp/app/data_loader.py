"""
server-nlp/app/data_loader.py
IdiomStore: loads the gold CSV + .npy embeddings + optional FAISS index
into memory at startup.  Asserts row counts match — catches stale
embeddings after dataset grows.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve paths relative to the project root (parent of server-nlp/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "idioms_with_phonetic_keys.csv")
_EMB_PATH = os.path.join(_PROJECT_ROOT, "data", "idiom_embeddings.npy")
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "index")
_INDEX_PATH = os.path.join(_INDEX_DIR, "konkani.index")
_META_PATH = os.path.join(_INDEX_DIR, "konkani_meta.json")


class IdiomStore:
    def __init__(self):
        # Load CSV
        if not os.path.exists(_CSV_PATH):
            raise FileNotFoundError(f"Gold dataset not found: {_CSV_PATH}")
        self.df = pd.read_csv(_CSV_PATH)
        self.count = len(self.df)

        # Load embeddings
        if not os.path.exists(_EMB_PATH):
            raise FileNotFoundError(
                f"Embeddings not found: {_EMB_PATH} — run server-nlp/scripts/build_embeddings.py first."
            )
        self.embeddings = np.load(_EMB_PATH)

        # Sanity check: row counts must match
        assert len(self.df) == len(self.embeddings), (
            f"Stale embeddings detected: CSV has {len(self.df)} rows "
            f"but embeddings has {len(self.embeddings)} vectors. "
            f"Re-run build_embeddings.py to regenerate."
        )

        # Load FAISS index if available
        self.faiss_index = None
        self.faiss_metadata = None
        self._load_faiss_index()

    def _load_faiss_index(self):
        """Try to load the pre-built FAISS index for fast semantic search."""
        if not os.path.exists(_INDEX_PATH):
            logger.warning(f"FAISS index not found at {_INDEX_PATH} — semantic search will use brute-force.")
            return

        try:
            import faiss

            self.faiss_index = faiss.read_index(_INDEX_PATH)

            if os.path.exists(_META_PATH):
                with open(_META_PATH, "r", encoding="utf-8") as f:
                    self.faiss_metadata = json.load(f)
            else:
                # Fall back to building metadata from CSV
                self.faiss_metadata = [
                    {col: str(row.get(col, "")) for col in self.df.columns}
                    for _, row in self.df.iterrows()
                ]

            assert self.faiss_index.ntotal == len(self.df), (
                f"FAISS index has {self.faiss_index.ntotal} vectors but CSV has {len(self.df)} rows. "
                f"Re-run src/build_index.py to rebuild."
            )

            logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors (dim={self.faiss_index.d})")

        except ImportError:
            logger.warning("faiss not installed — semantic search will use brute-force.")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")

    def faiss_search(self, query_emb: np.ndarray, top_k: int = 5) -> list:
        """
        Search using FAISS index if available, else brute-force cosine similarity.

        Args:
            query_emb: (dim,) L2-normalized query vector
            top_k: Number of results

        Returns:
            List of (row_index, score) tuples
        """
        if self.faiss_index is not None:
            q = query_emb.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(q, min(top_k, self.faiss_index.ntotal))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:
                    results.append((int(idx), float(score)))
            return results
        else:
            # Brute-force cosine similarity (already normalized)
            sims = self.embeddings @ query_emb.astype(np.float32)
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(int(i), float(sims[i])) for i in top_indices]

    def get_row(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "konkani_text":       str(row.get("konkani_text", "")),
            "romanized_text":     str(row.get("romanized_text", "")),
            "script":             str(row.get("script", "")),
            "marathi_meaning":    str(row.get("marathi_meaning", "")),
            "english_meaning":    str(row.get("english_meaning", "")),
            "figurative_meaning": str(row.get("figurative_meaning", "")),
            "literal_meaning":    str(row.get("literal_meaning", "")),
            "example_sentence":   str(row.get("example_sentence", "")),
            "phonetic_key":       str(row.get("phonetic_key", "")),
            "category":           str(row.get("category", "")),
            "cultural_context":   str(row.get("cultural_context", "")),
            "source":             str(row.get("source", "")),
        }

    def all_phonetic_keys(self) -> list[str]:
        return self.df["phonetic_key"].fillna("").tolist()

    def get_embedding(self, idx: int) -> np.ndarray:
        return self.embeddings[idx]


# Singleton loaded once at import time
store = IdiomStore()
