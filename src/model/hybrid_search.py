"""
Konkan Vani — Hybrid Search Engine
====================================
Combines two search signals:
  1. Semantic similarity (sentence-transformer embeddings via FAISS)
  2. Phonetic fuzzy matching (rapidfuzz on normalized Romanization keys)

The two scores are blended with adaptive weighting:
  - Short queries (< 4 words): heavier phonetic weight (spelling matters more)
  - Long queries (>= 4 words): heavier semantic weight (meaning matters more)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from rapidfuzz import fuzz, process

from .embeddings import get_embedding_model
from .search_index import get_search_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phonetic normalization (mirrors preprocessing.py phonetic_normalize)
# ---------------------------------------------------------------------------

def phonetic_normalize(text: str) -> str:
    """Normalize Romanized Konkani to a canonical phonetic key."""
    if not isinstance(text, str):
        return ""
    norm = text.lower()
    norm = re.sub(r'aa', 'a', norm)
    norm = re.sub(r'ee', 'i', norm)
    norm = re.sub(r'oo', 'u', norm)
    norm = re.sub(r'ph', 'f', norm)
    norm = re.sub(r'v', 'w', norm)
    norm = re.sub(r'zh', 'j', norm)
    norm = re.sub(r'[^\w\s]', '', norm)
    norm = " ".join(norm.split())
    return norm


# ---------------------------------------------------------------------------
# Script detection (mirrors preprocessing.py detect_script)
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """Classify text as Devanagari, Kannada, Roman, or Unknown."""
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    dev_chars = len(re.findall(r'[\u0900-\u097F]', text))
    kan_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
    rom_chars = len(re.findall(r'[a-zA-Z]', text))
    counts = {"Devanagari": dev_chars, "Kannada": kan_chars, "Roman": rom_chars}
    max_script = max(counts, key=counts.get)
    return max_script if counts[max_script] > 0 else "Unknown"


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Combines semantic (embedding) search with phonetic fuzzy matching.

    Scoring:
      - semantic_score: cosine similarity from FAISS (0..1 range after normalization)
      - phonetic_score: rapidfuzz ratio (0..100) normalized to 0..1
      - combined = alpha * semantic + (1-alpha) * phonetic

    Alpha adapts to query length:
      - Short query (1-3 words): alpha = 0.5  (balanced)
      - Medium query (4-6 words): alpha = 0.7  (semantic-heavy)
      - Long query (7+ words): alpha = 0.85  (almost purely semantic)
    """

    def __init__(self):
        self.model = get_embedding_model()
        self.index = get_search_index()
        self._phonetic_keys: Optional[List[str]] = None

    def _ensure_phonetic_keys(self):
        """Lazily compute phonetic keys for all indexed metadata."""
        if self._phonetic_keys is not None:
            return
        self._phonetic_keys = [
            meta.get("phonetic_key", "") for meta in self.index.metadata
        ]

    def _compute_alpha(self, query: str) -> float:
        """Adaptive blending weight based on query word count."""
        word_count = len(query.split())
        if word_count <= 3:
            return 0.5   # Short: balanced
        elif word_count <= 6:
            return 0.7   # Medium: semantic-leaning
        else:
            return 0.85  # Long: semantic-heavy

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search over the Konkani dataset.

        Args:
            query: User search query (any script or Roman transliteration)
            top_k: Number of results to return
            semantic_weight: Override adaptive alpha (0..1)

        Returns:
            List of result dicts with keys:
                text, script, source_dataset, type, phonetic_key,
                semantic_score, phonetic_score, combined_score, match_type
        """
        self._ensure_phonetic_keys()

        if not self.index._built or self.index.size == 0:
            logger.warning("Search index is empty or not built.")
            return []

        alpha = semantic_weight if semantic_weight is not None else self._compute_alpha(query)

        # --- 1. Semantic search via FAISS ---
        query_embedding = self.model.encode_single(query)
        semantic_results = self.index.search(query_embedding, top_k=min(top_k * 3, self.index.size))
        # semantic_results: [(idx, score), ...]

        # Build lookup: idx -> semantic_score
        semantic_scores = {idx: score for idx, score in semantic_results}

        # --- 2. Phonetic fuzzy matching via rapidfuzz ---
        query_phonetic = phonetic_normalize(query)
        query_script = detect_script(query)

        phonetic_scores: Dict[int, float] = {}
        if query_script == "Roman" and query_phonetic:
            # Only do phonetic matching for Roman-script queries
            # Use rapidfuzz process.extract to find best matches
            matches = process.extract(
                query_phonetic,
                self._phonetic_keys,
                scorer=fuzz.WRatio,
                limit=top_k * 3,
                score_cutoff=40,  # Minimum fuzzy match score (out of 100)
            )
            for match_text, score, idx in matches:
                phonetic_scores[idx] = score / 100.0  # Normalize to 0..1

        # --- 3. Combine scores ---
        # Union of all candidate indices
        all_indices = set(semantic_scores.keys()) | set(phonetic_scores.keys())

        combined: List[Dict[str, Any]] = []
        for idx in all_indices:
            sem = semantic_scores.get(idx, 0.0)
            phon = phonetic_scores.get(idx, 0.0)

            # For non-Roman queries, skip phonetic entirely
            if query_script != "Roman":
                combined_score = sem
                phon = 0.0
                alpha = 1.0
            else:
                combined_score = alpha * sem + (1 - alpha) * phon

            # Determine match type
            if phon > 0.7 and sem > 0.5:
                match_type = "phonetic+semantic"
            elif phon > 0.5:
                match_type = "phonetic"
            elif sem > 0.5:
                match_type = "semantic"
            else:
                match_type = "weak"

            meta = self.index.get_metadata(idx)
            combined.append({
                "text": meta.get("text", ""),
                "script": meta.get("script", "Unknown"),
                "source_dataset": meta.get("source_dataset", ""),
                "type": meta.get("type", ""),
                "phonetic_key": meta.get("phonetic_key", ""),
                "language_class": meta.get("language_class", ""),
                "semantic_score": round(sem, 4),
                "phonetic_score": round(phon, 4),
                "combined_score": round(combined_score, 4),
                "match_type": match_type,
            })

        # Sort by combined score descending
        combined.sort(key=lambda x: x["combined_score"], reverse=True)

        # Deduplicate by text (keep highest score)
        seen_texts = set()
        deduped = []
        for item in combined:
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                deduped.append(item)

        return deduped[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_indexed": self.index.size,
            "index_built": self.index._built,
            "embedding_dim": self.model.dim,
            "model_name": self.model.model_name,
        }


# Global singleton
_engine_instance: Optional[HybridSearchEngine] = None


def get_search_engine() -> HybridSearchEngine:
    """Get or create the global hybrid search engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridSearchEngine()
    return _engine_instance
