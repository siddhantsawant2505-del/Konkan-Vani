"""
server-nlp/app/core/model_registry.py
=======================================
Single shared LaBSE instance for the whole backend. Replaces the
pipeline-local lazy loader so every component (semantic match,
word-level analysis) encodes with the same model without duplicate
memory copies.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()


def get_model():
    """Lazy-load and cache sentence-transformers/LaBSE once (thread-safe)."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading LaBSE (sentence-transformers/LaBSE) ...")
                _MODEL = SentenceTransformer("sentence-transformers/LaBSE")
                logger.info("LaBSE loaded.")
    return _MODEL


def encode(texts, **kwargs):
    """Encode text(s) with the shared model (normalized, numpy)."""
    model = get_model()
    kwargs.setdefault("normalize_embeddings", True)
    kwargs.setdefault("convert_to_numpy", True)
    kwargs.setdefault("show_progress_bar", False)
    return model.encode(texts, **kwargs)
