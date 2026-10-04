"""
server-nlp/app/core/model_registry.py
=======================================
Single shared encoder for the whole backend.

Loads the domain fine-tuned model ``konkan-vani-encoder-v1`` (trained with
MultipleNegativesRankingLoss on the Konkani idiom dataset).  Falls back to
the generic ``sentence-transformers/LaBSE`` if the custom checkpoint is not
found, so the server still starts on a fresh clone before training.

Every component (semantic match, word-level analysis, embedding rebuild)
shares this singleton — no duplicate weights in VRAM.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()

# Path to the fine-tuned checkpoint (relative to project root)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_CUSTOM_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "konkan-vani-encoder-v1")
_FALLBACK_MODEL = "sentence-transformers/LaBSE"


def get_model():
    """Lazy-load and cache the Konkan Vani encoder once (thread-safe).

    Prefers the fine-tuned checkpoint at ``models/konkan-vani-encoder-v1/``.
    Falls back to the generic LaBSE if the checkpoint does not exist.
    """
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                if os.path.isdir(_CUSTOM_MODEL_PATH):
                    logger.info(
                        "Loading fine-tuned model from %s ...", _CUSTOM_MODEL_PATH
                    )
                    _MODEL = SentenceTransformer(_CUSTOM_MODEL_PATH)
                    logger.info("konkan-vani-encoder-v1 loaded (custom fine-tuned).")
                else:
                    logger.warning(
                        "Custom model not found at %s — falling back to %s",
                        _CUSTOM_MODEL_PATH,
                        _FALLBACK_MODEL,
                    )
                    _MODEL = SentenceTransformer(_FALLBACK_MODEL)
                    logger.info("Fallback LaBSE loaded.")
    return _MODEL


def encode(texts, **kwargs):
    """Encode text(s) with the shared model (normalized, numpy)."""
    model = get_model()
    kwargs.setdefault("normalize_embeddings", True)
    kwargs.setdefault("convert_to_numpy", True)
    kwargs.setdefault("show_progress_bar", False)
    return model.encode(texts, **kwargs)

