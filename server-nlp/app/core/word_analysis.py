"""
server-nlp/app/core/word_analysis.py
=======================================
Word-level analysis of user input using the SAME shared LaBSE model
(model_registry) — no new transformer is introduced.

The dataset is split into a cached vocabulary of unique Konkani tokens
(Devanagari words and romanized words). Each token's embedding is
encoded once and cached in memory (and on disk), so per-request word
analysis costs one encode per query word plus a matmul — no corpus-wide
recompute.

resolve_words(text) returns, for every alphabetic token in the input:
  word, best-matching dataset word, similarity, and a rough role guess
  (noun/verb/particle style hints based on known suffixes) so callers
  can understand WHAT the sentence is made of, not just which idiom
  it resembles.
"""

import os
import re
import logging
import numpy as np
import pandas as pd

from app.core.model_registry import encode

logger = logging.getLogger(__name__)

_CSV_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "data", "processed", "idioms_with_phonetic_keys.csv"))
_CACHE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "data", "processed", "word_vocab_cache.npz"))

_WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")

# Loose Konkani verb / particle suffix hints (heuristic, for role labeling)
_VERB_SUFFIXES = ("प", "ता", "लो", "ली", "लें", "चें", "चो", "की", "यता")
_KNOWN_PARTICLES = {"म्हणजे", "ना", "आनी", "जाता", "जातां", "कडेन", "निमतान"}

_vocab = None          # list[str]
_vocab_emb = None      # (V, 768) normalized
# role-hint vocabularies
_verbs = set()
_nouns = set()


def _extract_vocab():
    df = pd.read_csv(_CSV_PATH)
    words = set()
    for col in ("konkani_text", "romanized_text"):
        for t in df[col].fillna("").astype(str):
            for w in _WORD_RE.findall(t):
                if len(w) > 1:
                    words.add(w)
    return sorted(words)


def _load_vocab():
    """Load (or build + cache) the word vocabulary and its embeddings."""
    global _vocab, _vocab_emb, _verbs, _nouns
    if _vocab is not None:
        return

    vocab = _extract_vocab()

    if os.path.exists(_CACHE_PATH):
        try:
            npz = np.load(_CACHE_PATH, allow_pickle=False)
            if list(npz["vocab"]) == vocab:  # cache still matches dataset
                _vocab = vocab
                _vocab_emb = npz["emb"]
                logger.info("Word vocab cache loaded: %d tokens", len(_vocab))
                return
        except Exception as e:
            logger.warning("Word vocab cache unreadable, rebuilding: %s", e)

    logger.info("Building word-vocab embeddings for %d tokens ...", len(vocab))
    emb = encode(vocab, batch_size=256)
    _vocab = vocab
    _vocab_emb = emb.astype(np.float32)
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        np.savez_compressed(_CACHE_PATH, vocab=np.array(vocab), emb=_vocab_emb)
    except Exception as e:
        logger.warning("Could not persist word vocab cache: %s", e)


def _role_hint(word: str) -> str:
    """Very rough morphological hint — heuristic only, clearly labeled."""
    if word in _KNOWN_PARTICLES:
        return "particle"
    if word.endswith(_VERB_SUFFIXES):
        return "verb-form"
    return "content-word"


def analyze_words(text: str, top_k: int = 1):
    """
    Analyze every word of the input against the dataset vocabulary using
    the shared LaBSE model. Returns a list of per-word dicts.
    """
    if not text or not text.strip():
        return []
    _load_vocab()

    tokens = _WORD_RE.findall(text)
    if not tokens:
        return []

    tok_embs = encode(tokens)  # (T, 768) normalized
    sims = tok_embs @ _vocab_emb.T  # (T, V)

    results = []
    for tok, row in zip(tokens, sims):
        order = np.argsort(row)[::-1][:top_k]
        matches = [
            {"word": _vocab[i], "similarity": round(float(row[i]), 4)}
            for i in order
        ]
        results.append({
            "word": tok,
            "role_hint": _role_hint(tok),
            "best_matches": matches,
        })
    return results


def resolve_words(text: str, min_similarity: float = 0.65):
    """
    Word-level analysis stage used by the pipeline.

    Conservative gate — returns None unless ALL of:
      - the input has at least 2 content words (a single word is not an idiom),
      - at least 2 content words are EXACT vocabulary hits (sim >= 0.999), AND
      - at least 2/3 of the content words map to dataset words with
        cosine >= min_similarity.

    Rationale (from live smoke test): a single exact word (e.g. "gele",
    present in "पोळीत पाणी गेलें") plus moderate fuzzy matches let the
    made-up sentence "Ami gele bajar bharli pishwi" slip through at 0.79.
    Requiring >= 2 exact hits keeps fragments of real idioms in ("बैं
    सुक्तोच") while rejecting ordinary sentences built from vocab words.
    """
    analysis = analyze_words(text)
    if not analysis:
        return None

    content = [a for a in analysis if a["role_hint"] != "particle"]
    if len(content) < 2:
        return None

    best_sims = [a["best_matches"][0]["similarity"] for a in content]
    exact_hits = sum(1 for s in best_sims if s >= 0.999)
    strong = sum(1 for s in best_sims if s >= min_similarity)

    if exact_hits >= 2 and strong >= max(2, (2 * len(best_sims) + 2) // 3):
        # Compose a transparent, word-grounded response
        gloss = "; ".join(
            f"{a['word']}≈{a['best_matches'][0]['word']}"
            f"({a['best_matches'][0]['similarity']:.2f})"
            for a in analysis
        )
        return {
            "matched": True,
            "match_type": "word_level",
            "confidence": round(float(np.mean(best_sims)), 4),
            "literal_meaning": "",
            "figurative_meaning": f"Word-level analysis: {gloss}",
            "english_meaning": "Partial/word-level interpretation from dataset vocabulary — not a verified idiom match.",
            "marathi_meaning": "शब्द-स्तरावरील विश्लेषण — नक्की म्हण नाही.",
            "example_sentence": "",
            "word_analysis": analysis,
        }
    return None
