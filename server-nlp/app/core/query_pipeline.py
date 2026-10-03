"""
server-nlp/app/core/query_pipeline.py

Two-stage matching pipeline:
  Stage 1 — fuzzy_match():  phonetic-key fuzzy match via rapidfuzz
                             (process.extractOne with token_sort_ratio)
  Stage 2 — semantic_match(): LaBSE cosine similarity fallback
                             (FAISS index if available, brute-force else)
                             (threshold 0.55)
  resolve_idiom():          tries Stage 1, falls back to Stage 2,
                             returns a structured result dict.
"""

import sys
import os
import logging
import numpy as np
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# Ensure server-nlp root directory is on sys.path
_SERVER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from app.data_loader import store
from app.core.phonetic_normalize import phonetic_normalize
from app.core.dynamic_fallback import resolve_dynamic_fallback
from app.core.word_analysis import resolve_words, analyze_words

# ── Constants ──────────────────────────────────────────────────────────────
FUZZY_THRESHOLD    = 75    # rapidfuzz token_sort_ratio (0–100)
# Raised from 0.55 after diagnostics: unrelated Konkani sentences scored
# up to 0.6086 against the pattern-frame corpus (false positives). Real
# exact-copy hits score >= 0.7085, so 0.63 sits in the separation gap.
SEMANTIC_THRESHOLD = 0.63  # cosine similarity (0–1, normalized vecs)

# (LaBSE is loaded once via app.core.model_registry — shared with
#  word_analysis so both encode with the same model instance.)
from app.core.model_registry import encode as _encode


# ── Stage 1: Phonetic / Script Fuzzy Match ─────────────────────────────────
def fuzzy_match(text: str, store, threshold: int = FUZZY_THRESHOLD) -> dict | None:
    """
    Normalize input via phonetic_normalize, search phonetic keys and konkani_text.
    Returns the matched row + score if above threshold, else None.
    """
    query_key = phonetic_normalize(text)
    if not query_key:
        return None

    # Check for Devanagari script match first
    import re
    is_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    
    if is_devanagari:
        konkani_texts = [str(x) for x in store.df["konkani_text"].fillna("")]
        result = process.extractOne(
            query_key,
            konkani_texts,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            match_text, score, original_idx = result
            row = store.get_row(original_idx)
            return {
                "row": row,
                "score": score,
                "idx": original_idx,
            }

    # Phonetic Romanized match
    stored_keys = store.all_phonetic_keys()
    indexed_keys = [(i, key) for i, key in enumerate(stored_keys) if key]

    if not indexed_keys:
        return None

    result = process.extractOne(
        query_key,
        [key for _, key in indexed_keys],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )

    if result is None:
        return None

    match_text, score, match_idx = result
    original_idx = indexed_keys[match_idx][0]

    row = store.get_row(original_idx)
    return {
        "row": row,
        "score": score,
        "idx": original_idx,
    }


# ── Stage 2: Semantic Embedding Match ─────────────────────────────────────
def semantic_match(text: str, store, threshold: float = SEMANTIC_THRESHOLD) -> dict | None:
    """
    Encode input with LaBSE, compute cosine similarity against store's
    embeddings using FAISS index (if available) or brute-force dot product.
    Return the best row + score if above threshold, else None.
    """
    query_emb = _encode([text])[0]  # shape (768,)

    results = store.faiss_search(query_emb, top_k=1)
    if not results:
        return None

    best_idx, best_score = results[0]

    if best_score >= threshold:
        row = store.get_row(best_idx)
        return {
            "row": row,
            "score": best_score,
            "idx": best_idx,
        }

    return None


# ── Public API ─────────────────────────────────────────────────────────────
def resolve_idiom(text: str, store) -> dict:
    """
    Stage resolver:
      Stage 1 — Phonetic/Devanagari Fuzzy Match
      Stage 2 — Semantic Embedding Match (LaBSE + FAISS)
      Stage 3 — Word-level Analysis (shared LaBSE, conservative gate)
      Stage 4 — Dynamic AI Fallback (opt-in)
    Returns a structured dict with matched/match_type/confidence + idiom fields,
    or {"matched": False} if all fail. Every response carries word_analysis.
    """
    result = None

    # Stage 1: Fuzzy match
    fuzzy_result = fuzzy_match(text, store)
    if fuzzy_result is not None:
        row = fuzzy_result["row"]
        result = {
            "matched": True,
            "match_type": "phonetic",
            "confidence": round(fuzzy_result["score"] / 100.0, 4),
            "literal_meaning":    row["literal_meaning"],
            "figurative_meaning": row["figurative_meaning"],
            "english_meaning":    row["english_meaning"],
            "marathi_meaning":    row["marathi_meaning"],
            "example_sentence":   row["example_sentence"],
        }

    # Stage 2: Semantic match
    if result is None:
        sem_result = semantic_match(text, store)
        if sem_result is not None:
            row = sem_result["row"]
            result = {
                "matched": True,
                "match_type": "semantic",
                "confidence": round(sem_result["score"], 4),
                "literal_meaning":    row["literal_meaning"],
                "figurative_meaning": row["figurative_meaning"],
                "english_meaning":    row["english_meaning"],
                "marathi_meaning":    row["marathi_meaning"],
                "example_sentence":   row["example_sentence"],
            }

    # Stage 3: Word-level analysis (conservative — exact vocab hit + majority)
    if result is None:
        result = resolve_words(text)

    # Stage 4: Dynamic AI Fallback (opt-in via DYNAMIC_FALLBACK_ENABLED=1)
    # Can fabricate answers for idioms not in the dataset — disabled by default.
    if result is None:
        result = resolve_dynamic_fallback(text)

    # Attach word-level analysis to EVERY response for transparency
    try:
        analysis = analyze_words(text)
        if result is None:
            return {"matched": False, "word_analysis": analysis}
        result["word_analysis"] = analysis
    except Exception:
        if result is None:
            return {"matched": False}

    return result


def search_idioms(query: str, top_k: int = 10) -> list[dict]:
    """
    Multi-result hybrid search combining substring, phonetic fuzzy matching,
    and semantic vector search. Returns up to top_k ranked matches.
    """
    if not query or not query.strip():
        return []

    q = query.strip()
    q_norm = q.lower()
    query_key = phonetic_normalize(q)
    results = []
    seen_ids = set()

    # 1. Exact or partial substring matching in Romanized or Devanagari text
    for idx in range(store.count):
        row = store.get_row(idx)
        rom = row.get("romanized_text", "").lower()
        konk = row.get("konkani_text", "")
        if q_norm and (q_norm in rom or q in konk):
            seen_ids.add(idx)
            entry = dict(row)
            entry["id"] = str(idx)
            entry["match_type"] = "exact" if (q_norm == rom or q == konk) else "phonetic"
            entry["confidence"] = 1.0 if entry["match_type"] == "exact" else 0.95
            entry["matched_on"] = "romanized_text" if q_norm in rom else "konkani_text"
            results.append(entry)
            if len(results) >= top_k:
                break

    # 2. Phonetic / String Fuzzy Matching via rapidfuzz
    import re
    is_devanagari = bool(re.search(r'[\u0900-\u097F]', q))

    if is_devanagari and query_key:
        konkani_texts = [str(x) for x in store.df["konkani_text"].fillna("")]
        top_fuzzy = process.extract(
            query_key,
            konkani_texts,
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
            score_cutoff=FUZZY_THRESHOLD,
        )
        for match_text, score, idx in top_fuzzy:
            if idx not in seen_ids:
                seen_ids.add(idx)
                row = store.get_row(idx)
                entry = dict(row)
                entry["id"] = str(idx)
                entry["match_type"] = "phonetic"
                entry["confidence"] = round(score / 100.0, 4)
                entry["matched_on"] = "konkani_text"
                results.append(entry)

    stored_keys = store.all_phonetic_keys()
    indexed_keys = [(i, key) for i, key in enumerate(stored_keys) if key]
    if query_key and indexed_keys:
        top_fuzzy = process.extract(
            query_key,
            [key for _, key in indexed_keys],
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
            score_cutoff=FUZZY_THRESHOLD,
        )
        for match_text, score, m_idx in top_fuzzy:
            orig_idx = indexed_keys[m_idx][0]
            if orig_idx not in seen_ids:
                seen_ids.add(orig_idx)
                row = store.get_row(orig_idx)
                entry = dict(row)
                entry["id"] = str(orig_idx)
                entry["match_type"] = "phonetic"
                entry["confidence"] = round(score / 100.0, 4)
                entry["matched_on"] = "phonetic_key"
                results.append(entry)

    # 3. Semantic Search using LaBSE embedding + FAISS (or brute-force)
    try:
        query_emb = _encode([q])[0]
        sem_matches = store.faiss_search(query_emb, top_k=top_k)
        for idx, sim in sem_matches:
            if sim >= SEMANTIC_THRESHOLD and idx not in seen_ids:
                seen_ids.add(idx)
                row = store.get_row(idx)
                entry = dict(row)
                entry["id"] = str(idx)
                entry["match_type"] = "semantic"
                entry["confidence"] = round(float(sim), 4)
                entry["matched_on"] = "semantic_embedding"
                results.append(entry)
    except Exception as e:
        logger.warning(f"Semantic search error in search_idioms: {e}")

    results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return results[:top_k]

