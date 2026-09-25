"""
server-nlp/scripts/diagnose_matches.py
=======================================
DIAGNOSTIC ONLY — read-only. Modifies no matching code and no thresholds.

Loads LaBSE + dataset CSV + embeddings, then for a battery of test queries
prints, for EACH query:
  - top-3 semantic matches with cosine similarity vs ALL stored embeddings
  - best phonetic fuzzy score (rapidfuzz token_sort_ratio) using the
    existing phonetic_normalize function — against both phonetic_key
    and konkani_text (Devanagari) columns

Test cases:
  a) 5 idioms copied EXACTLY from the dataset (konkani_text, verbatim)
  b) the same 5 with minor rephrasing / spelling variation
  c) 3 unrelated Konkani sentences that should NOT match anything

Usage:  python -X utf8 server-nlp/scripts/diagnose_matches.py
"""

import os
import sys
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.stdout.reconfigure(encoding="utf-8")
_SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.core.phonetic_normalize import phonetic_normalize  # existing function

CSV_PATH = os.path.join("data", "processed", "idioms_with_phonetic_keys.csv")
EMB_PATH = os.path.join("data", "idiom_embeddings.npy")

# ── Test queries ─────────────────────────────────────────────────────────────
# (a) EXACT copies of konkani_text from the dataset (rows 0, 5, 9, 150, 3000)
EXACT = [
    "हाताक चून लावप",                                # row 0   (curated)
    "ताका पाय ना मुंडो ना",                          # row 5   (curated)
    "आग्या मोरांक पांखां",                           # row 9   (curated)
    "तारवां बोलप म्हणजे जोड मेळटा",                  # row 150 (pattern_v1)
    "गिरेस्त पळोवप म्हणजे दुख्ख जाता",               # row 3000 (pattern_v1)
]

# (b) minor rephrasing / spelling variation of the same 5
REPHRASED = [
    "हाताक चुन लावप",                                # chūn → chun (vowel sign dropped)
    "ताका पांय ना मुंडो ना",                         # पाय → पांय
    "आग्या मोरांके पांखां",                          # extra vowel sign
    "तारवां बोलता म्हणजे जोड मेळटा",                 # verb inflected
    "गिरेस्त पळोवता म्हणजे दुख्ख जाता",              # verb inflected
]

# (c) unrelated Konkani sentences — should NOT match
UNRELATED = [
    "आय आज बाजारा कडेन गेल्लो",                      # I went to the market today
    "ती शाळेंत भुरग्यांक शिकयता",                    # She teaches children at school
    "चड पावसा निमतान रस्तो भिजलो",                   # The road got wet due to heavy rain
]


def main():
    print("=" * 78)
    print("KONKAN VANI — MATCH DIAGNOSTICS (read-only)")
    print("=" * 78)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    emb = np.load(EMB_PATH)
    print(f"\nCSV rows            : {len(df)}")
    print(f"Embeddings shape    : {emb.shape}")
    print(f"Aligned             : {len(df) == emb.shape[0]}")
    print(f"Embedding L2 norms  : min={np.linalg.norm(emb, axis=1).min():.6f} "
          f"max={np.linalg.norm(emb, axis=1).max():.6f}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading LaBSE ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")

    konkani_texts = df["konkani_text"].fillna("").astype(str).tolist()
    phonetic_keys = df["phonetic_key"].fillna("").astype(str).tolist()

    def diagnose(label, query):
        print("\n" + "-" * 78)
        print(f"[{label}] QUERY: {query}")

        # encode the raw query exactly as semantic_match() would
        q_emb = model.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)[0]
        sims = emb @ q_emb  # cosine (both normalized)
        top3 = np.argsort(sims)[::-1][:3]
        print("  SEMANTIC top-3 (cosine vs ALL embeddings):")
        for rank, idx in enumerate(top3, 1):
            print(f"    {rank}. [{sims[idx]:.4f}] idx={idx} "
                  f"{konkani_texts[idx]}  ({df.iloc[idx]['source']})")

        q_key = phonetic_normalize(query)
        print(f"  PHONETIC key of query: {q_key!r}")

        fk = process.extractOne(q_key, phonetic_keys, scorer=fuzz.token_sort_ratio,
                                score_cutoff=0)
        fk_idx = phonetic_keys.index(fk[0]) if fk else None
        fk_disp = f"-> idx={fk_idx} {konkani_texts[fk_idx]}" if fk else "-> none"
        print(f"  FUZZY vs phonetic_key : {fk[1] if fk else 'n/a'} {fk_disp}")

        # Devanagari-vs-Devanagari comparison (what fuzzy_match does for
        # Devanagari queries: token_sort_ratio on the raw konkani texts)
        if any("\u0900" <= ch <= "\u097F" for ch in query):
            fd = process.extractOne(query, konkani_texts, scorer=fuzz.token_sort_ratio,
                                    score_cutoff=0)
            fd_idx = konkani_texts.index(fd[0]) if fd else None
            fd_disp = f"-> idx={fd_idx} {konkani_texts[fd_idx]}" if fd else "-> none"
            print(f"  FUZZY vs konkani_text : {fd[1] if fd else 'n/a'} {fd_disp}")

    print("\n" + "=" * 78)
    print("CASE (a): EXACT copies from dataset")
    print("=" * 78)
    for q in EXACT:
        diagnose("a-exact", q)

    print("\n" + "=" * 78)
    print("CASE (b): REPHRASED / spelling variants")
    print("=" * 78)
    for q in REPHRASED:
        diagnose("b-rephrased", q)

    print("\n" + "=" * 78)
    print("CASE (c): UNRELATED sentences (expect no good match)")
    print("=" * 78)
    for q in UNRELATED:
        diagnose("c-unrelated", q)

    print("\n" + "=" * 78)
    print("DIAGNOSTICS COMPLETE — no code was modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
