"""
Konkan Vani — Lexicon Embedding Calibration
===========================================
The compositional fallback glosses unknown Devanagari words with a
*cross-lingual* nearest-neighbour guess: the word's LaBSE vector is compared
with the ENGLISH glosses already in ``data/lexicon/konkani_lexicon.csv``.
LaBSE is trained to align languages, so a Konkani word sits near its own
English gloss — but only above some similarity is that trustworthy.

This script measures that boundary on the real lexicon:

  * the *content* gloss pool (grammar words and long paraphrases excluded,
    duplicate glosses collapsed) — this is what the runtime searches;
  * Recall@1 / Recall@5 and the true-vs-best-wrong similarity gap;
  * a threshold sweep giving precision and coverage per cutoff;
  * the same numbers for romanised input, which is why the runtime only uses
    the embedding stage on Devanagari (roman transliteration is too noisy).

Writes ``results/lexicon_calibration.json``.

Usage:  python -X utf8 src/calibrate_lexicon.py
"""

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_SERVER = os.path.abspath("server-nlp")
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)

from app.core.lexicon import LEXICON, EMBED_THRESHOLD   # noqa: E402
from app.core.model_registry import encode              # noqa: E402

OUT_PATH = os.path.join("results", "lexicon_calibration.json")
SWEEP = [0.50, 0.55, 0.60, 0.63, 0.66, 0.70, 0.75, 0.80]
SKIP_POS = {"connective", "auxiliary", "negation", "quantifier", "adverb", "pronoun"}


def content_pool(entries):
    """The runtime's gloss pool: content words, short glosses, deduplicated."""
    pool, seen = [], set()
    for e in entries:
        if e["pos"] in SKIP_POS:
            continue
        if len(e["english"].split()) > 3:
            continue
        if e["english"] in seen:
            continue
        seen.add(e["english"])
        pool.append(e)
    return pool


def evaluate(name, query_embs, gloss_embs, gloss_of, entries):
    sims = query_embs @ gloss_embs.T
    rows = [i for i, e in enumerate(entries) if e["english"] in gloss_of]
    true_idx = np.array([gloss_of[entries[i]["english"]] for i in rows])

    true_sim = sims[rows, true_idx]
    masked = sims[rows].copy()
    masked[np.arange(len(rows)), true_idx] = -1
    wrong_best = masked.max(axis=1)

    order = np.argsort(-sims[rows], axis=1)
    rank = np.array([
        int(np.where(order[i] == true_idx[i])[0][0]) + 1 for i in range(len(rows))
    ])

    print(f"\n[{name}] n={len(rows)}")
    print(f"  recall@1 {float((rank <= 1).mean()):.3f}   "
          f"recall@3 {float((rank <= 3).mean()):.3f}   "
          f"recall@5 {float((rank <= 5).mean()):.3f}")
    print(f"  true gloss cos : mean {true_sim.mean():.3f}  "
          f"p25 {np.percentile(true_sim, 25):.3f}  median {np.median(true_sim):.3f}")
    print(f"  best wrong cos : mean {wrong_best.mean():.3f}  "
          f"p95 {np.percentile(wrong_best, 95):.3f}")

    sweep = []
    for t in SWEEP:
        accept = true_sim >= t
        n = int(accept.sum())
        prec = float((true_sim[accept] > wrong_best[accept]).mean()) if n else None
        sweep.append({
            "threshold": t,
            "accepted": n,
            "precision": None if prec is None else round(prec, 4),
            "coverage": round(float(accept.mean()), 4),
        })
        print(f"  t={t:.2f}  accepted {n:>3}  coverage {accept.mean():.3f}  "
              f"precision {'-' if prec is None else f'{prec:.3f}'}")

    return {
        "n": len(rows),
        "recall_at_1": round(float((rank <= 1).mean()), 4),
        "recall_at_3": round(float((rank <= 3).mean()), 4),
        "recall_at_5": round(float((rank <= 5).mean()), 4),
        "true_gloss_cosine": {
            "mean": round(float(true_sim.mean()), 4),
            "p25": round(float(np.percentile(true_sim, 25)), 4),
            "median": round(float(np.median(true_sim)), 4),
        },
        "best_wrong_cosine": {
            "mean": round(float(wrong_best.mean()), 4),
            "p95": round(float(np.percentile(wrong_best, 95)), 4),
        },
        "sweep": sweep,
    }


def main():
    entries = LEXICON.entries
    pool = content_pool(entries)
    print(f"lexicon entries : {len(entries)}")
    print(f"content pool    : {len(pool)} unique English glosses")

    gloss_embs = np.asarray(encode([e["english"] for e in pool]), dtype=np.float32)
    gloss_of = {e["english"]: i for i, e in enumerate(pool)}

    dev = evaluate("devanagari headwords",
                   np.asarray(encode([e["devanagari"] for e in entries]), dtype=np.float32),
                   gloss_embs, gloss_of, entries)
    rom = evaluate("roman transliterations",
                   np.asarray(encode([e["roman"] for e in entries]), dtype=np.float32),
                   gloss_embs, gloss_of, entries)

    at_shipped = next(
        (s for s in dev["sweep"] if abs(s["threshold"] - EMBED_THRESHOLD) < 1e-9), None
    )

    payload = {
        "lexicon": "data/lexicon/konkani_lexicon.csv",
        "entries": len(entries),
        "content_gloss_pool": len(pool),
        "model": "models/konkani-vani-encoder-v1 (fine-tuned LaBSE)",
        "devanagari": dev,
        "roman": rom,
        "shipped_threshold": EMBED_THRESHOLD,
        "shipped_threshold_metrics": at_shipped,
        "decision": (
            "The embedding stage runs on Devanagari tokens only, at the shipped "
            "threshold; romanised tokens are handled by exact/stem/phonetic/fuzzy "
            "lookup instead, because Recall@1 on roman input is an order of "
            "magnitude worse. Approximate glosses are always rendered with "
            "'≈' and never presented as dictionary entries."
        ),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {OUT_PATH}")
    if at_shipped:
        print(f"shipped threshold {EMBED_THRESHOLD}: "
              f"precision {at_shipped['precision']}, coverage {at_shipped['coverage']}")


if __name__ == "__main__":
    main()
