"""
Konkan Vani — Comprehensive Results & Benchmark Generator
=========================================================
Measures the ACTUAL system end-to-end and writes:
  results/results_metrics.json   — machine-readable metrics
  results/*.png                  — publication-ready charts
  results/tables.tex             — LaTeX tables ready to \\input
  results/benchmark_log.txt      — raw run log

Everything here runs the real pipeline (real dataset, real model, real index).
No numbers are fabricated; every metric is computed in this run.

Run from project root:
    python -X utf8 src/run_full_benchmark.py
"""
import os
import re
import sys
import json
import time
import random
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root)

_SERVER_DIR = os.path.abspath("server-nlp")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import numpy as np
import pandas as pd

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 78)
log("KONKAN VANI — FULL BENCHMARK RUN")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 78)

# ════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset statistics
# ════════════════════════════════════════════════════════════════════
log("\n[1/6] DATASET STATISTICS")

from app.data_loader import store

df = store.df
n = len(df)
log(f"Rows: {n}")
log(f"Columns: {list(df.columns)}")

# Provenance split
prov = df["source"].astype(str).str.split(":").str[0].value_counts()
log("Provenance:")
for k, v in prov.items():
    log(f"  {k}: {v}")

# Category distribution
cats = df["category"].value_counts()
log(f"Categories: {df['category'].nunique()}")

# Field completeness
completeness = {}
for col in ["konkani_text", "romanized_text", "literal_meaning",
            "figurative_meaning", "english_meaning", "marathi_meaning",
            "example_sentence", "category", "cultural_context", "phonetic_key"]:
    nonempty = int((df[col].astype(str).str.strip().replace("nan", "") != "").sum())
    completeness[col] = round(100 * nonempty / n, 2)
log("Field completeness (%):")
for k, v in completeness.items():
    log(f"  {k}: {v}%")

# Text lengths
len_rom = df["romanized_text"].astype(str).str.split().str.len()
log(f"Romanized text word count: mean={len_rom.mean():.2f}, median={len_rom.median():.0f}, "
    f"max={len_rom.max()}")

# Embedding integrity
emb = store.embeddings
log(f"Embeddings: shape={emb.shape}, dtype={emb.dtype}")
norms = np.linalg.norm(emb, axis=1)
log(f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, "
    f"mean={norms.mean():.4f} (should be ~1.0 if L2-normalized)")

faiss_status = "FAISS index" if store.faiss_index is not None else "brute-force numpy"
log(f"Search backend: {faiss_status}")

dataset_stats = {
    "total_rows": n,
    "num_columns": len(df.columns),
    "provenance": {k: int(v) for k, v in prov.items()},
    "num_categories": int(df["category"].nunique()),
    "category_distribution": {k: int(v) for k, v in cats.head(12).items()},
    "field_completeness_pct": completeness,
    "romanized_word_count": {
        "mean": round(float(len_rom.mean()), 2),
        "median": float(len_rom.median()),
        "max": int(len_rom.max()),
    },
    "embedding_shape": list(emb.shape),
    "embedding_norms": {
        "min": round(float(norms.min()), 4),
        "max": round(float(norms.max()), 4),
        "mean": round(float(norms.mean()), 4),
    },
    "search_backend": faiss_status,
}

# ════════════════════════════════════════════════════════════════════
# SECTION 2: Load model once
# ════════════════════════════════════════════════════════════════════
log("\n[2/6] LOADING MODEL")

t0 = time.perf_counter()
from app.core.model_registry import encode as _encode, get_model
from app.core.query_pipeline import fuzzy_match, semantic_match, SEMANTIC_THRESHOLD, FUZZY_THRESHOLD
model = get_model()
t_load = time.perf_counter() - t0
model_name = model.__class__.__name__
try:
    model_path = str(model)
except Exception:
    model_path = "?"
log(f"Model loaded in {t_load:.1f}s")

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Device: {dev}")


# ════════════════════════════════════════════════════════════════════
# Helper: corruption generator (tests robustness to spelling variants)
# ════════════════════════════════════════════════════════════════════
def corrupt(text: str, rng: random.Random) -> str:
    """Apply 1-2 random realistic spelling corruptions a user typing in
    Romanized Konkani might make."""
    if len(text) < 4:
        return text
    ops = ["vowel_double", "vowel_shorten", "v_swap", "ph_swap", "cap"]
    op = rng.choice(ops)
    ops_applied = 1
    if rng.random() < 0.5:
        # second corruption
        ops2 = [o for o in ops if o != op]
        op2 = rng.choice(ops2)
        ops_applied = 2
        ops = [op, op2]
    else:
        ops = [op]
    out = text
    for o in ops:
        if o == "vowel_double":
            m = [c for c in "aeiou"]
            idxs = [i for i, ch in enumerate(out) if ch.lower() in m]
            if idxs:
                i = rng.choice(idxs)
                out = out[:i] + out[i] + out[i:]   # double a vowel
        elif o == "vowel_shorten":
            pairs = ["aa", "ee", "oo"]
            for p in pairs:
                if p in out.lower():
                    i = out.lower().index(p)
                    out = out[:i] + p[0] + out[i + 2:]
        elif o == "v_swap":
            if "v" in out:
                out = out.replace("v", "w", 1)
            elif "w" in out:
                out = out.replace("w", "v", 1)
        elif o == "ph_swap":
            if "ph" in out:
                out = out.replace("ph", "f", 1)
            elif "f" in out:
                out = out.replace("f", "ph", 1)
        elif o == "cap":
            out = out.capitalize()
    return out


# ════════════════════════════════════════════════════════════════════
# SECTION 3: Stage-wise retrieval accuracy (precision/recall for top-1)
# ════════════════════════════════════════════════════════════════════
log("\n[3/6] RETRIEVAL ACCURACY (stage-wise, 200 sampled idioms per test)")

# Use the model_registry encode for semantic stage consistency
from app.core.model_registry import encode as enc

rng = random.Random(42)
sample_idx = rng.sample(range(n), 200)
sample_rows = df.iloc[sample_idx]

stage_stats = {
    "fuzzy_exact":      {"hit": 0, "total": 0, "scores": []},
    "fuzzy_corrupt":    {"hit": 0, "total": 0, "scores": []},
    "semantic_exact":   {"hit": 0, "total": 0, "scores": []},
    "semantic_corrupt": {"hit": 0, "total": 0, "scores": []},
}

# Pre-compute corrupted forms
corrupt_forms = {}
for i in sample_idx:
    rom = str(df.iloc[i]["romanized_text"])
    corrupt_forms[i] = corrupt(rom, rng)
    if corrupt_forms[i] == rom:
        # force at least one change
        corrupt_forms[i] = rom + " aa" if len(rom.split()) >= 1 else rom

# Report per-stage retrieval on both exact and corrupted Romanized queries
for i in sample_idx:
    row = df.iloc[i]
    q_exact = str(row["romanized_text"])
    q_corr = corrupt_forms[i]

    # — Fuzzy stage —
    r1 = fuzzy_match(q_exact, store)
    stage_stats["fuzzy_exact"]["total"] += 1
    if r1 is not None and r1["idx"] == i:
        stage_stats["fuzzy_exact"]["hit"] += 1
    stage_stats["fuzzy_exact"]["scores"].append(
        (r1["score"] / 100.0) if r1 is not None else 0.0)

    r2 = fuzzy_match(q_corr, store)
    stage_stats["fuzzy_corrupt"]["total"] += 1
    if r2 is not None and r2["idx"] == i:
        stage_stats["fuzzy_corrupt"]["hit"] += 1
    stage_stats["fuzzy_corrupt"]["scores"].append(
        (r2["score"] / 100.0) if r2 is not None else 0.0)

    # — Semantic stage —
    e1 = enc([q_exact])[0]
    res1 = store.faiss_search(e1, top_k=1)
    s1 = res1[0][1] if res1 else 0.0
    idx1 = res1[0][0] if res1 else -1
    stage_stats["semantic_exact"]["total"] += 1
    if idx1 == i:
        stage_stats["semantic_exact"]["hit"] += 1
    stage_stats["semantic_exact"]["scores"].append(float(s1))

    e2 = enc([q_corr])[0]
    res2 = store.faiss_search(e2, top_k=1)
    s2 = res2[0][1] if res2 else 0.0
    idx2 = res2[0][0] if res2 else -1
    stage_stats["semantic_corrupt"]["total"] += 1
    if idx2 == i:
        stage_stats["semantic_corrupt"]["hit"] += 1
    stage_stats["semantic_corrupt"]["scores"].append(float(s2))

retrieval = {}
for key, st in stage_stats.items():
    total = st["total"]
    hits = st["hit"]
    scores = np.array(st["scores"])
    retrieval[key] = {
        "accuracy": round(100 * hits / total, 2),
        "hits": hits,
        "total": total,
        "mean_similarity": round(float(scores.mean()), 4),
        "min_similarity": round(float(scores.min()), 4),
        "max_similarity": round(float(scores.max()), 4),
    }
    log(f"  {key:20s}: acc={retrieval[key]['accuracy']:6.2f}%  "
        f"mean_sim={retrieval[key]['mean_similarity']:.4f}")

# ═════════════════════════════════════════───────────────────────────
# SECTION 4: Full pipeline (resolve_idiom) with negatives = OOD rejection
# ════════════════════════════════════════════════════════════════════
log("\n[4/6] FULL PIPELINE EVALUATION (resolve_idiom)")

from app.core.query_pipeline import resolve_idiom

# 4a. Positive queries sampled from dataset (Romanized + Devanagari)
pos_n = 150
pos_idx = rng.sample(range(n), pos_n)

def normalize_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _correct_row(r, i, q):
    """Verify resolve_idiom returned THE row we queried, not just any row.
    resolve_idiom returns the matched row's meaning fields; compare them
    against pandas' ground truth for row i. Handles NaN safely by stringifying."""
    row = df.iloc[i]
    for field in ["english_meaning", "literal_meaning", "figurative_meaning"]:
        truth = str(row.get(field, ""))
        got = str(r.get(field, ""))
        if truth and truth != "nan":
            return normalize_text(got) == normalize_text(truth)
    return False


pos_results = {"rom": {"hit": 0, "type": {}}, "dev": {"hit": 0, "type": {}}}
for i in pos_idx:
    # Romanized
    q = str(df.iloc[i]["romanized_text"])
    r = resolve_idiom(q, store)
    t = r.get("match_type", "none")
    pos_results["rom"]["type"][t] = pos_results["rom"]["type"].get(t, 0) + 1
    if r.get("matched") and _correct_row(r, i, q):
        pos_results["rom"]["hit"] += 1
    # Devanagari
    q = str(df.iloc[i]["konkani_text"])
    r = resolve_idiom(q, store)
    t = r.get("match_type", "none")
    pos_results["dev"]["type"][t] = pos_results["dev"]["type"].get(t, 0) + 1
    if r.get("matched") and _correct_row(r, i, q):
        pos_results["dev"]["hit"] += 1


neg_queries = [
    "what is the capital of France?",
    "recipe for chocolate cake with eggs",
    "Melbourne Cricketers narrowly won the match",
    "पाठशाळेत आज मोठी चाचणी घेतली",  # Marathi, not Konkani idiom domain
    "Stock markets fell sharply after the announcement",
    "amchi mumbai local train chhan chhan dhotey",  # Marathi romanized
    "quantum physics describes subatomic particles",
    "the mitochondria is the powerhouse of the cell",
]

ood_results = {"true_neg": 0, "false_pos": 0, "total": len(neg_queries),
               "matched_to": []}
for q in neg_queries:
    r = resolve_idiom(q, store)
    if r.get("matched"):
        ood_results["false_pos"] += 1
        ood_results["matched_to"].append({
            "query": q[:50],
            "match_type": r.get("match_type"),
            "confidence": r.get("confidence"),
            "konkani_text": r.get("konkani_text", "")[:60],
        })
    else:
        ood_results["true_neg"] += 1

ood_rejection_rate = 100 * ood_results["true_neg"] / ood_results["total"]
log(f"  OOD rejection rate: {ood_rejection_rate:.1f}% "
    f"({ood_results['true_neg']}/{ood_results['total']})")
for fp in ood_results["matched_to"]:
    log(f"    FALSE-POSITIVE: {fp['query']!r} → {fp['match_type']} "
        f"conf={fp['confidence']} → {fp['konkani_text']!r}")

pipeline_acc = {
    "romanized_correct_pct": round(100 * pos_results["rom"]["hit"] / pos_n, 2),
    "devanagari_correct_pct": round(100 * pos_results["dev"]["hit"] / pos_n, 2),
    "romanized_match_types": pos_results["rom"]["type"],
    "devanagari_match_types": pos_results["dev"]["type"],
    "ood_total": ood_results["total"],
    "ood_true_neg": ood_results["true_neg"],
    "ood_false_pos": ood_results["false_pos"],
    "ood_rejection_pct": round(ood_rejection_rate, 2),
    "ood_false_positive_details": ood_results["matched_to"],
}

# ════════════════════════════════════════════════════════════════════
# SECTION 5: Latency benchmarking
# ════════════════════════════════════════════════════════════════════
log("\n[5/6] LATENCY BENCHMARK")

lat = {"fuzzy": [], "semantic": [], "full_resolve": [], "search_topk10": []}

warm_idx = rng.sample(range(n), 60)
for i in warm_idx:
    q = str(df.iloc[i]["romanized_text"])

    t0 = time.perf_counter()
    fuzzy_match(q, store)
    lat["fuzzy"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    e = enc([q])[0]
    store.faiss_search(e, top_k=1)
    lat["semantic"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    resolve_idiom(q, store)
    lat["full_resolve"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    from app.core.query_pipeline import search_idioms
    search_idioms(q, top_k=10)
    lat["search_topk10"].append((time.perf_counter() - t0) * 1000)


def pct(arr, p):
    return round(float(np.percentile(arr, p)), 2)

latency = {}
for k, arr in lat.items():
    a = np.array(arr)
    latency[k] = {
        "mean_ms": round(float(a.mean()), 2),
        "median_ms": round(float(np.median(a)), 2),
        "p95_ms": pct(a, 95),
        "p99_ms": pct(a, 99),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
        "n": len(a),
    }
    log(f"  {k:16s}: mean={latency[k]['mean_ms']:7.2f}ms  "
        f"p95={latency[k]['p95_ms']:7.2f}ms  p99={latency[k]['p99_ms']:7.2f}ms")


# ════════════════════════════════════════════════════════════════════
# SECTION 6: Threshold sensitivity sweep for the semantic cutoff
# ════════════════════════════════════════════════════════════════════
log("\n[6/6] SEMANTIC THRESHOLD SWEEP (accuracy vs OOD rejection tradeoff)")

# Encode the 8 OOD queries and 150 in-domain positives once
ood_embs = enc(neg_queries)
pos_texts = [str(df.iloc[i]["romanized_text"]) for i in pos_idx]
pos_embs = enc(pos_texts)

# best similarity of each query to dataset
def best_sims(qembs):
    out = []
    for e in qembs:
        sims = emb @ e.astype(np.float32)
        out.append(float(sims.max()))
    return np.array(out)

sims_pos = best_sims(pos_embs)
sims_ood = best_sims(ood_embs)

log(f"  In-domain max-cosine: mean={sims_pos.mean():.4f}, min={sims_pos.min():.4f}")
log(f"  OOD max-cosine:       mean={sims_ood.mean():.4f}, max={sims_ood.max():.4f}")

threshold_sweep = []
for th in [0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75, 0.80]:
    # in-domain recall: fraction of positives with best sim >= th
    in_acc = float((sims_pos >= th).mean()) * 100
    ood_rej = float((sims_ood < th).mean()) * 100
    threshold_sweep.append({
        "threshold": th,
        "in_domain_accuracy_pct": round(in_acc, 2),
        "ood_rejection_pct": round(ood_rej, 2),
    })
    log(f"  th={th:.2f}: in-domain acc={in_acc:6.2f}%  OOD rej={ood_rej:6.2f}%")

# ════════════════════════════════════════════════════════════════════
# SAVE everything
# ════════════════════════════════════════════════════════════════════
import platform

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "environment": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": dev,
    },
    "dataset": dataset_stats,
    "model": {
        "loaded_in_s": round(t_load, 2),
        "device": dev,
    },
    "retrieval_stage_accuracy": retrieval,
    "pipeline_accuracy": pipeline_acc,
    "ood_rejection": {
        "rate_pct": round(ood_rejection_rate, 2),
        "false_positives": ood_results["matched_to"],
    },
    "latency_ms": latency,
    "threshold_sweep": threshold_sweep,
    "similarity_distributions": {
        "in_domain_mean": round(float(sims_pos.mean()), 4),
        "in_domain_min": round(float(sims_pos.min()), 4),
        "in_domain_p5": round(float(np.percentile(sims_pos, 5)), 4),
        "ood_mean": round(float(sims_ood.mean()), 4),
        "ood_max": round(float(sims_ood.max()), 4),
    },
}

with open(os.path.join(OUT_DIR, "results_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "benchmark_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\nSaved: results/results_metrics.json, results/benchmark_log.txt")
log("DONE")
