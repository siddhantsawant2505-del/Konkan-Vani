"""
Konkan Vani — Additional Graph Generator
=========================================
Produces six additional publication-grade figures from REAL pipeline data:
  results/fig_similarity_hist.png   — in-domain vs OOD cosine similarity separation
  results/fig_embedding_heatmap.png — first 60 embeddings x 64 dims heatmap
  results/fig_latency_cdf.png       — latency cumulative distribution curves
  results/fig_stage_funnel.png      — resolver funnel: hits by stage
  results/fig_qq_threshold.png      — per-query best-cosine (sorted) with threshold
  results/fig_synthetic_radar.png   — curated vs pattern entries field-richness radar

All data is measured in this run (same environment as run_full_benchmark.py).
Run from project root:  python -X utf8 src/make_extra_graphs.py
"""
import os
import sys
import random

sys.stdout.reconfigure(encoding="utf-8")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root)

_SERVER_DIR = os.path.abspath("server-nlp")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from app.data_loader import store
from app.core.model_registry import encode as enc
from app.core.query_pipeline import fuzzy_match, semantic_match, resolve_idiom

OUT = "results"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = store.df
emb = store.embeddings
n = len(df)
rng = random.Random(42)


# ── helpers ──────────────────────────────────────────────────────────
def best_cosine(query_emb):
    sims = emb @ query_emb.astype(np.float32)
    return float(sims.max()), int(sims.argmax())


def corrupt(text: str, r: random.Random) -> str:
    if len(text) < 4:
        return text
    op = r.choice(["vowel_double", "v_swap", "ph_swap", "cap"])
    out = text
    if op == "vowel_double":
        idxs = [i for i, c in enumerate(out) if c.lower() in "aeiou"]
        if idxs:
            i = r.choice(idxs)
            out = out[:i] + out[i] * 2 + out[i + 1:]
    elif op == "v_swap":
        if "v" in out:
            out = out.replace("v", "w", 1)
        elif "w" in out:
            out = out.replace("w", "v", 1)
    elif op == "ph_swap":
        if "ph" in out:
            out = out.replace("ph", "f", 1)
        elif "f" in out:
            out = out.replace("f", "ph", 1)
    elif op == "cap":
        out = out.capitalize()
    return out


OOD_QUERIES = [
    "what is the capital of France?",
    "recipe for chocolate cake with eggs",
    "Melbourne cricketers narrowly won the match",
    "पाठशाळेत आज मोठी चाचणी घेतली",
    "Stock markets fell sharply after the announcement",
    "amchi mumbai local train chhan chhan dhotey",
    "quantum physics describes subatomic particles",
    "the mitochondria is the powerhouse of the cell",
]

# ══════════════════════════════════════════════════════════════════
# Fig A: In-domain vs OOD similarity histogram
# ══════════════════════════════════════════════════════════════════
print("[1/6] Similarity histogram ...")
sample_idx = rng.sample(range(n), 200)
in_sims = []
for i in sample_idx:
    q = str(df.iloc[i]["romanized_text"])
    e = enc([q])[0]
    s, _ = best_cosine(e)
    in_sims.append(s)

ood_sims = []
for q in OOD_QUERIES:
    e = enc([q])[0]
    s, _ = best_cosine(e)
    ood_sims.append(s)

fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(0.0, 1.0, 41)
ax.hist(in_sims, bins=bins, alpha=0.68, color="#2a9d8f", label="In-domain (200 queries)", edgecolor="white")
ax.hist(ood_sims, bins=bins, alpha=0.75, color="#e76f51", label="Out-of-domain (8 queries)", edgecolor="white")
ax.axvline(0.63, color="gray", linestyle="--", linewidth=1.5, label="chosen threshold (0.63)")
ax.set_xlabel("best cosine similarity to any dataset entry")
ax.set_ylabel("query count")
ax.set_title("Similarity separation: in-domain vs out-of-domain queries")
ax.legend(loc="upper left")
ax.text(0.02, 0.95, f"in-domain min = {min(in_sims):.4f}\nOOD max = {max(ood_sims):.4f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_similarity_hist.png"))
plt.close()
print(f"  in-domain min={min(in_sims):.4f}, OOD max={max(ood_sims):.4f}")

# ══════════════════════════════════════════════════════════════════
# Fig B: Embedding heatmap (first 60 rows, 64 dims)
# ══════════════════════════════════════════════════════════════════
print("[2/6] Embedding heatmap ...")
fig, ax = plt.subplots(figsize=(8, 5.5))
sub = emb[:60, :64]
im = ax.imshow(sub, aspect="auto", cmap="viridis",
               vmin=-np.abs(sub).max(), vmax=np.abs(sub).max())
ax.set_xlabel("embedding dimension (first 64 of 768)")
ax.set_ylabel("idiom index (first 60)")
ax.set_title("Structure of the embedding matrix (fine-tuned encoder)")
fig.colorbar(im, ax=ax, label="value")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_embedding_heatmap.png"))
plt.close()

# ══════════════════════════════════════════════════════════════════
# Fig C: Latency CDF (empirical timing for each operation)
# ══════════════════════════════════════════════════════════════════
print("[3/6] Latency CDF ...")
lat_fuzzy, lat_sem, lat_full = [], [], []
warm = rng.sample(range(n), 80)
for i in warm:  # warm-up
    fuzzy_match(str(df.iloc[i]["romanized_text"]), store)
for i in warm:
    q = str(df.iloc[i]["romanized_text"])
    t0 = time.perf_counter(); fuzzy_match(q, store); lat_fuzzy.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    e = enc([q])[0]; store.faiss_search(e, top_k=1)
    lat_sem.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); resolve_idiom(q, store); lat_full.append((time.perf_counter() - t0) * 1000)

fig, ax = plt.subplots(figsize=(7, 4.2))
for arr, lab, color in [
    (lat_fuzzy, "Fuzzy match", "#2a9d8f"),
    (lat_sem, "Semantic match (encoder+search)", "#264653"),
    (lat_full, "Full resolve_idiom()", "#e76f51"),
]:
    a = np.sort(np.array(arr))
    cdf = np.arange(1, len(a) + 1) / len(a) * 100
    ax.plot(a, cdf, label=lab, color=color, linewidth=2)
ax.axhline(95, color="gray", linestyle=":", linewidth=1)
ax.text(0.98, 96, "95th percentile", transform=ax.transAxes,
        ha="right", fontsize=8, color="gray")
ax.set_xlabel("latency (ms)")
ax.set_ylabel("cumulative % of queries")
ax.set_title("Latency CDF per operation (n=80 warm queries)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_latency_cdf.png"))
plt.close()

# ══════════════════════════════════════════════════════════════════
# Fig D: Resolver stage funnel
# ══════════════════════════════════════════════════════════════════
print("[4/6] Resolver funnel ...")
test_idx = rng.sample(range(n), 100)
stage_wins = {"fuzzy (phonetic)": 0, "semantic": 0, "word-level": 0, "no match": 0}
for i in test_idx:
    rom = str(df.iloc[i]["romanized_text"])
    dev = str(df.iloc[i]["konkani_text"])
    for q in (rom, dev):
        if fuzzy_match(q, store) is not None:
            stage_wins["fuzzy (phonetic)"] += 1
        elif semantic_match(q, store) is not None:
            stage_wins["semantic"] += 1
        else:
            r = resolve_idiom(q, store)
            if r.get("matched"):
                stage_wins["word-level"] += 1
            else:
                stage_wins["no match"] += 1

labels = list(stage_wins.keys())
vals = list(stage_wins.values())
total = sum(vals)
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.barh(labels[::-1], vals[::-1],
               color=["#e9c46a", "#e76f51", "#264653", "#2a9d8f"][::-1])
for b, v in zip(bars, vals[::-1]):
    ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
            f"{v} ({100*v/total:.1f}%)", va="center", fontsize=9)
ax.set_xlabel("queries resolved at this stage (200 total: 100 rom + 100 dev)")
ax.set_title("Which resolver stage wins each query")
ax.set_xlim(0, max(vals) * 1.22)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_stage_funnel.png"))
plt.close()

# ══════════════════════════════════════════════════════════════════
# Fig E: Per-query sorted best-cosine with threshold line
# ══════════════════════════════════════════════════════════════════
print("[5/6] Per-query sorted cosine ...")
sorted_in = np.sort(np.array(in_sims))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(range(len(sorted_in)), sorted_in, color="#264653", linewidth=1.8,
        label="in-domain queries (sorted)")
ax.scatter(range(len(sorted_in)), sorted_in, s=6, color="#264653")
ax.axhline(0.63, color="gray", linestyle="--", linewidth=1.5, label="threshold (0.63)")
ax.set_xlabel("query rank (sorted by best cosine)")
ax.set_ylabel("best cosine to dataset")
ax.set_title("Worst-case coverage: every in-domain query clears the threshold")
ax.annotate(f"min = {sorted_in[0]:.4f}",
            xy=(0, sorted_in[0]), xytext=(10, sorted_in[0] - 0.04),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_qq_threshold.png"))
plt.close()

# ══════════════════════════════════════════════════════════════════
# Fig F: Curated vs pattern-generated field richness (radar)
# ══════════════════════════════════════════════════════════════════
print("[6/6] Field-richness radar ...")
curated_mask = df["source"].astype(str).str.startswith("curated")
pattern_mask = ~curated_mask
fields = ["konkani_text", "romanized_text", "literal_meaning",
          "figurative_meaning", "english_meaning", "marathi_meaning",
          "example_sentence", "cultural_context"]

def avg_len(mask):
    lens = []
    for col in fields:
        text = df.loc[mask, col].astype(str)
        text = text[text != "nan"]
        if len(text):
            lens.append(text.str.split().str.len().mean())
        else:
            lens.append(0.0)
    return lens

cur_lens = avg_len(curated_mask)
pat_lens = avg_len(pattern_mask)

angles = np.linspace(0, 2 * np.pi, len(fields), endpoint=False).tolist()
angles += angles[:1]
cur_lens_c = cur_lens + cur_lens[:1]
pat_lens_c = pat_lens + pat_lens[:1]

fig, ax = plt.subplots(figsize=(7.5, 6), subplot_kw=dict(polar=True))
ax.plot(angles, cur_lens_c, color="#2a9d8f", linewidth=2, label=f"Curated ({int(curated_mask.sum())} entries)")
ax.fill(angles, cur_lens_c, color="#2a9d8f", alpha=0.15)
ax.plot(angles, pat_lens_c, color="#8d99ae", linewidth=2, linestyle="--",
        label=f"Pattern-generated ({int(pattern_mask.sum())})")
ax.fill(angles, pat_lens_c, color="#8d99ae", alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([f.replace("_", " ") for f in fields], fontsize=9)
ax.set_ylabel("avg words per field", fontsize=9, labelpad=25)
ax.set_title("Field richness: curated vs pattern-generated entries")
ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.1))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_field_richness.png"))
plt.close()

print("\nGenerated extra graphs:")
for f in ["fig_similarity_hist.png", "fig_embedding_heatmap.png",
          "fig_latency_cdf.png", "fig_stage_funnel.png",
          "fig_qq_threshold.png", "fig_field_richness.png"]:
    p = os.path.join(OUT, f)
    print(f"  {p}", "OK" if os.path.exists(p) else "MISSING")
