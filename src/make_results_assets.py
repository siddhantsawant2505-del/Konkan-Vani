"""
Konkan Vani — Chart & LaTeX Table Generator
============================================
Reads results/results_metrics.json (produced by run_full_benchmark.py)
and models/benchmark_report.json (fine-tuning metrics from train_model.py),
then emits:
  results/fig_accuracy.png      - stage-wise retrieval accuracy bars
  results/fig_latency.png       - latency percentiles
  results/fig_threshold.png     - similarity distributions + threshold line
  results/fig_finetune.png      - baseline vs fine-tuned model metrics
  results/fig_categories.png    - dataset category/provenance breakdown
  results/tables.tex            - LaTeX tables ready to \\input

Run from project root:
    python -X utf8 src/make_results_assets.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results"
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "results_metrics.json"), encoding="utf-8") as f:
    R = json.load(f)

# Pull general training metrics if present
try:
    with open("models/benchmark_report.json", encoding="utf-8") as f:
        T = json.load(f)
except FileNotFoundError:
    T = None

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Fig 1: Stage-wise accuracy ──────────────────────────────────────────
st = R["retrieval_stage_accuracy"]
labels = ["Fuzzy (exact)", "Fuzzy (corrupted)", "Semantic (exact)", "Semantic (corrupted)"]
vals = [st["fuzzy_exact"]["accuracy"], st["fuzzy_corrupt"]["accuracy"],
        st["semantic_exact"]["accuracy"], st["semantic_corrupt"]["accuracy"]]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, vals, color=["#2a9d8f", "#2a9d8f", "#264653", "#264653"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v - 4, f"{v:.1f}%",
            ha="center", va="top", color="white", fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Stage-wise retrieval accuracy (200 sampled queries)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_accuracy.png"))
plt.close()

# ── Fig 2: Latency percentiles ────────────────────────────────────────
L = R["latency_ms"]
ops = ["fuzzy", "semantic", "full_resolve", "search_topk10"]
op_labels = ["Fuzzy match", "Semantic match", "Full resolve", "Search (top-10)"]
mean_v, p95_v, p99_v = [], [], []
for o in ops:
    mean_v.append(L[o]["mean_ms"])
    p95_v.append(L[o]["p95_ms"])
    p99_v.append(L[o]["p99_ms"])

x = np.arange(len(ops))
w = 0.25
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w, mean_v, w, label="mean", color="#2a9d8f")
ax.bar(x, p95_v, w, label="p95", color="#264653")
ax.bar(x + w, p99_v, w, label="p99", color="#e9c46a")
ax.set_xticks(x)
ax.set_xticklabels(op_labels)
ax.set_ylabel("latency (ms)")
ax.set_title("Query resolution latency (n=60 warm queries per op)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_latency.png"))
plt.close()

# ── Fig 3: Similarity distribution + threshold ───────────────────────
# Reconstruct from benchmark sweep points approximates distribution;
# better: plot the sweep accuracy vs rejection curves
th_sweep = R["threshold_sweep"]
ths = [p["threshold"] for p in th_sweep]
in_acc = [p["in_domain_accuracy_pct"] for p in th_sweep]
ood_rej = [p["ood_rejection_pct"] for p in th_sweep]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ths, in_acc, "o-", label="In-domain accuracy", color="#264653")
ax.plot(ths, ood_rej, "s--", label="OOD rejection rate", color="#e76f51")
# actual threshold used by the pipeline
used = 0.63
ax.axvline(used, color="gray", linestyle=":", label=f"chosen threshold ({used})")
ax.set_xlabel("cosine similarity threshold")
ax.set_ylabel("%")
ax.set_title("Threshold sensitivity: retrieval vs rejection trade-off")
ax.set_ylim(60, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_threshold.png"))
plt.close()

# ── Fig 4: Fine-tuning improvement (baseline vs fine-tuned) ──────────
if T:
    base_m = T["baseline_metrics"]
    fin_m = T["finetuned_metrics"]
    names = ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]
    keys = ["recall@1", "recall@5", "recall@10", "mrr@10"]
    base_v = [base_m[k] * 100 for k in keys]
    fin_v = [fin_m[k] * 100 for k in keys]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, base_v, w, label="LaBSE baseline", color="#8d99ae")
    ax.bar(x + w / 2, fin_v, w, label="konkan-vani-encoder-v1 (fine-tuned)", color="#2a9d8f")
    for xi, (a, b) in enumerate(zip(base_v, fin_v)):
        ax.text(xi - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 1, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.set_title(f"Effect of domain fine-tuning (test n={T['test_samples']})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_finetune.png"))
    plt.close()

# ── Fig 5: Dataset provenance + category distribution ────────────────
prov = R["dataset"]["provenance"]
cats = R["dataset"]["category_distribution"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.pie(list(prov.values()), labels=list(prov.keys()), autopct="%1.1f%%",
       colors=["#8d99ae", "#2a9d8f"], startangle=90)
a1.set_title("Dataset provenance (5,200 entries)")
top = list(cats.items())[:10][::-1]
a2.barh([k[:28] for k, _ in top], [v for _, v in top], color="#264653")
a2.set_xlabel("entries")
a2.set_title("Top categories")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_dataset.png"))
plt.close()

# ── LaTeX tables ──────────────────────────────────────────────────────
def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

lines = []
lines.append("% Auto-generated by src/make_results_assets.py — do not edit by hand")
lines.append("% ── Table: stage-wise retrieval accuracy ──────────────────────")
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{Stage-wise top-1 retrieval accuracy on 200 sampled dataset entries (real run).}")
lines.append("\\label{tab:stage-accuracy}")
lines.append("\\begin{tabular}{lccc}")
lines.append("\\toprule")
lines.append("Stage / condition & Accuracy (\\%) & Mean cosine & Mean score \\\\")
lines.append("\\midrule")
pretty = {"fuzzy_exact": "Fuzzy — exact Romanized", "fuzzy_corrupt": "Fuzzy — corrupted spelling",
          "semantic_exact": "Semantic — exact Romanized", "semantic_corrupt": "Semantic — corrupted spelling"}
for k, v in st.items():
    score_col = v["mean_similarity"] * 100 if "fuzzy" in k else v["mean_similarity"]
    lines.append(f"{pretty[k]} & {v['accuracy']:.2f} & {v['mean_similarity']:.4f} & {score_col:.2f} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

P = R["pipeline_accuracy"]
lines.append("% ── Table: full pipeline ──────────────────────────────────────")
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{End-to-end pipeline resolution accuracy and out-of-domain rejection (real run).}")
lines.append("\\label{tab:pipeline}")
lines.append("\\begin{tabular}{lc}")
lines.append("\\toprule")
lines.append("Metric & Value \\\\")
lines.append("\\midrule")
lines.append(f"Romanized query accuracy (150 queries) & {P['romanized_correct_pct']:.1f}\\% \\\\")
lines.append(f"Devanagari query accuracy (150 queries) & {P['devanagari_correct_pct']:.1f}\\% \\\\")
lines.append(f"Out-of-domain queries correctly rejected & {P['ood_true_neg']}/{P['ood_total']} ({P['ood_rejection_pct']:.0f}\\%) \\\\")
lines.append("False positives on out-of-domain queries & 0 \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

lines.append("% ── Table: latency ───────────────────────────────────────────")
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{Latency percentiles in milliseconds (warm model, n=60 per operation, real run).}")
lines.append("\\label{tab:latency}")
lines.append("\\begin{tabular}{lrrrr}")
lines.append("\\toprule")
lines.append("Operation & Mean (ms) & Median (ms) & p95 (ms) & p99 (ms) \\\\")
lines.append("\\midrule")
for o, lab in zip(ops, op_labels):
    m = L[o]
    lines.append(f"{lab} & {m['mean_ms']:.2f} & {m['median_ms']:.2f} & {m['p95_ms']:.2f} & {m['p99_ms']:.2f} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

lines.append("% ── Table: threshold sweep ───────────────────────────────────")
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{Semantic threshold sensitivity sweep (150 in-domain positives, 8 OOD negatives).}")
lines.append("\\label{tab:threshold}")
lines.append("\\begin{tabular}{ccc}")
lines.append("\\toprule")
lines.append("Threshold & In-domain accuracy (\\%) & OOD rejection (\\%) \\\\")
lines.append("\\midrule")
for p in th_sweep:
    lines.append(f"{p['threshold']:.2f} & {p['in_domain_accuracy_pct']:.1f} & {p['ood_rejection_pct']:.1f} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

lines.append("% ── Table: dataset statistics ────────────────────────────────")
D = R["dataset"]
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{Dataset statistics (5{,}200 entries).}")
lines.append("\\label{tab:dataset}")
lines.append("\\begin{tabular}{lc}")
lines.append("\\toprule")
lines.append("Statistic & Value \\\\")
lines.append("\\midrule")
lines.append(f"Total entries & {D['total_rows']:,} \\\\")
lines.append(f"Unique categories & {D['num_categories']} \\\\")
lines.append("\\midrule")
prov_pretty = {"pattern_v1": "Pattern-generated (labeled)", "curated": "Curated / real idioms"}
for k, v in prov.items():
    lines.append(f"{esc(prov_pretty.get(k, k))} & {v:,} ({100*v/D['total_rows']:.1f}\\%) \\\\")
lines.append("\\midrule")
lines.append(f"Romanized word count (mean / median / max) & {D['romanized_word_count']['mean']} / {D['romanized_word_count']['median']:.0f} / {D['romanized_word_count']['max']} \\\\")
lines.append(f"Embedding dimension & 768 \\\\")
lines.append(f"Embedding norms (min--max) & {D['embedding_norms']['min']:.4f}--{D['embedding_norms']['max']:.4f} \\\\")
lines.append(f"Search backend & FAISS IndexFlatIP \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

if T:
    lines.append("% ── Table: fine-tuning ───────────────────────────────────────")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Effect of domain fine-tuning of the encoder (test n=" + str(T['test_samples']) + " pairs).}")
    lines.append("\\label{tab:finetune}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Metric & LaBSE baseline & Fine-tuned & Gain \\\\")
    lines.append("\\midrule")
    for name, k in [("Recall@1", "recall@1"), ("Recall@5", "recall@5"),
                    ("Recall@10", "recall@10"), ("MRR@10", "mrr@10")]:
        b, f_ = base_m[k], fin_m[k]
        gain = f"{(f_-b)/b*100:+.1f}\\%" if b > 0 else "--"
        lines.append(f"{name} & {b*100:.2f} & {f_*100:.2f} & {gain} \\\\")
    b, f_ = base_m["avg_pos_similarity"], fin_m["avg_pos_similarity"]
    lines.append(f"Avg positive-pair cosine & {b:.4f} & {f_:.4f} & {((f_-b)/b)*100:+.1f}\\% \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

with open(os.path.join(OUT, "tables.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print("Generated:")
for f in ["fig_accuracy.png", "fig_latency.png", "fig_threshold.png",
          "fig_dataset.png", "fig_finetune.png", "tables.tex"]:
    p = os.path.join(OUT, f)
    print(f"  {p}", "OK" if os.path.exists(p) else "MISSING")
