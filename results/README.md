# Konkan Vani — Benchmark Results

Everything in this folder was produced by running the **actual system** (real
dataset `data/processed/idioms_with_phonetic_keys.csv`, fine-tuned encoder
`models/konkan-vani-encoder-v1`, FAISS index) on **2026-10-12**, Windows 11,
CUDA. Nothing is fabricated; regenerate any file with the commands below.

## Files

| File | What it is | Regenerate with |
|---|---|---|
| `results_metrics.json` | Machine-readable metrics from the full benchmark | `python -X utf8 src/run_full_benchmark.py` |
| `benchmark_log.txt` | Raw console log of the benchmark run | (same run) |
| `fig_accuracy.png` | Stage-wise retrieval accuracy chart | `python -X utf8 src/make_results_assets.py` |
| `fig_latency.png` | Latency percentile chart | (same script) |
| `fig_threshold.png` | Threshold sensitivity chart | (same script) |
| `fig_finetune.png` | Baseline vs fine-tuned encoder chart | (same script) |
| `fig_dataset.png` | Provenance & category chart | (same script) |
| `fig_similarity_hist.png` | In-domain vs OOD similarity histogram | `python -X utf8 src/make_extra_graphs.py` |
| `fig_embedding_heatmap.png` | Embedding matrix heatmap | (same script) |
| `fig_latency_cdf.png` | Latency CDF curves | (same script) |
| `fig_stage_funnel.png` | Resolver stage funnel | (same script) |
| `fig_qq_threshold.png` | Per-query sorted cosine vs threshold | (same script) |
| `fig_field_richness.png` | Curated vs pattern field radar | (same script) |
| `tables.tex` | LaTeX tables of every metric | (same script) |
| `chapter5_results.tex` | Full LaTeX "Testing and Results" chapter, ready to `\input` | written by hand from the metrics |

## Headline numbers (all measured, not estimated)

- **Stage-wise top-1 accuracy** (200 sampled entries): fuzzy 100% exact /
  100% corrupted spelling; semantic 100% exact / 99% corrupted.
- **End-to-end pipeline** (`resolve_idiom`, 150 Romanized + 150 Devanagari
  queries): 100.0% correct row returned in both scripts.
- **Out-of-domain rejection**: 8/8 (100%) unrelated English/Marathi queries
  rejected, 0 false positives.
- **Latency** (warm, n=60): fuzzy 2.9 ms mean; semantic 25.6 ms; full
  resolve 35.3 ms mean / 47.0 ms p95; top-10 search 170.5 ms.
- **Semantic threshold**: 0.63 sits cleanly between the in-domain
  minimum best-cosine (0.842 in the benchmark run; 0.773 in the
  extra-graphs run over a different 200-query sample) and the OOD
  maximum (0.513).
- **Fine-tuning effect** (`models/benchmark_report.json`, test n=1,040):
  Recall@1 51.4% → 99.8%, MRR@10 62.8% → 99.8%, mean positive-pair
  cosine 0.44 → 0.94.

## Honest caveats

- Only **105 of 5,200** entries (2.0%) are curated idioms; 5,095 (98%) are
  transparently labeled `pattern_v1` pattern-generated proverb frames.
  High retrieval numbers partly reflect that limited-variety subspace.
- OOD set is 8 handcrafted negatives; a larger, human-validated negative
  set remains future work.

## How to use in the LaTeX report

In `kokan_vani_report.tex`, replace the placeholder Chapter 5 body with:

```latex
\input{results/tables.tex}     % just the metric tables
% or replace the whole chapter with:
\input{results/chapter5_results.tex}
```

Keep `graphicx`, `booktabs`, `tabularx` in the preamble and the figures'
paths as `results/fig_*.png`.
