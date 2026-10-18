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
| `fig_composition_methods.png` | How the word lexicon glosses each corpus word | `python -X utf8 src/make_composition_assets.py` |
| `fig_composition_coverage.png` | Per-idiom glossary coverage histogram | (same script) |
| `tables_composition.tex` | Composed readings for out-of-corpus proverbs | (same script) |
| `composition_metrics.json` | Lookup-method counts + example readings | (same script) |
| `lexicon_calibration.json` | Accuracy/coverage of the approximate gloss layer | `python -X utf8 src/calibrate_lexicon.py` |
| `chapter5_results.tex` | Full LaTeX "Testing and Results" chapter, ready to `\input` | written by hand from the metrics |

## Headline numbers (all measured, not estimated)

- **Stage-wise top-1 accuracy** (200 sampled entries): fuzzy 100% exact /
  100% corrupted spelling; semantic 100% exact / 99% corrupted.
- **End-to-end pipeline** (`resolve_idiom`, 150 Romanized + 150 Devanagari
  queries): 100.0% correct row returned in both scripts.
- **Out-of-domain behaviour**: 8/8 unrelated English/Marathi queries were
  *not* claimed as known proverbs (0 false positives); all 8 received a
  provisional word-by-word reading instead of an empty screen.
- **Latency** (warm, n=60, latest run): fuzzy 5.4 ms mean; semantic 25.7 ms;
  full resolve 33.9 ms mean / 41.3 ms p95; top-10 search 38.1 ms (down from
  ~350 ms after the substring index was cached).
- **Word-by-word fallback** (`composition_metrics.json`): the 376-entry word
  lexicon glosses 664/763 corpus words (87.0%) and 99.7% of idioms have every
  word glossed; the approximate cross-lingual layer is 86.6% precise at its
  0.70 threshold (`lexicon_calibration.json`).
- **Semantic threshold**: 0.63 sits cleanly between the in-domain
  minimum best-cosine (0.842 in the benchmark run; 0.773 in the
  extra-graphs run over a different 200-query sample) and the OOD
  maximum (0.513).
- **Fine-tuning effect** (`models/benchmark_report.json`, test n=1,040):
  Recall@1 51.4% → 99.8%, MRR@10 62.8% → 99.8%, mean positive-pair
  cosine 0.44 → 0.94.

## How an unknown proverb is answered

The corpus only resolves what it contains. For anything else, `resolve_idiom`
falls through to a compositional stage (`server-nlp/app/core/compose.py`) that
glosses the query word by word from `data/lexicon/konkani_lexicon.csv` and
composes a provisional reading:

1. **lookup ladder** per word — exact headword, inflected stem (case/verb
   endings stripped), phonetic key, romanised spelling variant, then (for
   Devanagari only) an approximate cross-lingual gloss from LaBSE;
2. **composition** — literal gloss sequence, action word (Konkani perfective
   compounds merged), imagery theme from semantic tags, negation and
   reduplication cues;
3. **honesty** — the answer is flagged `provisional`, words that could not be
   glossed are left untranslated, and the nearest corpus idioms are shown
   separately as comparisons.

The semantic stage additionally requires lexical overlap (>= 34% of the
query's content words) with a candidate, so a topically related but different
proverb is no longer returned as the answer.

Regenerate the lexicon with `python -X utf8 src/build_lexicon.py`.

## Honest caveats

- Only **105 of 5,200** entries (2.0%) are curated idioms; 5,095 (98%) are
  transparently labeled `pattern_v1` pattern-generated proverb frames.
  High retrieval numbers partly reflect that limited-variety subspace.
- OOD set is 8 handcrafted negatives; a larger, human-validated negative
  set remains future work.
- The word lexicon is hand-curated (376 entries) and only approximates
  coverage; composed readings are provisional by construction and should be
  confirmed by a native speaker.

## How to use in the LaTeX report

In `kokan_vani_report.tex`, replace the placeholder Chapter 5 body with:

```latex
\input{results/tables.tex}     % just the metric tables
% or replace the whole chapter with:
\input{results/chapter5_results.tex}
```

Keep `graphicx`, `booktabs`, `tabularx` in the preamble and the figures'
paths as `results/fig_*.png`.
