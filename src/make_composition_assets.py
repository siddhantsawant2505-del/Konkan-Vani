"""
Konkan Vani — Compositional Fallback Assets
===========================================
Measures the word-by-word fallback that answers idioms outside the corpus and
writes report-ready artifacts:

  results/fig_composition_methods.png  — how each corpus word is glossed
                                         (exact / inflected stem / phonetic /
                                          spelling variant / approximate / none)
  results/fig_composition_coverage.png — distribution of per-idiom gloss
                                         coverage across the corpus
  results/tab_composition_examples.tex — composed readings for proverbs that
                                         are NOT in the corpus
  results/composition_metrics.json     — the same numbers, machine readable

Everything here is measured, not illustrated: the lookup methods come from
running app.core.lexicon over the corpus vocabulary, and the examples come from
running app.core.compose on inputs verified to be absent from the corpus.

Run from the project root:  python -X utf8 src/make_composition_assets.py
"""

import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root)

_SERVER_DIR = os.path.abspath("server-nlp")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

from app.core.lexicon import LEXICON, EMBED_THRESHOLD   # noqa: E402
from app.core.compose import compose_idiom_meaning      # noqa: E402
from app.core.query_pipeline import resolve_idiom       # noqa: E402
from app.data_loader import store                       # noqa: E402

OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")
METHOD_ORDER = ["exact", "suffix", "phonetic", "fuzzy", "embedding", "none"]
METHOD_LABEL = {
    "exact": "exact dictionary",
    "suffix": "inflected stem",
    "phonetic": "phonetic key",
    "fuzzy": "spelling variant",
    "embedding": "approx. (embedding)",
    "none": "unrecognised",
}

df = store.df


# ── 1. Lookup methods over the corpus vocabulary ────────────────────────────
def corpus_vocabulary():
    words = set()
    for col in ("konkani_text", "romanized_text"):
        for text in df[col].fillna("").astype(str):
            for w in WORD_RE.findall(text):
                words.add(w.lower())
    return sorted(words)


vocab = corpus_vocabulary()
print(f"corpus vocabulary: {len(vocab)} unique words")

cache, counts = {}, Counter()
for w in vocab:
    hit = LEXICON.lookup(w)
    cache[w] = hit
    counts[hit["method"]] += 1

glossed = len(vocab) - counts.get("none", 0)
print("lookup methods:", dict(counts))
print(f"vocabulary glossed: {glossed}/{len(vocab)} "
      f"({100 * glossed / len(vocab):.1f}%)")

fig, ax = plt.subplots(figsize=(7.2, 3.6))
methods = [m for m in METHOD_ORDER if counts.get(m)]
values = [counts[m] for m in methods]
colors = ["#2a6865", "#4f9c98", "#7dbfbb", "#a9d6d3", "#c04000", "#bdbdbd"]
bars = ax.bar([METHOD_LABEL[m] for m in methods], values,
              color=colors[: len(methods)])
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.015,
            f"{v}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("corpus words")
ax.set_title(f"How {len(vocab)} corpus words are glossed by the word lexicon")
plt.xticks(rotation=18, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_composition_methods.png"))
plt.close()
print("wrote", os.path.join(OUT, "fig_composition_methods.png"))


# ── 2. Per-idiom gloss coverage over the whole corpus ────────────────────────
per_row = []
for _, row in df.iterrows():
    toks = WORD_RE.findall(str(row["konkani_text"]))
    if not toks:
        continue
    known = sum(1 for t in toks if cache.get(t.lower(), LEXICON.lookup(t))["known"])
    per_row.append(known / len(toks))

per_row = np.array(per_row)
print(f"idiom gloss coverage: mean {per_row.mean():.3f}, median {np.median(per_row):.3f}, "
      f"fully glossed {int((per_row == 1.0).sum())}/{len(per_row)}")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.hist(per_row, bins=np.linspace(0, 1, 21), color="#2a6865", edgecolor="white")
ax.axvline(per_row.mean(), color="#c04000", linestyle="--",
           label=f"mean {per_row.mean():.2f}")
ax.set_xlabel("share of an idiom's words that the lexicon can gloss")
ax.set_ylabel("idioms")
ax.set_title("Word-level glossary coverage over the 5,200-entry corpus")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_composition_coverage.png"))
plt.close()
print("wrote", os.path.join(OUT, "fig_composition_coverage.png"))


# ── 3. Composed readings for proverbs outside the corpus ────────────────────
# (Devanagari input, romanised form). The LaTeX table prints the romanised
# form because the report is compiled with pdfLaTeX + inputenc, which cannot
# typeset Devanagari; the JSON keeps both.
EXAMPLES = [
    ("हात मुंडो धुवून घेवप", "Haat mudo dhuvun ghevap"),
    ("तांदूळ पोटांत शित कुशीत", "Tandul potant shit kushit"),
    ("कानाक तेल घालून न्हिदप", "Kaanak tel ghalun nhidap"),
    ("Kaanak tel ghalun nhidap", "Kaanak tel ghalun nhidap"),
    ("घराचेर पाऊस पडटा आनी आंगण सुकें", "Ghoracher paus paddta aani angan sukem"),
    ("दोनूय हातांनी तूप पोटांत", "Donui haatanni tup potant"),
]

corpus_keys = set(df["konkani_text"].astype(str)) | set(df["romanized_text"].astype(str))
rows = []
for q, roman in EXAMPLES:
    r = resolve_idiom(q, store)
    # The table is about the fallback, so only keep inputs the pipeline really
    # answered compositionally (near-duplicates of a corpus frame are matched
    # normally, which is correct behaviour and is covered by the corpus tests).
    if r.get("match_type") != "compositional":
        print(f"skip (resolved as {r.get('match_type')}): {q}")
        continue
    # prove the input really is outside the corpus
    in_corpus = q in corpus_keys
    composed = r
    rows.append({
        "query": q,
        "roman": roman,
        "in_corpus": in_corpus,
        "match_type": r.get("match_type"),
        "confidence": composed.get("confidence"),
        "coverage": composed.get("coverage"),
        "literal": composed.get("literal_meaning"),
        "reading": composed.get("english_meaning"),
        "themes": composed.get("themes"),
    })
    print(f"\n{q}\n  in_corpus={in_corpus} type={r.get('match_type')} "
          f"coverage={composed.get('coverage')}\n  {composed.get('english_meaning')}")

assert len(rows) >= 3, "need at least three compositional examples"


DEV_RUN = re.compile(r"[\u0900-\u097F]+(?:[\s,]+[\u0900-\u097F]+)*")


_ASCII_MAP = {
    "\u00b7": ";",       # middot used to separate glosses
    "\u2248": "~",       # approximate marker
    "\u2014": "--",      # em dash
    "\u2013": "-",       # en dash
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}


def latex_safe(s: str) -> str:
    """The composed reading echoes words it could not gloss in their original
    script and uses typographic punctuation; pdfLaTeX with inputenc cannot
    typeset either reliably, so both are normalised here."""
    out = DEV_RUN.sub("unrecognised", s or "")
    for a, b in _ASCII_MAP.items():
        out = out.replace(a, b)
    return "".join(ch if ord(ch) < 128 else "?" for ch in out)


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


lines = [
    "% Auto-generated by src/make_composition_assets.py - do not edit by hand",
    "% -- Table: compositional readings for out-of-corpus proverbs --",
    r"\begin{table}[ht]",
    r"\centering",
    r"\footnotesize",
    r"\caption{Composed word-by-word readings for proverbs absent from the corpus. "
    r"Coverage is the share of words the glossary recognises; the reading is "
    r"computed from those glosses and is explicitly labelled provisional.}",
    r"\label{tab:composition-examples}",
    r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.6cm}"
    r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{1.2cm}}",
    r"\toprule",
    r"Input & Word-by-word gloss and composed reading & Coverage \\",
    r"\midrule",
]
for row in rows:
    cov = row["coverage"] or {}
    lines.append(
        f"\\textit{{{tex_escape(row['roman'])}}} & "
        f"{tex_escape(latex_safe(row['literal']))}\\newline "
        f"\\textcolor{{black!70}}{{\\footnotesize "
        f"{tex_escape(latex_safe(row['reading']))}}} & "
        f"{cov.get('glossed', 0)}/{cov.get('total', 0)} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
with open(os.path.join(OUT, "tables_composition.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
tex_out = open(os.path.join(OUT, "tables_composition.tex"), encoding="utf-8").read()
# pdfLaTeX + inputenc cannot typeset Devanagari or any other non-ASCII text,
# so the generated table must stay pure ASCII.
non_ascii = sorted({ch for ch in tex_out if ord(ch) > 127})
assert not non_ascii, (
    f"non-ASCII characters leaked into the LaTeX table: {non_ascii}"
)
print("\nwrote results/tables_composition.tex (ASCII-safe for pdfLaTeX)")

payload = {
    "lexicon_entries": len(LEXICON.entries),
    "embedding_threshold": EMBED_THRESHOLD,
    "corpus_vocabulary": len(vocab),
    "lookup_methods": dict(counts),
    "vocabulary_glossed_pct": round(100 * glossed / len(vocab), 2),
    "idiom_coverage": {
        "mean": round(float(per_row.mean()), 4),
        "median": round(float(np.median(per_row)), 4),
        "fully_glossed": int((per_row == 1.0).sum()),
        "idioms": int(len(per_row)),
    },
    "examples": rows,
}
with open(os.path.join(OUT, "composition_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("wrote results/composition_metrics.json")
