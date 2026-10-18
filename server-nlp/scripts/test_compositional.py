"""
server-nlp/scripts/test_compositional.py

Tests for the compositional fallback: an idiom that is NOT in the corpus must
produce a word-by-word reading instead of "not found".

Checks:
  A) Corpus idioms still resolve as corpus matches (no regression).
  B) Unknown proverb, all words in the lexicon  -> full coverage, composed meaning.
  C) Unknown proverb with out-of-vocabulary words -> partial coverage, and the
     unknown words are reported as unknown instead of being given a gloss.
  D) Romanised unknown proverb -> compositional reading too.
  E) Ordinary sentence -> flagged as sentence-like, not passed off as a proverb.
  F) Input with no letters -> matched False (nothing to analyse).
  G) The reading is always labelled provisional, never as an attested meaning.

Run from the project root:
    python server-nlp/scripts/test_compositional.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

_SERVER_DIR = os.path.abspath("server-nlp")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.core.query_pipeline import resolve_idiom  # noqa: E402
from app.data_loader import store                 # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def run(query):
    return resolve_idiom(query, store)


print("=" * 78)
print("A) CORPUS REGRESSION — known idioms must still be found")
print("=" * 78)
for q in ("उदक पिऊन विसर", "Modde marap", "Haat dakhvun ayaak"):
    r = run(q)
    print(f"  {q!r} -> {r.get('match_type')} ({r.get('confidence')})")
    check(f"corpus match: {q}", r.get("matched") is True
          and r.get("match_type") in {"phonetic", "semantic"}, str(r.get("match_type")))
    check(f"not provisional: {q}", not r.get("provisional"))

print()
print("=" * 78)
print("B) UNKNOWN PROVERB, ALL WORDS KNOWN -> composed meaning")
print("=" * 78)
UNKNOWN_FULL = "हात मुंडो धुवून घेवप"
r = run(UNKNOWN_FULL)
print(f"  query      : {UNKNOWN_FULL}")
print(f"  match_type : {r.get('match_type')}")
print(f"  provisional: {r.get('provisional')}")
print(f"  coverage   : {r.get('coverage')}")
print(f"  literal    : {r.get('literal_meaning')}")
print(f"  english    : {r.get('english_meaning')}")
check("unknown -> compositional", r.get("match_type") == "compositional")
check("unknown -> provisional flag", r.get("provisional") is True)
check("unknown -> every word glossed", (r.get("coverage") or {}).get("ratio") == 1.0,
      str(r.get("coverage")))
check("unknown -> composed english meaning", bool(r.get("english_meaning")))
check("unknown -> composed marathi meaning", bool(r.get("marathi_meaning")))
check("unknown -> word_glosses present", len(r.get("word_glosses") or []) == 3)
check("gloss of हात is hand",
      any(g["token"] == "हात" and g["english"] == "hand" for g in r.get("word_glosses") or []))
check("compound verb merged (धुवून + घेवप)",
      any(g.get("merged_from") for g in r.get("word_glosses") or []))
check("nearest corpus entries attached", len(r.get("closest_known") or []) > 0)
check("reading is marked as not attested",
      "not an attested meaning" in (r.get("figurative_meaning") or ""))

print()
print("=" * 78)
print("C) UNKNOWN PROVERB WITH OUT-OF-VOCABULARY WORDS -> partial coverage")
print("   (no invented glosses; the gaps are reported)")
print("=" * 78)
# One known Konkani word plus tokens that are deliberately absent from both the
# corpus and the word lexicon. Romanised tokens skip the approximate embedding
# stage by design, so this case is deterministic.
UNKNOWN_PARTIAL = "हात zubr qeltar vonsh"
r = run(UNKNOWN_PARTIAL)
print(f"  query    : {UNKNOWN_PARTIAL}")
print(f"  coverage : {r.get('coverage')}")
print(f"  words    : " + ", ".join(
    f"{g['token']}={'?' if not g['known'] else g['english']}({g['method']})"
    for g in r.get("word_glosses") or []))
check("partial -> still compositional", r.get("match_type") == "compositional")
ratio = (r.get("coverage") or {}).get("ratio", 1.0)
check("partial -> coverage below 1", ratio < 1.0, str(ratio))
unknown = [g for g in r.get("word_glosses") or [] if not g["known"]]
check("partial -> unknown words kept unglossed", len(unknown) > 0)
check("unknown words have no invented gloss",
      all(g["english"] == "" for g in unknown))
check("partial -> confidence below full reading", (r.get("confidence") or 1) < 0.72)
check("partial -> unglossed words listed as a limitation",
      "Still unrecognised" in (r.get("figurative_meaning") or ""))

print()
print("=" * 78)
print("D) ROMANISED UNKNOWN PROVERB")
print("=" * 78)
ROMAN = "Tandul potant shit kushit"
r = run(ROMAN)
print(f"  query      : {ROMAN}")
print(f"  match_type : {r.get('match_type')}  coverage={r.get('coverage')}")
check("roman unknown -> compositional", r.get("match_type") == "compositional")
check("roman unknown -> has glosses or explicit unknowns",
      len(r.get("word_glosses") or []) > 0)

print()
print("=" * 78)
print("E) ORDINARY SENTENCE -> flagged, not sold as a proverb")
print("=" * 78)
SENTENCE = "आमी बाजारात गेलो आनी भाजी घेतली"
r = run(SENTENCE)
print(f"  query : {SENTENCE}")
print(f"  notes : {r.get('notes')}")
check("sentence -> compositional", r.get("match_type") == "compositional")
check("sentence -> sentence-like note",
      any("ordinary sentence" in n for n in (r.get("notes") or [])))
check("sentence -> caveat inside the reading",
      "ordinary sentence" in (r.get("figurative_meaning") or ""))

print()
print("=" * 78)
print("F) NO LETTERS / EMPTY INPUT")
print("=" * 78)
for q in ("", "   ", "!!! ???"):
    r = run(q)
    print(f"  {q!r} -> matched={r.get('matched')}")
    check(f"empty/符号 input stays unmatched: {q!r}", r.get("matched") is False)

print()
print("=" * 78)
print("G) EVERY COMPOSED READING IS LABELLED")
print("=" * 78)
for q in (UNKNOWN_FULL, UNKNOWN_PARTIAL, ROMAN, SENTENCE):
    r = run(q)
    check(f"provisional + caveat: {q!r}",
          r.get("provisional") is True
          and any("not a corpus match" in n for n in (r.get("notes") or [])))

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} check(s) FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("RESULT: all compositional checks passed")
print("=" * 78)
