"""
server-nlp/scripts/test_pipeline.py

Standalone test script for the resolve_idiom() pipeline.
Tests:
  A) 5 idioms exactly as they appear in the dataset       → expect matched: True
  B) Same 5 with deliberately altered spelling             → expect matched: True
  C) 2 sentences that are NOT dataset idioms  → must NOT be claimed as a
     corpus match; they get a provisional word-by-word reading instead
     (match_type="compositional"). This replaces the old "matched: False"
     contract, which showed the user an empty result screen.

Run from project root:
    python server-nlp/scripts/test_pipeline.py
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")

# Add server-nlp/ to sys.path so 'app' package resolves correctly
_SERVER_DIR = os.path.abspath("server-nlp")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from app.core.query_pipeline import resolve_idiom
from app.data_loader import store

# ─────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────

# A) Exact idioms (should all pass phonetic fuzzy match)
exact_cases = [
    ("Haat dakhvun ayaak",          "exact_1"),
    ("Udak pionn visor",             "exact_2"),
    ("Modde marap",                  "exact_3"),
    ("Kullantlo kollso",             "exact_4"),
    ("Kanak tel ghalun bosop",       "exact_5"),
]

# B) Same 5 with deliberate spelling alterations
#    Double vowels changed, one letter swapped — tests phonetic normalizer
altered_cases = [
    ("Haat dakhwun aayaak",          "altered_1"),  # aa→a normalised; w/v swap
    ("Udaak pioone visor",           "altered_2"),  # extra vowel
    ("Modde Maarap",                 "altered_3"),  # capital + aa
    ("Kulantlo kolso",               "altered_4"),  # missing ll
    ("Kaanak tel ghaalun bosop",     "altered_5"),  # aa→a
]

# C) Sentences that are NOT in the dataset. They must not be reported as a
#    corpus match, but they must now be explained word by word.
negative_cases = [
    ("The weather in Mumbai is very humid today",          "negative_1"),
    ("Ami gele bajar bharli pishwi",                       "negative_2"),  # plausible-looking Konkani not in dataset
    ("आजचो दिस खूब गरम आसा",                              "negative_3"),  # Devanagari sentence, not an idiom
]

CORPUS_MATCHES = {"phonetic", "semantic"}

# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"

total = 0
passed = 0

def run_case(query: str, label: str, expect_match: bool):
    global total, passed
    total += 1
    result = resolve_idiom(query, store)
    matched    = result.get("matched", False)
    match_type = result.get("match_type", "none")
    confidence = result.get("confidence", 0.0)
    ok = (matched == expect_match)
    if ok:
        passed += 1
    status = PASS if ok else FAIL
    print(f"{status}  [{label}]")
    print(f"      query      : {query!r}")
    print(f"      expect     : matched={expect_match}")
    print(f"      got        : matched={matched}, type={match_type}, confidence={confidence:.4f}")
    if matched:
        print(f"      konkani    : {result.get('konkani_text', '')}")
        print(f"      figurative : {result.get('figurative_meaning', '')[:80]}")
    print()

print("=" * 70)
print("KONKAN VANI — PIPELINE TEST SUITE")
print("=" * 70)

print("\n── A) EXACT DATASET IDIOMS (expect: matched=True) ──────────────────")
for query, label in exact_cases:
    run_case(query, label, expect_match=True)

print("\n── B) SPELLING-VARIANT IDIOMS (expect: matched=True) ────────────────")
for query, label in altered_cases:
    run_case(query, label, expect_match=True)

print("\n── C) NON-DATASET SENTENCES (expect: provisional reading, not a corpus match) ──")
for query, label in negative_cases:
    total += 1
    result = resolve_idiom(query, store)
    match_type = result.get("match_type")
    provisional = result.get("provisional")
    ok = (
        result.get("matched") is True
        and match_type == "compositional"
        and provisional is True
        and match_type not in CORPUS_MATCHES
    )
    if ok:
        passed += 1
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  [{label}]")
    print(f"      query      : {query!r}")
    print(f"      expect     : match_type=compositional, provisional=True")
    print(f"      got        : match_type={match_type}, provisional={provisional}, "
          f"confidence={result.get('confidence')}")
    print(f"      coverage   : {result.get('coverage')}")
    print(f"      literal    : {result.get('literal_meaning')}")
    print()

print("=" * 70)
print(f"RESULT: {passed}/{total} tests passed")
print("=" * 70)

if passed < total:
    sys.exit(1)
