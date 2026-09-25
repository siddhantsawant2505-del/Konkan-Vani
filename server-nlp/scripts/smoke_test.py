"""
server-nlp/scripts/smoke_test.py

Live smoke test against a RUNNING Konkan Vani server (default 127.0.0.1:8000).
Covers:
  1. Health check
  2. Exact curated idiom            -> matched: true (phonetic)
  3. Misspelled variant             -> matched: true (phonetic)
  4. User-contributed idiom (Devanagari + Romanized) -> matched: true
  5. Idiom fragment                 -> word_level analysis (not a full match)
  6. Unrelated sentences            -> matched: false (no fabrication;
                                       regression guard for the 0.55-era
                                       false positives found in diagnostics)

Usage:
    python -X utf8 server-nlp/scripts/smoke_test.py [base_url]
"""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

EXACT_QUERY = "Haatak chun lavap"                    # curated real idiom
MISSPELLED_QUERY = "Haataak chun laavap"             # vowel-altered variant
USER_IDIOM_DEV = "बैं सुक्तोच, उदकाचो व्हाळेर कोल्लटा"
USER_IDIOM_ROM = "Baim suktoch, udcacho valor collta"
FRAGMENT_QUERY = "बैं सुक्तोच"                        # fragment of user idiom
NEGATIVE_QUERY = "Ami gele bajar bharli pishwi"      # absent from dataset
FALSE_POS_REGRESSION = "ती शाळेंत भुरग्यांक शिकयता"   # scored 0.5728 vs 0.55 before

failures = []


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print(f"Smoke-testing {BASE}\n")

# 1. Health
h = get("/health")
print("GET /health ->", json.dumps(h))
check("health.status == ok", h.get("status") == "ok")
check("health.idioms_loaded == 5200", h.get("idioms_loaded") == 5200, str(h.get("idioms_loaded")))

# 2. Exact curated idiom
r = post("/api/resolve", {"text": EXACT_QUERY})
print("\nPOST", json.dumps(EXACT_QUERY), "->", json.dumps(r, ensure_ascii=False))
check("exact matched=true", r.get("matched") is True)
check("exact phonetic @1.0", r.get("match_type") == "phonetic" and r.get("confidence") == 1.0)
check("word_analysis attached", isinstance(r.get("word_analysis"), list))

# 3. Misspelled variant
r = post("/api/resolve", {"text": MISSPELLED_QUERY})
print("\nPOST", json.dumps(MISSPELLED_QUERY), "->", json.dumps(r, ensure_ascii=False))
check("misspelled matched=true", r.get("matched") is True and r.get("confidence") == 1.0)

# 4. User-contributed idiom (both scripts)
r = post("/api/resolve", {"text": USER_IDIOM_DEV})
print("\nPOST", USER_IDIOM_DEV, "->", json.dumps(r, ensure_ascii=False))
check("user idiom (Devanagari) matched=true", r.get("matched") is True, str(r.get("match_type")))
check("user idiom confidence >= 0.63", (r.get("confidence") or 0) >= 0.63, str(r.get("confidence")))

r = post("/api/resolve", {"text": USER_IDIOM_ROM})
print("\nPOST", json.dumps(USER_IDIOM_ROM), "->", json.dumps(r, ensure_ascii=False))
check("user idiom (Romanized) matched=true", r.get("matched") is True, str(r.get("match_type")))

# 5. Fragment -> word_level (partial understanding, clearly labeled)
r = post("/api/resolve", {"text": FRAGMENT_QUERY})
print("\nPOST", FRAGMENT_QUERY, "->", json.dumps(r, ensure_ascii=False))
check("fragment -> word_level", r.get("matched") is True and r.get("match_type") == "word_level",
      str(r.get("match_type")))
check("fragment word_analysis has 2 words",
      r.get("match_type") == "word_level" and len(r.get("word_analysis") or []) == 2)

# 6. Negatives / regression guards
r = post("/api/resolve", {"text": NEGATIVE_QUERY})
print("\nPOST", json.dumps(NEGATIVE_QUERY), "->", json.dumps(r, ensure_ascii=False))
check("negative matched=false", r.get("matched") is False)

r = post("/api/resolve", {"text": FALSE_POS_REGRESSION})
print("\nPOST", FALSE_POS_REGRESSION, "->", json.dumps(r, ensure_ascii=False))
check("false-positive regression: matched=false", r.get("matched") is False,
      f"match_type={r.get('match_type')}")

print("\n" + "=" * 50)
if failures:
    print(f"SMOKE TEST FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED: all checks OK")
