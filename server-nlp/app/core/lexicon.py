"""
server-nlp/app/core/lexicon.py
==============================
Konkani word-gloss lookup for the compositional-meaning fallback.

The idiom corpus stores *whole-idiom* meanings, never word meanings, so this
module supplies the missing word layer from
``data/lexicon/konkani_lexicon.csv`` (built by ``src/build_lexicon.py``).

Lookup is layered, cheapest first, and every hit records HOW it was found so
the UI can be honest about it:

  1. ``exact``      — token is literally a lexicon headword (Devanagari or
                      the roman transliteration, or its phonetic key).
  2. ``suffix``     — token is an inflected form; Konkani case / verb endings
                      are stripped ("पाण्यात" -> "पाणी", "ghalun" -> "ghal")
                      and the stem is looked up again.
  3. ``phonetic``   — after phonetic_normalize(), the token matches a headword
                      key (v/w, aa/a, ee/i, oo/u, zh/j variants).
  4. ``fuzzy``      — romanised input that is a near-miss spelling of a
                      headword (rapidfuzz, cutoff 84).
  5. ``embedding``  — a Devanagari word not listed at all. Its LaBSE vector is
                      compared with the ENGLISH glosses of neighbouring
                      lexicon entries (LaBSE is cross-lingual), which yields a
                      *candidate* gloss plus a similarity. Marked approximate.
                      Romanised tokens skip this stage: measured Recall@1 on
                      the lexicon is 0.06 for roman vs 0.43 for Devanagari.
  6. ``none``       — nothing plausible; the word is reported as unglossed
                      rather than given an invented meaning.

No network access and no per-request corpus scan are involved: the CSV is
loaded once and the embedding fallback caches its matrix.
"""

import csv
import logging
import os
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_LEXICON_PATH = os.path.join(
    _PROJECT_ROOT, "data", "lexicon", "konkani_lexicon.csv"
)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Minimum cosine for the cross-lingual "approximate" gloss. Measured by
# src/calibrate_lexicon.py on the real lexicon: at 0.70 the accepted glosses
# are ~89% correct (35% of words), while romanised input is far noisier
# (Recall@1 0.06), so the embedding stage is only used for Devanagari tokens.
EMBED_THRESHOLD = 0.70

# rapidfuzz cutoff for romanised spelling variants (short words, so strict).
ROMAN_FUZZY_CUTOFF = 84

# Konkani case / postposition / verb endings, longest first.
_DEV_SUFFIXES = [
    "सारखे", "सारखा", "सारखी", "ल्याला", "्यात", "ांच्या", "ाच्या",
    "ांनी", "ांना", "ांत", "ाचो", "ाचें", "ाची", "ाचा", "ाकडे",
    "ावर", "ेर", "ात", "ाक", "ाल", "ाने", "ाण", "तून", "ून",
    "लें", "लो", "ली", "तात", "ता", "ते", "णे", "ण्या", "ीत",
    "ींत", "े", "ो", "ी", "ा", "ां", "त", "न", "क",
]

# Roman endings (compared after phonetic_normalize, so v/w and aa/a merges
# have already happened).
_ROM_SUFFIXES = [
    "sarkhe", "sarkha", "sarkhi", "lyala", "achi", "acho", "achem",
    "anna", "anni", "ache", "chya", "chao", "chem", "chi", "cho", "cha",
    "akde", "avar", "ne", "ak", "al", "er", "at", "un", "tat", "ta", "te",
    "lem", "lo", "li", "ny", "ne", "it", "nt", "ny", "o", "e", "i", "a",
]


class Lexicon:
    """In-memory word glossary with layered, self-describing lookups."""

    def __init__(self, path: str = _LEXICON_PATH):
        self.path = path
        self.entries = []
        self.by_dev = {}
        self.by_key = {}
        self._roman_pairs = []
        self._emb_lock = threading.Lock()
        self._gloss_emb = None
        self._gloss_list = None
        self._load()

    # ── loading ────────────────────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.path):
            logger.error(
                "Word lexicon missing at %s — run 'python src/build_lexicon.py'.",
                self.path,
            )
            return
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                entry = {
                    "devanagari": (row.get("devanagari") or "").strip(),
                    "roman": (row.get("roman") or "").strip(),
                    "phonetic_key": (row.get("phonetic_key") or "").strip(),
                    "english": (row.get("english_gloss") or "").strip(),
                    "marathi": (row.get("marathi_gloss") or "").strip(),
                    "pos": (row.get("pos") or "").strip(),
                    "tag": (row.get("semantic_tag") or "").strip(),
                }
                if not entry["devanagari"] or not entry["english"]:
                    continue
                self.entries.append(entry)
                self.by_dev.setdefault(entry["devanagari"], entry)
                if entry["phonetic_key"]:
                    self.by_key.setdefault(entry["phonetic_key"], entry)
                if entry["roman"]:
                    rkey = _key(entry["roman"])
                    self.by_key.setdefault(rkey, entry)
                    if rkey:
                        self._roman_pairs.append((rkey, entry))
        logger.info("Word lexicon loaded: %d entries", len(self.entries))

    # ── public API ─────────────────────────────────────────────────────────
    def lookup(self, token: str, allow_embedding: bool = True) -> dict:
        """Gloss one token. Always returns a dict describing what happened."""
        token = (token or "").strip()
        if not token:
            return _miss(token, "none")

        is_dev = bool(_DEVANAGARI_RE.search(token))

        # 1. exact
        key = _key(token) if not is_dev else ""
        if is_dev and token in self.by_dev:
            return _hit(token, self.by_dev[token], "exact", 1.0, token)
        if key and key in self.by_key:
            return _hit(token, self.by_key[key], "exact", 1.0, key)

        # 2. suffix stripping (inflected forms)
        for stem in _stems(token, is_dev):
            if is_dev:
                if stem in self.by_dev:
                    return _hit(token, self.by_dev[stem], "suffix", 0.92, stem)
            else:
                skey = _key(stem)
                if skey and skey in self.by_key:
                    return _hit(token, self.by_key[skey], "suffix", 0.88, skey)
                # one more strip after phonetic merging ("ghaalun" -> "ghalun")
                for stem2 in _stems(skey, False):
                    k2 = _key(stem2)
                    if k2 and k2 in self.by_key:
                        return _hit(token, self.by_key[k2], "suffix", 0.84, k2)

        # 3. phonetic equivalence only (adds aa/a, v/w, zh/j, ee/i merges)
        if key and key in self.by_key:
            return _hit(token, self.by_key[key], "phonetic", 0.85, key)

        # 4. romanised spellings: fuzzy match against the roman headwords
        if not is_dev:
            fuzzy = self._roman_fuzzy(token)
            if fuzzy is not None:
                return fuzzy

        # 5. cross-lingual approximate gloss (Devanagari only — see above)
        if allow_embedding and is_dev:
            approx = self._embedding_gloss(token)
            if approx is not None:
                return approx

        return _miss(token, "none")

    def lookup_many(self, tokens: list[str]) -> list[dict]:
        return [self.lookup(t) for t in tokens]

    # ── romanised spelling variants ────────────────────────────────────────
    def _roman_fuzzy(self, token: str) -> dict | None:
        if not self._roman_pairs:
            return None
        from rapidfuzz import fuzz, process

        result = process.extractOne(
            _key(token),
            [k for k, _ in self._roman_pairs],
            scorer=fuzz.ratio,
            score_cutoff=ROMAN_FUZZY_CUTOFF,
        )
        if result is None:
            return None
        _matched_key, score, idx = result
        entry = self._roman_pairs[idx][1]
        return _hit(token, entry, "fuzzy", round(score / 100.0, 4), entry["roman"])

    # ── cross-lingual fallback ─────────────────────────────────────────────
    def _ensure_gloss_matrix(self):
        if self._gloss_emb is not None or not self.entries:
            return
        with self._emb_lock:
            if self._gloss_emb is not None:
                return
            from app.core.model_registry import encode

            glosses = [e["english"] for e in self.entries]
            try:
                emb = np.asarray(encode(glosses), dtype=np.float32)
            except Exception as exc:  # model unavailable — degrade silently
                logger.warning("Lexicon embedding fallback unavailable: %s", exc)
                self._gloss_emb = np.zeros((0, 768), dtype=np.float32)
                self._gloss_list = []
                return
            self._gloss_emb = emb
            self._gloss_list = glosses
            logger.info("Lexicon gloss matrix ready: %s", emb.shape)

    def _embedding_gloss(self, token: str) -> dict | None:
        self._ensure_gloss_matrix()
        if self._gloss_emb is None or len(self._gloss_emb) == 0:
            return None
        from app.core.model_registry import encode

        try:
            vec = np.asarray(encode([token])[0], dtype=np.float32)
        except Exception as exc:
            logger.debug("Embedding gloss encode failed: %s", exc)
            return None

        sims = self._gloss_emb @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < EMBED_THRESHOLD:
            return None
        entry = self.entries[best]
        return {
            "token": token,
            "devanagari": entry["devanagari"],
            "roman": entry["roman"],
            "english": entry["english"],
            "marathi": entry["marathi"],
            "pos": entry["pos"],
            "tag": entry["tag"],
            "method": "embedding",
            "similarity": round(score, 4),
            "matched_on": entry["english"],
            "approximate": True,
            "known": True,
        }


def _key(text: str) -> str:
    """Same phonetic rules as app/core/phonetic_normalize.py."""
    norm = (text or "").lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"),
                 ("ph", "f"), ("v", "w"), ("zh", "j")):
        norm = norm.replace(a, b)
    norm = re.sub(r"[^a-z\s]", "", norm)
    return " ".join(norm.split())


def _stems(token: str, is_dev: bool) -> list[str]:
    """Candidate stems produced by stripping one or two Konkani endings."""
    suffixes = _DEV_SUFFIXES if is_dev else _ROM_SUFFIXES
    out = []
    for suf in suffixes:
        if len(token) > len(suf) + (2 if is_dev else 3) and token.endswith(suf):
            out.append(token[: -len(suf)])
    # second pass on the first-pass stems (e.g. "पाण्यात" -> "पाण्य" -> "पाणी")
    for stem in list(out):
        for suf in suffixes:
            if len(stem) > len(suf) + (2 if is_dev else 3) and stem.endswith(suf):
                out.append(stem[: -len(suf)])
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _hit(token: str, entry: dict, method: str, sim: float, matched_on: str) -> dict:
    return {
        "token": token,
        "devanagari": entry["devanagari"],
        "roman": entry["roman"],
        "english": entry["english"],
        "marathi": entry["marathi"],
        "pos": entry["pos"],
        "tag": entry["tag"],
        "method": method,
        "similarity": round(sim, 4),
        "matched_on": matched_on,
        "approximate": method == "embedding",
        "known": True,
    }


def _miss(token: str, method: str) -> dict:
    return {
        "token": token,
        "devanagari": token if _DEVANAGARI_RE.search(token) else "",
        "roman": "" if _DEVANAGARI_RE.search(token) else token,
        "english": "",
        "marathi": "",
        "pos": _guess_pos(token),
        "tag": "",
        "method": method,
        "similarity": 0.0,
        "matched_on": "",
        "approximate": False,
        "known": False,
    }


def _guess_pos(token: str) -> str:
    """Morphological guess for unglossed words, so the composer can still tell
    an action from a thing. Deliberately coarse and labelled as a guess."""
    t = token.lower()
    if t.endswith(("प", "णे", "ता", "लो", "ली", "लें", "चें", "यता")):
        return "verb?"
    if t.endswith(("ाक", "ाला", "ात", "ेर", "ांनी", "चो", "ाचो")):
        return "noun?"
    if t.endswith(("un", "la", "ak", "at", "ne", "lo", "li", "lem", "ap")):
        return "verb?"
    return "word?"


# Singleton — the CSV is tiny, load once per process.
LEXICON = Lexicon()
