"""
server-nlp/app/core/compose.py
==============================
Compositional meaning for idioms that are NOT in the corpus.

The corpus resolves known proverbs directly. For an unknown one the old
behaviour was ``{"matched": false}`` — an empty screen. This module replaces
that dead end with a word-by-word reading:

  1. split the input into words (Devanagari or romanised);
  2. gloss every word with ``app.core.lexicon`` (dictionary hit, inflected
     stem, phonetic equivalent, spelling variant, or a labelled approximate
     cross-lingual candidate);
  3. infer the imagery / theme from the glossed words' semantic tags;
  4. compose a *provisional* phrase-level meaning from the glosses, the
     action word and the theme, stating exactly how much of the phrase was
     actually recognised;
  5. attach the nearest corpus entries, so the user can compare with real,
     attested idioms even when none of them crossed the matching threshold.

Everything this module returns is labelled ``provisional``. It never presents
a computed reading as a dictionary-attested meaning, and it never invents a
gloss for a word it could not recognise — unglossed words are shown verbatim.
"""

import logging
import re

from app.core.lexicon import LEXICON

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")

# Parts of speech that shape a phrase rather than name its imagery.
FUNCTION_POS = {"connective", "auxiliary", "negation", "quantifier", "adverb", "pronoun"}

# Konkani builds perfective compounds from a participle plus a light verb:
# "धुवून घेवप" = wash (for oneself), "मारून टाकप" = strike away. Merging the
# pair stops the light verb being read as a separate action word.
_COMPOUND_SENSE = {
    "घेवप": "for oneself", "घेणे": "for oneself", "घेतली": "for oneself",
    "घेतलो": "for oneself", "दिवप": "for someone else", "दिलो": "for someone else",
    "दिली": "for someone else", "घालप": "into / onto", "टाकप": "away / down",
    "बसप": "down", "जावप": "become", "सोडप": "off",
    "ghevap": "for oneself", "divop": "for someone else", "takap": "away / down",
}
_PERFECTIVE = ("ून", "ऊन", "un")
NEGATION_POS = {"negation"}

# Most specific imagery first; "motion" is the least informative tag because
# every verb carries it.
_THEME_PRIORITY = [
    "water", "food", "animal", "body", "ritual", "place",
    "person", "emotion", "quality", "speech", "perception", "nature", "motion",
]

_EN_THEME = {
    "water": "water, flow and things that cannot be held back",
    "food": "food, sustenance and what is shared or kept back",
    "animal": "animal behaviour used as a picture of human habit",
    "body": "the body and what a person does with or to it",
    "ritual": "ritual duty, blessing and the sacred",
    "place": "home, village and the shared spaces of daily life",
    "person": "relations between people",
    "emotion": "feeling and its consequences",
    "quality": "worth, gain and loss",
    "speech": "speech, promises and reputation",
    "perception": "noticing, knowing and remembering",
    "nature": "seasons, weather and the natural cycle",
    "motion": "action and the result it produces",
}

_MR_THEME = {
    "water": "पाणी, प्रवाह आणि जे धरून ठेवता येत नाही",
    "food": "अन्न, उदरनिर्वाह आणि वाटून घेणे",
    "animal": "प्राण्यांचे वर्तन माणसाच्या सवयीचे रूपक म्हणून",
    "body": "शरीर आणि माणूस स्वतःशी करत असलेले वर्तन",
    "ritual": "विधी, आशीर्वाद आणि पवित्रता",
    "place": "घर, गाव आणि रोजच्या आयुष्यातील जागा",
    "person": "माणसांमधील नाती",
    "emotion": "भावना आणि त्यांचे परिणाम",
    "quality": "मोल, फायदा आणि हानी",
    "speech": "बोलणे, वचन आणि प्रतिष्ठा",
    "perception": "लक्ष देणे, जाणणे आणि विसरणे",
    "nature": "ऋतू, हवामान आणि निसर्गचक्र",
    "motion": "कृती आणि त्याचा परिणाम",
}

_CONF_BY_METHOD = {
    "exact": 1.0,
    "suffix": 0.90,
    "phonetic": 0.85,
    "fuzzy": 0.80,
}
# A computed phrase-level reading is never as trustworthy as a corpus match.
COMPOSITION_CEILING = 0.72


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _dedupe_reduplication(tokens: list[str]) -> tuple[list[str], bool]:
    """Konkani repeats a word for emphasis ("मोठो मोठो"). Collapse for
    glossing, but remember that it happened."""
    out, repeated = [], False
    for tok in tokens:
        if out and out[-1].lower() == tok.lower():
            repeated = True
            continue
        out.append(tok)
    return out, repeated


def _word_confidence(g: dict) -> float:
    if not g["known"]:
        return 0.0
    if g["method"] in _CONF_BY_METHOD:
        return _CONF_BY_METHOD[g["method"]]
    if g["method"] == "embedding":
        # cosine above the 0.70 acceptance floor, scaled into 0.50-0.80
        return round(min(0.80, 0.50 + (g["similarity"] - 0.70) * 3.0), 4)
    return 0.0


# Irregular past forms used by the lexicon's own glosses, so the composed
# phrase reads "to take" instead of "to took".
_IRREGULAR = {
    "took": "take", "went": "go", "gone": "go", "gave": "give",
    "became": "become", "did": "do", "came": "come", "was": "be",
    "saw": "see", "ate": "eat", "drank": "drink", "drunk": "drink",
    "ran": "run", "sang": "sing", "wrote": "write", "written": "write",
    "spoke": "speak", "spoken": "speak", "held": "hold", "found": "find",
    "kept": "keep", "broke": "break", "built": "build", "learnt": "learn",
    "wept": "weep", "forgot": "forget", "flew": "fly", "sat": "sit",
    "won": "win", "told": "tell", "said": "say", "shown": "show",
    "felt": "feel", "grew": "grow", "threw": "throw", "understood": "understand",
}


def _verb_lemma(english: str) -> str:
    """Reduce a gloss like 'having washed' / 'to wash' / 'took (f.)' to a bare
    action so it can be dropped into an English phrase."""
    e = (english or "").split(" / ")[0].strip()
    for prefix in ("having ", "to "):
        if e.startswith(prefix):
            e = e[len(prefix):]
    e = re.sub(r"\s*\((?:m\.|f\.)\)", "", e).strip()
    head, _, tail = e.partition(" ")
    if head.lower() in _IRREGULAR:
        e = _IRREGULAR[head.lower()] + (" " + tail if tail else "")
    elif head.endswith("ed") and "(" not in head and head.lower() not in _IRREGULAR:
        e = head[:-2] + (" " + tail if tail else "")
    return e


def _merge_compounds(glosses: list[dict]) -> list[dict]:
    """Collapse participle + light-verb pairs into one action entry."""
    out, i = [], 0
    while i < len(glosses):
        cur = glosses[i]
        nxt = glosses[i + 1] if i + 1 < len(glosses) else None
        tok = cur["token"].lower()
        if (cur["known"] and nxt is not None and nxt["token"] in _COMPOUND_SENSE
                and tok.endswith(_PERFECTIVE)):
            merged = dict(cur)
            lemma = _verb_lemma(cur["english"])
            merged["english"] = f"{lemma} ({_COMPOUND_SENSE[nxt['token']]})"
            merged["merged_from"] = [cur["token"], nxt["token"]]
            merged["pos"] = "verb"
            out.append(merged)
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _primary_gloss(g: dict) -> str:
    """First sense of a gloss ('stomach / belly' -> 'stomach') for composing
    readable English. The full gloss is still shown in the word table."""
    return (g.get("english") or "").split(" / ")[0].strip()


# Leading material that reads badly inside a composed English phrase
# ("to the ear" -> "ear", "in the stomach" -> "stomach").
_LEAD_WORDS = ("to ", "in ", "on ", "at ", "of ", "with ", "from ", "the ", "a ", "an ")


def _noun_phrase(g: dict) -> str:
    e = _primary_gloss(g)
    changed = True
    while changed and e:
        changed = False
        for prefix in _LEAD_WORDS:
            if e.lower().startswith(prefix) and len(e) > len(prefix) + 1:
                e = e[len(prefix):]
                changed = True
    return e


def _things_text(things: list[dict]) -> str:
    parts = [_noun_phrase(t) for t in things if t.get("english")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + " and " + parts[-1]
    if things and all(t["tag"] == "body" for t in things if t.get("english")):
        return f"one's own {body}"
    return body


def _literal_sequence(glosses: list[dict]) -> str:
    parts = []
    for g in glosses:
        if g["known"]:
            marker = "≈" if g["method"] == "embedding" else ""
            parts.append(f"{marker}{_primary_gloss(g)}")
        else:
            parts.append(f"[{g['token']}]")
    return " · ".join(parts)


def _pick_themes(glosses: list[dict]) -> list[str]:
    seen = []
    for tag in _THEME_PRIORITY:
        if any(g["tag"] == tag and g["known"] for g in glosses):
            seen.append(tag)
    return seen[:2]


def _neighbours(text: str, k: int = 3, floor: float = 0.35) -> list[dict]:
    """Closest corpus entries even when they are too weak to count as a match."""
    try:
        from app.core.model_registry import encode
        from app.data_loader import store

        vec = encode([text])[0]
        out = []
        for idx, sim in store.faiss_search(vec, top_k=k * 3):
            if sim < floor or len(out) >= k:
                continue
            row = store.get_row(idx)
            out.append({
                "id": str(idx),
                "konkani_text": row["konkani_text"],
                "romanized_text": row["romanized_text"],
                "english_meaning": row["english_meaning"],
                "figurative_meaning": row["figurative_meaning"],
                "category": row["category"],
                "similarity": round(float(sim), 4),
            })
        return out
    except Exception as exc:  # never let the hint break resolution
        logger.debug("Neighbour lookup failed: %s", exc)
        return []


def compose_idiom_meaning(text: str) -> dict | None:
    """Compose a provisional meaning for an idiom that is not in the corpus.

    Returns ``None`` only when the input contains no letters at all.
    """
    text = (text or "").strip()
    tokens = _tokenize(text)
    if not tokens:
        return None

    tokens, repeated = _dedupe_reduplication(tokens)
    glosses = _merge_compounds([LEXICON.lookup(t) for t in tokens])

    known = [g for g in glosses if g["known"]]
    content = [g for g in glosses if g["pos"] not in FUNCTION_POS]
    known_content = [g for g in content if g["known"]]
    negations = [g for g in glosses if g["pos"] in NEGATION_POS]
    approximate = [g for g in known if g["method"] == "embedding"]
    unglossed = [g for g in glosses if not g["known"]]

    coverage = len(known) / len(glosses)
    themes = _pick_themes(glosses)

    verbs = [g for g in known_content if g["pos"] == "verb"]
    things = [g for g in known_content if g["pos"] != "verb"]
    action = verbs[-1] if verbs else None  # Konkani is verb-final

    literal = _literal_sequence(glosses)
    theme_clause = _EN_THEME[themes[0]] if themes else ""
    action_lemma = _verb_lemma(action["english"]) if action is not None else ""
    things_text = _things_text(things)
    verb_lemmas = [_verb_lemma(v["english"]) for v in verbs]

    connectives = [g for g in glosses if g["pos"] == "connective"]
    looks_like_sentence = bool(connectives) and len(glosses) >= 5
    sentence_hint = " (the input reads like an ordinary sentence, not a proverb)" \
        if looks_like_sentence else ""

    # ── one-line reading ────────────────────────────────────────────────────
    lead = f"Provisional reading ({len(known)}/{len(glosses)} words recognised)"
    if len(verbs) >= 2:
        # Several actions: describe instead of forcing one English clause.
        phrase = (
            f"the words name {things_text or 'several things'} with the actions "
            + ", ".join(f"\u201c{v}\u201d" for v in verb_lemmas)
        )
    elif action is not None and things:
        phrase = f"to {action_lemma} {things_text}"
    elif action is not None:
        phrase = f"to {action_lemma}"
    elif things:
        phrase = f"a phrase naming {things_text}"
    else:
        phrase = f"\u201c{literal}\u201d"

    if known and theme_clause:
        english = f"{lead}: {phrase} \u2014 a saying about {theme_clause}{sentence_hint}."
    elif known:
        english = (
            f"{lead}: {phrase} \u2014 too little of the phrase is recognised to "
            f"name a reliable theme{sentence_hint}."
        )
    else:
        english = (
            "None of these words is in the Konkani glossary yet, so no meaning "
            "could be composed. Add the words or submit the proverb to the corpus."
        )

    # ── fuller paragraph ────────────────────────────────────────────────────
    sentences = [f"Word by word the phrase reads: \u201c{literal}\u201d."]

    if action is not None and things:
        acting = f"acting on {things_text}" if things_text else "with no object named"
        sentences.append(
            f"The action is \u201c{action_lemma}\u201d"
            + (f" (compound: \u201c{_primary_gloss(action)}\u201d)" if action.get("merged_from") else "")
            + f", {acting}."
        )
    elif action is not None:
        sentences.append(f"The action is \u201c{action_lemma}\u201d.")
    elif things:
        sentences.append(
            "No action word was recognised; the phrase names "
            f"{things_text}."
        )

    if theme_clause:
        sentences.append(
            f"Grouped by meaning, those words belong to {theme_clause}. That is the "
            f"frame someone would use the proverb in \u2014 an image of that kind is "
            f"normally turned into a lesson about the same domain."
        )

    if negations:
        marks = ", ".join(f"\u201c{n['token']}\u201d" for n in negations)
        sentences.append(
            f"The negative marker {marks} reverses the sense, so the phrase denies "
            f"or forbids the action rather than recommending it."
        )

    if repeated:
        sentences.append(
            "A word is repeated, which in Konkani intensifies the idea (\u201ctrue "
            "true\u201d = \u201cabsolutely true\u201d)."
        )

    if approximate:
        words = ", ".join(f"\u201c{a['token']}\u201d\u2248{a['english']}" for a in approximate)
        sentences.append(
            f"Approximate glosses (cross-lingual embedding, not dictionary entries): {words}."
        )

    if unglossed:
        words = ", ".join(f"\u201c{u['token']}\u201d" for u in unglossed[:8])
        more = " and others" if len(unglossed) > 8 else ""
        sentences.append(
            f"Still unrecognised, so left untranslated: {words}{more}. "
            f"Those words are what limit the accuracy of this reading."
        )

    if looks_like_sentence:
        sentences.append(
            "The input also contains a connective and reads like an ordinary "
            "sentence rather than a proverb; for a sentence only this word-by-word "
            "reading applies."
        )

    sentences.append(
        "This is a computed reading of the words, not an attested meaning: the "
        "proverb is not in the corpus, so confirm it with a native speaker before "
        "citing it."
    )
    figurative = " ".join(sentences)

    # ── Marathi ────────────────────────────────────────────────────────────
    mr_parts = [
        (("≈" if g["method"] == "embedding" else "") + g["marathi"]) if g["known"]
        else f"[{g['token']}]"
        for g in glosses
    ]
    mr_theme = _MR_THEME[themes[0]] if themes else ""
    marathi = f"शब्दार्थ: {' · '.join(mr_parts)}."
    if mr_theme:
        marathi += f" यातील शब्द {mr_theme} या विषयाशी जोडले जातात."
    marathi += " (हा शब्दार्थांवरून काढलेला संभाव्य अर्थ आहे — शब्दकोशातील नोंद नाही.)"

    # ── confidence ─────────────────────────────────────────────────────────
    if glosses:
        base = sum(_word_confidence(g) for g in glosses) / len(glosses)
    else:
        base = 0.0
    confidence = round(min(COMPOSITION_CEILING, base * (0.55 + 0.45 * coverage)), 4)

    return {
        "matched": True,
        "match_type": "compositional",
        "provisional": True,
        "confidence": confidence,
        "konkani_text": text,
        "romanized_text": "",
        "literal_meaning": literal,
        "figurative_meaning": figurative,
        "english_meaning": english,
        "marathi_meaning": marathi,
        "example_sentence": "",
        "category": "",
        "cultural_context": "",
        "word_glosses": glosses,
        "coverage": {
            "glossed": len(known),
            "total": len(glosses),
            "ratio": round(coverage, 4),
        },
        "themes": themes,
        "negated": bool(negations),
        "closest_known": _neighbours(text),
        "notes": [
            "Provisional meaning composed word by word; not a corpus match.",
            "Unglossed words are shown in square brackets and are excluded from the reading.",
            *(
                ["Input looks like an ordinary sentence rather than a proverb."]
                if looks_like_sentence else []
            ),
        ],
    }
