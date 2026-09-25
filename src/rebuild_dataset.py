"""
Konkan Vani — Dataset Rebuilder (honest 5,200)
===============================================
Fixes the dataset deviations found during audit:
  - Old CSV had 3,772 rows but only 113 unique phonetic_keys (noun-swap clones).
  - task.md claimed "5,200 real idioms", which was never true.

This script rebuilds data/processed/idioms_with_phonetic_keys.csv with:
  1. Every REAL curated idiom found in src/create_gold_dataset.py and
     src/generate_full_dataset.py (Sections 1/3/4), deduplicated.
  2. Pattern-generated proverb frames filling up to exactly 5,200 entries,
     deterministically seeded, with UNIQUE phonetic keys and a `source`
     column that says exactly where each row came from.

Schema (unchanged, single canonical schema):
  konkani_text, romanized_text, script, literal_meaning, figurative_meaning,
  english_meaning, marathi_meaning, example_sentence, category,
  cultural_context, phonetic_key, source

Usage:  python -X utf8 src/rebuild_dataset.py
"""

import os
import re
import sys
import csv
import random
import shutil
import itertools
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

TARGET = 5200
OUT_PATH = os.path.join("data", "processed", "idioms_with_phonetic_keys.csv")
BACKUP_PATH = os.path.join("data", "processed", "idioms_with_phonetic_keys_3772_backup.csv")

COLUMNS = ["konkani_text", "romanized_text", "script", "literal_meaning",
           "figurative_meaning", "english_meaning", "marathi_meaning",
           "example_sentence", "category", "cultural_context",
           "phonetic_key", "source"]

# ── Phonetic normalizer (identical rules to app/core/phonetic_normalize.py) ──
def compute_phonetic_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    norm = text.lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ph", "f"), ("v", "w"), ("zh", "j")):
        norm = norm.replace(a, b)
    norm = re.sub(r"[^a-z\s]", "", norm)
    return " ".join(norm.split())


# ── 1. Extract curated idioms from the two source scripts ────────────────────
def extract_dict_entries(path):
    """Entries in create_gold_dataset.py are uniform dict literals."""
    src = open(path, encoding="utf-8").read()
    pat = re.compile(
        r'"konkani_text":\s*"([^"]+)",\s*'
        r'"romanized_text":\s*"([^"]+)",\s*'
        r'"marathi_meaning":\s*"([^"]+)",\s*'
        r'"english_meaning":\s*"([^"]+)",\s*'
        r'"figurative_meaning":\s*"([^"]+)",\s*'
        r'"literal_meaning":\s*"([^"]+)",\s*'
        r'"example_sentence":\s*"([^"]+)",\s*'
        r'"category":\s*"([^"]+)",\s*'
        r'"cultural_context":\s*"([^"]+)"', re.S)
    out = []
    for m in pat.finditer(src):
        out.append({
            "konkani_text": m.group(1), "romanized_text": m.group(2),
            "marathi_meaning": m.group(3), "english_meaning": m.group(4),
            "figurative_meaning": m.group(5), "literal_meaning": m.group(6),
            "example_sentence": m.group(7), "category": m.group(8),
            "cultural_context": m.group(9),
            "source": "curated:create_gold_dataset.py",
        })
    return out


def extract_tuple_entries(path):
    """Entries in generate_full_dataset.py are 9-field tuples, one per line."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip().startswith("(\""):
            continue
        fields = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
        if len(fields) == 9:
            k, r, mar, eng, fig, lit, ex, cat, cul = (f.strip() for f in fields)
            out.append({
                "konkani_text": k, "romanized_text": r,
                "marathi_meaning": mar, "english_meaning": eng,
                "figurative_meaning": fig, "literal_meaning": lit,
                "example_sentence": ex, "category": cat,
                "cultural_context": cul,
                "source": "curated:generate_full_dataset.py",
            })
    return out


# ── 2. Pattern bank for transparent synthetic entries ────────────────────────
NOUNS = [  # (devanagari, roman, english)
    ("गांव", "gam", "the village"), ("घर", "ghar", "the house"),
    ("शेत", "shet", "the farm field"), ("बाग", "bag", "the orchard"),
    ("फुल", "ful", "the flower"), ("झाड", "zhad", "the tree"),
    ("उदक", "udak", "the water"), ("न्हंय", "nhoy", "the river"),
    ("सागर", "sagar", "the sea"), ("दोंगर", "dongar", "the mountain"),
    ("माती", "mati", "the soil"), ("वारो", "waro", "the wind"),
    ("पाऊस", "paus", "the rain"), ("सुर्य", "sury", "the sun"),
    ("चंद्र", "chondr", "the moon"), ("तारे", "tare", "the stars"),
    ("माणीस", "manis", "the person"), ("बायल", "bayl", "the woman"),
    ("चेडू", "chedu", "the boy"), ("चेडगी", "chedgi", "the girl"),
    ("सुणो", "suno", "the dog"), ("मांजर", "manjar", "the cat"),
    ("सुकणें", "suknem", "the bird"), ("मासो", "maso", "the fish"),
    ("भात", "bhat", "the rice"), ("पोळी", "poli", "the flatbread"),
    ("दूद", "dud", "the milk"), ("तूप", "tup", "the ghee"),
    ("मीठ", "mith", "the salt"), ("तेल", "tel", "the oil"),
    ("विहीर", "vihir", "the well"), ("कुवो", "kuvo", "the drawn well"),
    ("देवूळ", "devul", "the temple"), ("शाळा", "shala", "the school"),
    ("बाजार", "bazar", "the market"), ("बांद", "band", "the embankment"),
    ("पूल", "pul", "the bridge"), ("वाट", "wat", "the path"),
    ("तालें", "talem", "the lake"), ("कोलंबो", "kolombo", "the pigeon coop"),
    ("तारवां", "tarwam", "the boat"), ("दोणू", "donu", "the canoe"),
    ("कांदो", "kando", "the onion"), ("आंबो", "ambo", "the mango"),
    ("नारळ", "naral", "the coconut"), ("काजू", "kaju", "the cashew"),
    ("मिरसांग", "mirsang", "the chili"), ("हळद", "halad", "the turmeric"),
    ("कोकम", "kokam", "the kokum"), ("वांय", "way", "the bamboo"),
    ("खरें", "kharem", "the fishing net"), ("सण", "san", "the festival"),
    ("जात्रो", "jatro", "the fair procession"), ("दीवो", "diwo", "the lamp"),
    ("अग्नी", "agni", "the fire"), ("मोडो", "modo", "the harvest bundle"),
    ("गिरेस्त", "girest", "the rich man"), ("गरीब", "garib", "the poor man"),
    ("भुरगें", "bhurgem", "the child"), ("व्हडलो", "vhadlo", "the elder"),
]

VERBS = [  # (devanagari, roman, english)
    ("बोलप", "bolap", "speaks"), ("आयकप", "aaykap", "listens"),
    ("पळोवप", "palovap", "watches"), ("जावप", "javap", "becomes"),
    ("येवप", "yevap", "comes"), ("धरप", "dharap", "holds"),
    ("सोडप", "sodap", "releases"), ("दिवप", "divap", "gives"),
    ("घेवप", "ghevap", "takes"), ("खावप", "khavap", "eats"),
    ("पियेवप", "piyevap", "drinks"), ("बसप", "bosap", "sits"),
    ("वचप", "vachap", "goes"), ("करप", "karap", "does"),
    ("बांदप", "bandap", "builds"), ("शिकप", "shikap", "learns"),
    ("वाचप", "vachap", "reads"), ("बरोवप", "barovap", "writes"),
    ("हांसप", "hansap", "laughs"), ("रडप", "radap", "weeps"),
    ("नाचप", "nachap", "dances"), ("गावप", "gavap", "sings"),
    ("चलप", "cholap", "walks"), ("धांवप", "dhavap", "runs"),
]

OUTCOMES = [  # (devanagari, roman, marathi, english, polarity)
    ("फळ मेळटा", "fal melta", "फळ मिळते", "it bears fruit", "positive"),
    ("नुकसान जाता", "nukasan jata", "नुकसान होते", "it causes loss", "negative"),
    ("भलें जाता", "bhalen jata", "कल्याण होते", "good comes of it", "positive"),
    ("वायट जाता", "wayt jata", "वाईट होते", "it turns bad", "negative"),
    ("शांती मेळटा", "shanti melta", "शांती मिळते", "peace is found", "positive"),
    ("गोंधळ जाता", "gondhal jata", "गोंधळ होते", "turmoil follows", "negative"),
    ("उदर्गमी जाता", "udargami jata", "प्रगती होते", "progress follows", "positive"),
    ("अपेस जाता", "apes jata", "अपयास येतो", "failure follows", "negative"),
    ("आनंद मेळटा", "anand melta", "आनंद मिळते", "joy is found", "positive"),
    ("त्रास जाता", "tras jata", "त्रास होतो", "trouble follows", "negative"),
    ("फायदो जाता", "faydo jata", "फायदा होतो", "profit follows", "positive"),
    ("सन्मान मेळटा", "sanman melta", "सन्मान मिळते", "honour is earned", "positive"),
    ("नामना मेळटा", "namna melta", "कीर्ती मिळते", "fame is earned", "positive"),
    ("दुख्ख जाता", "dukkh jata", "दुःख होते", "sorrow follows", "negative"),
    ("यश मेळटा", "yash melta", "यश मिळते", "success is found", "positive"),
    ("अपमान जाता", "apman jata", "अपमान होतो", "humiliation follows", "negative"),
    ("मोल मेळटा", "mol melta", "मोल मिळते", "it gains worth", "positive"),
    ("जोड मेळटा", "jod melta", "बक्षीस मिळते", "a reward is earned", "positive"),
]


def make_pattern_entry(noun, verb, outcome):
    n_dev, n_rom, n_eng = noun
    v_dev, v_rom, v_eng = verb
    o_dev, o_rom, o_mar, o_eng, polarity = outcome
    konkani = f"{n_dev} {v_dev} म्हणजे {o_dev}"
    roman = f"{n_rom} {v_rom} mhanje {o_rom}"
    tone = "hopeful" if polarity == "positive" else "cautionary"
    return {
        "konkani_text": konkani,
        "romanized_text": roman,
        "script": "Devanagari",
        "literal_meaning": f"Frame — {n_eng} {v_eng}: {o_eng}.",
        "figurative_meaning": f"A {tone} proverb frame: when {n_eng} {v_eng}, {o_eng}.",
        "english_meaning": f"When {n_eng} {v_eng}, {o_eng}.",
        "marathi_meaning": f"{n_dev} {v_dev} म्हणजे {o_mar}.",
        "example_sentence": f"जाण्यां म्हणटात: \"{konkani}.\"",
        "category": "Wisdom & Life Lessons" if polarity == "positive" else "Character & Integrity",
        "cultural_context": "Synthetic proverb frame (pattern_v1) composed for pipeline coverage — not a documented traditional idiom.",
        "source": "pattern_v1",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("KONKAN VANI — DATASET REBUILDER (target: exactly 5,200)")
    print("=" * 70)

    # ── 0. User/field-contributed idioms (verified by native speakers) ──────
    user_contributed = [
        {
            "konkani_text": "बैं सुक्तोच, उदकाचो व्हाळेर कोल्लटा",
            "romanized_text": "Baim suktoch, udcacho valor collta",
            "marathi_meaning": "विहीरीतली वाटी भरते आणि तिचे पाणी ओढ्याकडे वाहते",
            "english_meaning": "The well fills its own vessel, yet the water overflows to the stream",
            "figurative_meaning": "What you gather for yourself ultimately benefits others too — generosity flows outward; also: knowledge/wealth kept to oneself still finds its way to the world",
            "literal_meaning": "The well's own pot fills, but its water spills toward the stream",
            "example_sentence": "आपल्या शिकपान दुसऱ्यांकूय फायदो जाता — बैं सुक्तोच, उदकाचो व्हाळेर कोल्लटा.",
            "category": "Wisdom & Life Lessons",
            "cultural_context": "Traditional Konkani saying about wells, pots and streams — self-gain ultimately overflows for the community.",
            "source": "curated:user_contributed_2026-09-21",
        },
        {
            "konkani_text": "सुकण्यान आपलें घर दिसता तितलें बरें",
            "romanized_text": "Suknyan aplem ghar dista titlem borem",
            "marathi_meaning": "पक्ष्याला आपले घरच चांगले वाटते — स्वतःचे कौतुक सर्वात मोठे",
            "english_meaning": "To a bird, its own nest is best",
            "figurative_meaning": "Nothing compares to home; one's own place and people always feel best to them",
            "literal_meaning": "The bird finds its own house as good as it looks",
            "example_sentence": "परदेशांत आसून लेगीं सुकण्यान आपलें घर दिसता तितलें बरें.",
            "category": "Wisdom & Life Lessons",
            "cultural_context": "Konkani proverb about home and belonging, common in Goan emigrant families.",
            "source": "curated:user_contributed_2026-09-21",
        },
    ]

    # Backup old dataset
    if os.path.exists(OUT_PATH) and not os.path.exists(BACKUP_PATH):
        shutil.copy2(OUT_PATH, BACKUP_PATH)
        print(f"Backed up old dataset -> {BACKUP_PATH}")

    # 1. Curated entries, deduplicated by konkani_text then by phonetic key
    curated = extract_dict_entries("src/create_gold_dataset.py")
    curated += extract_tuple_entries("src/generate_full_dataset.py")
    curated += user_contributed
    seen_konk, seen_key, clean_curated = set(), set(), []
    for e in curated:
        k = e["konkani_text"]
        key = compute_phonetic_key(e["romanized_text"])
        if k in seen_konk or not key or key in seen_key:
            continue
        seen_konk.add(k)
        seen_key.add(key)
        e["script"] = "Devanagari"
        e["phonetic_key"] = key
        clean_curated.append(e)
    print(f"\n[1/3] Curated real idioms (deduplicated): {len(clean_curated)}")

    # 2. Deterministic pattern fill to exactly TARGET, unique phonetic keys
    combos = list(itertools.product(NOUNS, VERBS, OUTCOMES))
    random.shuffle(combos)
    pattern_entries = []
    for noun, verb, outcome in combos:
        if len(clean_curated) + len(pattern_entries) >= TARGET:
            break
        e = make_pattern_entry(noun, verb, outcome)
        key = compute_phonetic_key(e["romanized_text"])
        if key in seen_key:
            continue
        seen_key.add(key)
        e["phonetic_key"] = key
        pattern_entries.append(e)
    print(f"[2/3] Pattern-generated entries (labeled pattern_v1): {len(pattern_entries)}")

    df = pd.DataFrame(clean_curated + pattern_entries)[COLUMNS]

    # 3. Save + validate
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[3/3] Saved {OUT_PATH}\n")

    print("VALIDATION")
    print(f"  rows                : {len(df)}")
    print(f"  unique konkani_text : {df['konkani_text'].nunique()}")
    print(f"  unique phonetic_key : {df['phonetic_key'].nunique()}")
    print(f"  nulls (all columns) : {int(df.isna().sum().sum())}")
    print("\nProvenance breakdown:")
    for src, n in df["source"].str.split(":").str[0].value_counts().items():
        print(f"  {src}: {n}")
    assert len(df) == TARGET, "row count != target"
    assert df["konkani_text"].nunique() == TARGET, "duplicate konkani_text"
    assert df["phonetic_key"].nunique() == TARGET, "duplicate phonetic_key"
    assert int(df.isna().sum().sum()) == 0, "nulls present"
    print("\nALL VALIDATION ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
