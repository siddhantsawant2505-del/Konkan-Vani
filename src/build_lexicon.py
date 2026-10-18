"""
Konkan Vani — Word Gloss Lexicon Builder
========================================
Builds ``data/lexicon/konkani_lexicon.csv`` — a Konkani -> English/Marathi
word gloss table used by the compositional-meaning fallback
(``server-nlp/app/core/compose.py``).

Why a separate lexicon is needed
--------------------------------
The idiom dataset stores *whole-idiom* meanings only. When an unknown
proverb arrives, the system must still explain it word by word, which
requires a word-level gloss source. This script assembles one from two
transparent layers:

  1. ``pattern bank``  — every NOUN/VERB/OUTCOME triple in
     ``src/rebuild_dataset.py`` is already trilingual (Devanagari /
     Roman / English), so those glosses are inherited verbatim.
  2. ``curated``       — a hand-written table of function words, body
     parts, nature, food, kinship and verb forms that actually occur in
     the corpus. Only words whose meaning is certain are included; words
     the authors were not confident about are deliberately left out so
     the runtime marks them "unknown" rather than inventing a gloss.

Schema
------
  devanagari, roman, phonetic_key, english_gloss, marathi_gloss, pos, semantic_tag, source

``semantic_tag`` drives the theme inference in ``compose.py``
(body, water, food, animal, nature, place, person, emotion, quality,
speech, motion, perception, negation, intensifier, connective, auxiliary,
quantifier, time, ritual).

Usage:
    python -X utf8 src/build_lexicon.py
"""

import os
import re
import sys
import csv
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

# Import the pattern bank without running the rebuilder's main()
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import rebuild_dataset as rb  # noqa: E402

OUT_PATH = os.path.join("data", "lexicon", "konkani_lexicon.csv")
COLUMNS = ["devanagari", "roman", "phonetic_key", "english_gloss",
           "marathi_gloss", "pos", "semantic_tag", "source"]


# ── Identical phonetic rules to app/core/phonetic_normalize.py ───────────────
def phonetic_key(roman: str) -> str:
    if not roman:
        return ""
    norm = roman.lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"),
                 ("ph", "f"), ("v", "w"), ("zh", "j")):
        norm = norm.replace(a, b)
    norm = re.sub(r"[^a-z\s]", "", norm)
    return " ".join(norm.split())


def clean_gloss(text: str) -> str:
    """Drop leading articles so glosses compose into sentences cleanly."""
    t = text.strip()
    for art in ("the ", "to ", "a ", "an "):
        if t.lower().startswith(art) and len(t) > len(art) + 2:
            return t[len(art):]
    return t


# ── Curated gloss table ──────────────────────────────────────────────────────
# (devanagari, roman, english, marathi, pos, semantic_tag)
CURATED = [
    # ── grammar / function words ────────────────────────────────────────────
    ("म्हणजे", "mhanje", "means / that is to say", "म्हणजे", "connective", "connective"),
    ("आनी", "aani", "and", "आणि", "connective", "connective"),
    ("जिल्यार", "jilyar", "if / when", "जर", "connective", "connective"),
    ("एक", "ek", "one / a single", "एक", "quantifier", "quantifier"),
    ("दोनूय", "donui", "both", "दोन्ही", "quantifier", "quantifier"),
    ("सर्वात", "sarvaat", "above all / most of all", "सर्वात", "quantifier", "quantifier"),
    ("कुठली", "kuthli", "some / any", "कोणती", "quantifier", "quantifier"),
    ("तितलें", "titlem", "that much", "तेवढे", "quantifier", "quantifier"),
    ("मोठो", "mothho", "big / great", "मोठा", "adjective", "quality"),
    ("मोठं", "motham", "great / large", "मोठे", "adjective", "quality"),
    ("लांब", "laamb", "far / long", "लांब", "adjective", "quality"),
    ("हलके", "halke", "lightly / gently", "हलके", "adverb", "quality"),
    ("खरें", "kharem", "true / genuine", "खरे", "adjective", "quality"),
    ("खरे", "khare", "true / genuine", "खरे", "adjective", "quality"),
    ("गोड", "god", "sweet / pleasant", "गोड", "adjective", "quality"),
    ("ताजे", "taaje", "fresh", "ताजे", "adjective", "quality"),
    ("ओले", "oele", "wet / damp", "ओले", "adjective", "quality"),
    ("उबदार", "ubadaar", "warm / cosy", "उबदार", "adjective", "quality"),
    ("तेजस्वी", "tejasvi", "radiant / bright", "तेजस्वी", "adjective", "quality"),
    ("हिरव्यागार", "hirvyaagar", "lush green", "हिरवागार", "adjective", "quality"),
    ("सोनेरी", "soneri", "golden", "सोनेरी", "adjective", "quality"),
    ("योग्य", "yogya", "proper / fitting", "योग्य", "adjective", "quality"),
    ("अंतिम", "antim", "final", "अंतिम", "adjective", "quality"),
    ("साधू", "saadhu", "ascetic / holy man", "साधू", "noun", "person"),
    ("भोंदू", "bhoodu", "fraud / impostor", "भोंदू", "noun", "person"),
    ("आपलें", "aplem", "one's own", "आपले", "quantifier", "quality"),
    ("आहे", "aahe", "is / there is", "आहे", "auxiliary", "auxiliary"),
    ("असता", "asta", "is / happens", "असते", "auxiliary", "auxiliary"),
    ("आसा", "aasa", "is", "आहे", "auxiliary", "auxiliary"),
    ("हवे", "have", "is needed / should", "हवे", "auxiliary", "auxiliary"),
    ("हवा", "havaa", "is needed / should", "हवा", "auxiliary", "auxiliary"),
    ("पाहिजे", "pahije", "must / ought to", "पाहिजे", "auxiliary", "auxiliary"),
    # negation
    ("न्हय", "nhay", "is not", "नाही", "negation", "negation"),
    ("नाही", "nahi", "is not", "नाही", "negation", "negation"),
    ("ना", "na", "not / no", "नाही", "negation", "negation"),
    ("नात", "naat", "not / is not", "नाही", "negation", "negation"),
    ("नका", "naka", "do not / must not", "नको", "negation", "negation"),
    ("न", "n", "not", "न", "negation", "negation"),
    # ── body ────────────────────────────────────────────────────────────────
    ("हात", "haat", "hand", "हात", "noun", "body"),
    ("हाताक", "haatak", "to the hand", "हाताला", "noun", "body"),
    ("पाय", "paay", "foot / leg", "पाय", "noun", "body"),
    ("पायां", "payam", "feet", "पायां", "noun", "body"),
    ("मुंडो", "mudo", "head", "डोके", "noun", "body"),
    ("डोळे", "dolle", "eyes", "डोळे", "noun", "body"),
    ("डोळ्यांत", "dollyant", "in the eyes", "डोळ्यांत", "noun", "body"),
    ("कान", "kaan", "ear", "कान", "noun", "body"),
    ("कानाक", "kanak", "to the ear", "कानाला", "noun", "body"),
    ("पोट", "pot", "stomach / belly", "पोट", "noun", "body"),
    ("पाठीशी", "paatshi", "at the back / in support", "पाठीशी", "noun", "body"),
    ("कुशीत", "kushit", "in the lap / side", "कुशीत", "noun", "body"),
    ("चव", "chav", "taste", "चव", "noun", "food"),
    # ── water / nature ──────────────────────────────────────────────────────
    ("पाणी", "paani", "water", "पाणी", "noun", "water"),
    ("पाण्यात", "paanyat", "in the water", "पाण्यात", "noun", "water"),
    ("पाण्याच्या", "panyachi", "of the water", "पाण्याच्या", "noun", "water"),
    ("पाण्यावर", "panyavar", "on the water", "पाण्यावर", "noun", "water"),
    ("उदक", "udak", "water", "पाणी", "noun", "water"),
    ("उदकाचो", "udcacho", "of the water", "पाण्याचा", "noun", "water"),
    ("व्हाळेर", "valor", "overflow / flood", "पूर", "noun", "water"),
    ("खळखळ", "khalkhal", "gurgling sound of flowing water", "खळखळ", "noun", "water"),
    ("माती", "maati", "soil / earth", "माती", "noun", "nature"),
    ("मातीत", "maatit", "in the soil", "मातीत", "noun", "nature"),
    ("फुल", "phul", "flower", "फुल", "noun", "nature"),
    ("फुलांना", "phulanna", "to the flowers", "फुलांना", "noun", "nature"),
    ("कळी", "kali", "bud", "कळी", "noun", "nature"),
    ("कोंब", "komb", "sprout / shoot", "कोंब", "noun", "nature"),
    ("रोप", "rop", "sapling", "रोप", "noun", "nature"),
    ("झाड", "zhad", "tree", "झाड", "noun", "nature"),
    ("झुडूपात", "zhudupaat", "in the bush", "झुडुपात", "noun", "nature"),
    ("पाऊस", "paus", "rain", "पाऊस", "noun", "nature"),
    ("पावसाळ्यात", "paavsaalyaat", "in the rainy season", "पावसाळ्यात", "noun", "time"),
    ("उन्हाळ्यात", "unhaalyaat", "in the hot season", "उन्हाळ्यात", "noun", "time"),
    ("हिवाळ्यात", "hivaalyaat", "in the cold season", "हिवाळ्यात", "noun", "time"),
    ("ऊन", "un", "sunshine / heat", "ऊन", "noun", "nature"),
    ("सुर्य", "surya", "sun", "सूर्य", "noun", "nature"),
    ("सूर्याला", "suryala", "to the sun", "सूर्याला", "noun", "nature"),
    ("चंद्र", "chondr", "moon", "चंद्र", "noun", "nature"),
    ("तारे", "tare", "stars", "तारे", "noun", "nature"),
    ("तार्यांकडे", "taaryaakade", "towards the stars", "ताऱ्यांकडे", "noun", "nature"),
    ("वारो", "waro", "wind", "वारा", "noun", "nature"),
    ("वाऱ्यासोबत", "vaaryasobat", "along with the wind", "वाऱ्यासोबत", "noun", "nature"),
    ("सागर", "sagar", "sea", "समुद्र", "noun", "water"),
    ("सागरात", "saagarat", "in the sea", "समुद्रात", "noun", "water"),
    ("सागरी", "saagari", "at the sea", "समुद्री", "noun", "water"),
    ("नदी", "nadi", "river", "नदी", "noun", "water"),
    ("नदीत", "nadit", "in the river", "नदीत", "noun", "water"),
    ("नदीला", "nadilaa", "to the river", "नदीला", "noun", "water"),
    ("तालें", "talem", "lake", "तळे", "noun", "water"),
    ("तळ्याच्या", "talyachya", "of the lake", "तळ्याच्या", "noun", "water"),
    ("ढग", "dhag", "cloud", "ढग", "noun", "nature"),
    ("ढगांमध्ये", "dhagaanmadhye", "among the clouds", "ढगांमध्ये", "noun", "nature"),
    ("धूळ", "dhool", "dust", "धूळ", "noun", "nature"),
    ("छाया", "chaayaa", "shade", "सावली", "noun", "nature"),
    ("बिजूरी", "bijuri", "lightning", "वीज", "noun", "nature"),
    ("उब", "ub", "warmth", "उब", "noun", "nature"),
    ("मासो", "maso", "fish", "मासा", "noun", "animal"),
    ("मासे", "maase", "fish (plural)", "मासे", "noun", "animal"),
    ("मास", "maas", "fish", "मासा", "noun", "animal"),
    ("गाय", "gaay", "cow", "गाय", "noun", "animal"),
    ("दुधारी", "dudhari", "milch (giving milk)", "दुधाळ", "adjective", "animal"),
    ("मांजर", "manjar", "cat", "मांजर", "noun", "animal"),
    ("मांजराक", "mazrak", "to the cat", "मांजराला", "noun", "animal"),
    ("मांजरासारखा", "mazrasarkha", "like a cat", "मांजरासारखा", "adjective", "animal"),
    ("सुणो", "suno", "dog", "कुत्रा", "noun", "animal"),
    ("साप", "saap", "snake", "साप", "noun", "animal"),
    ("वाघ", "vaagh", "tiger", "वाघ", "noun", "animal"),
    ("कोंबो", "kombo", "rooster / chick", "पिल्लू", "noun", "animal"),
    ("कोंबा", "komba", "rooster / chick", "पिल्लू", "noun", "animal"),
    ("कोंब्याक", "kombyak", "to the chick", "पिल्लाला", "noun", "animal"),
    ("कोंब्याला", "kombyala", "to the chick", "पिल्लाला", "noun", "animal"),
    ("सुकणें", "suknem", "bird", "पक्षी", "noun", "animal"),
    ("सुकण्यान", "suknyan", "bird (ergative form)", "पक्ष्याने", "noun", "animal"),
    ("माशी", "maashi", "fly (insect)", "माशी", "noun", "animal"),
    ("मोरांक", "morank", "to the peacocks", "मोरांना", "noun", "animal"),
    # ── food / livelihood ───────────────────────────────────────────────────
    ("भात", "bhat", "rice", "भात", "noun", "food"),
    ("भाताच्या", "bhatyachi", "of the rice", "भाताच्या", "noun", "food"),
    ("पोळी", "poli", "flatbread", "पोळी", "noun", "food"),
    ("पोळीत", "polit", "in the flatbread", "पोळीत", "noun", "food"),
    ("दूध", "doodh", "milk", "दूध", "noun", "food"),
    ("तूप", "toop", "ghee", "तूप", "noun", "food"),
    ("नून", "noon", "salt", "मीठ", "noun", "food"),
    ("मीठ", "mith", "salt", "मीठ", "noun", "food"),
    ("मिठा", "mitha", "salt (oblique form)", "मिठाचा", "noun", "food"),
    ("तेल", "tel", "oil", "तेल", "noun", "food"),
    ("भाजी", "bhaji", "vegetable dish", "भाजी", "noun", "food"),
    ("भाज्यासारखे", "bhajyasarakhe", "like a vegetable", "भाजीसारखे", "adjective", "food"),
    ("आंबो", "ambo", "mango", "आंबा", "noun", "food"),
    ("आंब्याच्या", "aambyaachi", "of the mango", "आंब्याच्या", "noun", "food"),
    ("अन्न", "anna", "food / grain", "अन्न", "noun", "food"),
    ("जेवण", "jevan", "meal", "जेवण", "noun", "food"),
    ("प्रसाद", "prasaad", "blessed offering (prasad)", "प्रसाद", "noun", "ritual"),
    ("मेथी", "methi", "fenugreek", "मेथी", "noun", "food"),
    ("मसाले", "masale", "spices", "मसाले", "noun", "food"),
    ("संभार", "sambhaar", "ingredients / provisions", "साहित्य", "noun", "food"),
    ("हळद", "halad", "turmeric", "हळद", "noun", "food"),
    ("हळदी", "haladi", "of turmeric", "हळदी", "noun", "food"),
    ("कापूस", "kaapus", "cotton", "कापूस", "noun", "food"),
    # ── places / objects ────────────────────────────────────────────────────
    ("घर", "ghar", "house / home", "घर", "noun", "place"),
    ("घराचेर", "ghoracher", "on the house", "घरावर", "noun", "place"),
    ("गांव", "gaam", "village", "गाव", "noun", "place"),
    ("गावात", "gaavaat", "in the village", "गावात", "noun", "place"),
    ("गावाकडच्या", "gaavakadchya", "of the village side", "गावाकडच्या", "noun", "place"),
    ("बाजार", "bazar", "market", "बाजार", "noun", "place"),
    ("शाळा", "shala", "school", "शाळा", "noun", "place"),
    ("देवूळ", "devul", "temple", "देवूळ", "noun", "place"),
    ("देवळातला", "devlaatla", "of the temple", "देवळातला", "noun", "place"),
    ("विहीर", "vihir", "well", "विहीर", "noun", "place"),
    ("विहिरीत", "vihirrit", "in the well", "विहिरीत", "noun", "place"),
    ("बांद", "band", "field embankment", "बांध", "noun", "place"),
    ("बांदार", "bandar", "harbour / port", "बंदर", "noun", "place"),
    ("पूल", "pul", "bridge", "पूल", "noun", "place"),
    ("वाट", "wat", "path / way", "वाट", "noun", "place"),
    ("तारवां", "tarwam", "boats", "नावा", "noun", "place"),
    ("तारवांचेर", "tarvancheir", "on the boats", "नावांवर", "noun", "place"),
    ("कुंभार", "kumbhaar", "potter", "कुंभार", "noun", "person"),
    ("कुंभाराच्या", "kumbharyachya", "of the potter", "कुंभाराच्या", "noun", "person"),
    ("भांडे", "bhande", "pot / vessel", "भांडे", "noun", "place"),
    ("भांड्यासारखे", "bhandyasarkhe", "like a pot", "भांड्यासारखे", "adjective", "place"),
    ("कुलूप", "kulup", "lock", "कुलूप", "noun", "place"),
    ("दिवो", "divo", "lamp", "दिवा", "noun", "ritual"),
    ("दिवा", "divaa", "lamp", "दिवा", "noun", "ritual"),
    ("आरतीचा", "aarateechea", "of the aarti lamp", "आरतीचा", "noun", "ritual"),
    ("ज्योतीत", "jyotit", "in the flame", "ज्योतीत", "noun", "ritual"),
    ("विंदी", "vindi", "dot / rangoli mark", "रांगोळी", "noun", "ritual"),
    ("मांडव", "mandav", "wedding pandal", "मांडव", "noun", "ritual"),
    ("खरें", "kharem", "fishing net", "जाळे", "noun", "place"),
    ("कोलंबीत", "kolambit", "in the pigeon coop", "कबुतरखान्यात", "noun", "place"),
    ("उंबरा", "umbra", "threshold", "उंबरा", "noun", "place"),
    ("आळ्याच्या", "aalyachya", "of the lane", "गल्लीच्या", "noun", "place"),
    # ── people / kinship ────────────────────────────────────────────────────
    ("माणूस", "manus", "person / human being", "माणूस", "noun", "person"),
    ("माणसाला", "mansala", "to the person", "माणसाला", "noun", "person"),
    ("मनशान", "manshan", "person (ergative form)", "माणसाने", "noun", "person"),
    ("बायल", "bayl", "woman", "स्त्री", "noun", "person"),
    ("भुरगें", "bhurgem", "child", "मूल", "noun", "person"),
    ("मुलांना", "mulaanna", "to the children", "मुलांना", "noun", "person"),
    ("आई", "aai", "mother", "आई", "noun", "person"),
    ("आईचा", "aaichee", "of the mother", "आईचा", "noun", "person"),
    ("वडील", "vadil", "father / elder", "वडील", "noun", "person"),
    ("वडिलांचे", "vadilaanche", "of the parents/elders", "वडिलांचे", "noun", "person"),
    ("भावाभावी", "bhaavabhaavi", "brother with brother", "भावाभावात", "noun", "person"),
    ("कुटुंब", "kutumb", "family", "कुटुंब", "noun", "person"),
    ("नात", "nant", "relation / kin", "नाते", "noun", "person"),
    ("गिरेस्त", "girest", "wealthy man", "श्रीमंत", "noun", "person"),
    ("गरीब", "garib", "poor man", "गरीब", "noun", "person"),
    ("पोट्यार", "potyaar", "boatman", "नावाडी", "noun", "person"),
    # ── abstract qualities ──────────────────────────────────────────────────
    ("शांती", "shanti", "peace", "शांती", "noun", "emotion"),
    ("आनंद", "anand", "joy", "आनंद", "noun", "emotion"),
    ("दुख्ख", "dukkh", "sorrow", "दुःख", "noun", "emotion"),
    ("भलें", "bhalen", "good / well-being", "कल्याण", "noun", "emotion"),
    ("वायट", "wayt", "bad / evil", "वाईट", "noun", "emotion"),
    ("यश", "yash", "success", "यश", "noun", "emotion"),
    ("अपेस", "apes", "failure", "अपयश", "noun", "emotion"),
    ("सन्मान", "sanman", "respect / honour", "सन्मान", "noun", "emotion"),
    ("अपमान", "apman", "humiliation", "अपमान", "noun", "emotion"),
    ("नामना", "namna", "fame", "कीर्ती", "noun", "emotion"),
    ("मोल", "mol", "worth / value", "मोल", "noun", "quality"),
    ("फायदो", "faydo", "profit / benefit", "फायदा", "noun", "quality"),
    ("नुकसान", "nukasan", "loss", "नुकसान", "noun", "quality"),
    ("त्रास", "tras", "trouble / hardship", "त्रास", "noun", "emotion"),
    ("गोंधळ", "gondhal", "turmoil / confusion", "गोंधळ", "noun", "emotion"),
    ("विश्वास", "vishwaas", "trust / faith", "विश्वास", "noun", "emotion"),
    ("मदत", "madat", "help", "मदत", "noun", "emotion"),
    ("मेहनत", "mehnat", "hard work", "मेहनत", "noun", "quality"),
    ("शिक्षण", "shikshan", "education", "शिक्षण", "noun", "quality"),
    ("आरोग्य", "aarogy", "health", "आरोग्य", "noun", "quality"),
    ("धन", "dhan", "wealth", "धन", "noun", "quality"),
    ("पैसा", "paisaa", "money", "पैसा", "noun", "quality"),
    ("काम", "kaam", "work", "काम", "noun", "quality"),
    ("व्यापारात", "vyaapaaraat", "in business / trade", "व्यापारात", "noun", "quality"),
    ("नोकरीचा", "nokreecha", "of the job", "नोकरीचा", "noun", "quality"),
    ("आदर", "aadar", "respect", "आदर", "noun", "emotion"),
    ("आशीर्वाद", "aashirvaad", "blessing", "आशीर्वाद", "noun", "ritual"),
    ("शब्द", "shabd", "word / promise", "शब्द", "noun", "speech"),
    ("वेळ", "vel", "time", "वेळ", "noun", "time"),
    ("वेळेवर", "velevar", "on time", "वेळेवर", "noun", "time"),
    ("शेवटाला", "shevtala", "in the end / finally", "शेवटी", "adverb", "time"),
    ("सुसेगाद", "susegad", "peaceful, unhurried contentment", "शांत समाधान", "noun", "emotion"),
    ("उदर्गमी", "udargami", "progress / advancement", "प्रगती", "noun", "quality"),
    ("शरीर", "shareer", "body", "शरीर", "noun", "body"),
    ("योग", "yoga", "yoga / discipline", "योग", "noun", "quality"),
    ("उत्सव", "utsav", "festival", "उत्सव", "noun", "ritual"),
    ("सण", "san", "festival", "सण", "noun", "ritual"),
    ("जात्रो", "jatro", "temple fair procession", "जत्रा", "noun", "ritual"),
    ("अग्नी", "agni", "fire", "अग्नी", "noun", "nature"),
    ("सुक्तोच", "suktoch", "dries up / empties", "सुकते", "verb", "motion"),
    ("कोल्लटा", "collta", "flows down / spills", "वाहते", "verb", "motion"),
    ("बरें", "borem", "good / well", "बरे", "adjective", "quality"),
    ("दिसता", "dista", "seems / looks", "दिसते", "verb", "perception"),
    ("शिंक", "shink", "sneeze", "शिंक", "noun", "body"),
    ("कोंदळ", "kondal", "tangle / clutter", "गोंधळ", "noun", "emotion"),
    ("मळ", "mal", "dirt / grime", "घाण", "noun", "quality"),
    ("गाठप", "gaathap", "to reach", "गाठणे", "verb", "motion"),
    # ── verbs ───────────────────────────────────────────────────────────────
    ("करा", "kara", "do (polite imperative)", "करा", "verb", "motion"),
    ("करप", "karap", "to do", "करणे", "verb", "motion"),
    ("करू", "karu", "let us do / will do", "करू", "verb", "motion"),
    ("करणे", "karane", "to do", "करणे", "verb", "motion"),
    ("जावप", "javap", "to become", "होणे", "verb", "motion"),
    ("जाता", "jata", "goes / becomes", "जाते", "verb", "motion"),
    ("येवप", "yevap", "to come", "येणे", "verb", "motion"),
    ("येत", "yete", "comes", "येते", "verb", "motion"),
    ("येता", "yeta", "comes", "येते", "verb", "motion"),
    ("आला", "aala", "came", "आला", "verb", "motion"),
    ("वचप", "vachap", "to go", "जाणे", "verb", "motion"),
    ("गेलें", "gele", "went / gone", "गेले", "verb", "motion"),
    ("गेले", "gele", "went / gone", "गेले", "verb", "motion"),
    ("जायला", "jaayla", "in order to go", "जायला", "verb", "motion"),
    ("व्हा", "vhaa", "become (imperative)", "हो", "verb", "motion"),
    ("बनवप", "banavap", "to make", "बनवणे", "verb", "motion"),
    ("बनून", "banun", "having become", "बनून", "verb", "motion"),
    ("धरप", "dharap", "to hold", "धरणे", "verb", "motion"),
    ("सोडप", "sodap", "to leave / release", "सोडणे", "verb", "motion"),
    ("दिवप", "divop", "to give", "देणे", "verb", "speech"),
    ("द्या", "dyaa", "give (imperative)", "द्या", "verb", "speech"),
    ("दाखवप", "dakhavap", "to show", "दाखवणे", "verb", "speech"),
    ("दाखवून", "dakhvun", "having shown", "दाखवून", "verb", "speech"),
    ("सांगप", "sangap", "to tell", "सांगणे", "verb", "speech"),
    ("सांगणे", "sangane", "to tell", "सांगणे", "verb", "speech"),
    ("बोलप", "bolap", "to speak", "बोलणे", "verb", "speech"),
    ("आयकप", "aaykap", "to listen", "ऐकणे", "verb", "speech"),
    ("पळोवप", "palovap", "to watch / look at", "पाहणे", "verb", "perception"),
    ("खावप", "khavap", "to eat", "खाणे", "verb", "food"),
    ("पियेवप", "piyevap", "to drink", "पिणे", "verb", "food"),
    ("पिऊन", "pionn", "having drunk", "पिऊन", "verb", "food"),
    ("विसर", "visor", "forget", "विसर", "verb", "perception"),
    ("बसप", "bosap", "to sit", "बसणे", "verb", "motion"),
    ("बसून", "bosun", "having sat", "बसून", "verb", "motion"),
    ("रावप", "ravop", "to stay / remain", "राहणे", "verb", "motion"),
    ("राहा", "rahaa", "stay (imperative)", "रहा", "verb", "motion"),
    ("रहा", "rahaa", "stay (imperative)", "रहा", "verb", "motion"),
    ("घेवप", "ghevap", "to take", "घेणे", "verb", "motion"),
    ("घालप", "ghalap", "to put / wear", "घालणे", "verb", "motion"),
    ("घालून", "ghalun", "having put", "घालून", "verb", "motion"),
    ("घालचें", "ghalchem", "has to be put", "घालायचे", "verb", "motion"),
    ("मारप", "marap", "to hit / strike", "मारणे", "verb", "motion"),
    ("मारप", "maarap", "to hit / strike", "मारणे", "verb", "motion"),
    ("मारून", "maarun", "having hit / doing thoroughly", "मारून", "verb", "motion"),
    ("लावप", "lavap", "to apply / plant", "लावणे", "verb", "motion"),
    ("लागप", "laap", "to be needed / take effect", "लागणे", "verb", "motion"),
    ("लागतो", "lagto", "is needed / takes", "लागतो", "verb", "motion"),
    ("लावण्याचा", "laavanyaacha", "of planting", "लावण्याचा", "verb", "motion"),
    ("ठेवप", "thevap", "to keep", "ठेवणे", "verb", "motion"),
    ("ठेवा", "thevaa", "keep (imperative)", "ठेवा", "verb", "motion"),
    ("राखप", "rakhap", "to keep / protect", "राखणे", "verb", "motion"),
    ("राखपाक", "rakhpak", "in order to keep", "राखण्यासाठी", "verb", "motion"),
    ("टाकप", "takap", "to put / throw", "टाकणे", "verb", "motion"),
    ("टाका", "taka", "put / throw (imperative)", "टाका", "verb", "motion"),
    ("टाकलेली", "takleli", "having been put", "टाकलेली", "verb", "motion"),
    ("सापडप", "sapdap", "to be found", "सापडणे", "verb", "perception"),
    ("सापडलेल्याला", "sapadlyala", "to the one who found", "सापडलेल्याला", "verb", "perception"),
    ("शोधणे", "shodhane", "to search", "शोधणे", "verb", "perception"),
    ("उगवप", "ugavap", "to sprout / rise", "उगवणे", "verb", "motion"),
    ("उडप", "udap", "to fly", "उडणे", "verb", "motion"),
    ("उडता", "udta", "flies", "उडतो", "verb", "motion"),
    ("हलप", "halap", "to move / shake", "हलणे", "verb", "motion"),
    ("पाडप", "padap", "to make fall / drop", "पाडणे", "verb", "motion"),
    ("फुटप", "futap", "to break / split", "फुटणे", "verb", "motion"),
    ("फोडप", "fodap", "to break open", "फोडणे", "verb", "motion"),
    ("भांडप", "bhandap", "to quarrel", "भांडणे", "verb", "speech"),
    ("शिंपडप", "shimpdap", "to sprinkle", "शिंपडणे", "verb", "motion"),
    ("शिंपडतात", "shimpadtaat", "they sprinkle", "शिंपडतात", "verb", "motion"),
    ("मापप", "maapap", "to measure", "मोजणे", "verb", "motion"),
    ("धुणप", "dhunap", "to wash", "धुणे", "verb", "motion"),
    ("जिंकप", "jinkap", "to win / conquer", "जिंकणे", "verb", "motion"),
    ("जिखलो", "jikhlo", "conquered", "जिंकले", "verb", "motion"),
    ("हांसप", "hansap", "to laugh", "हसणे", "verb", "emotion"),
    ("रडप", "radap", "to weep", "रडणे", "verb", "emotion"),
    ("नाचप", "nachap", "to dance", "नाचणे", "verb", "motion"),
    ("गावप", "gavap", "to sing", "गाणे", "verb", "speech"),
    ("चलप", "cholap", "to walk", "चालणे", "verb", "motion"),
    ("धांवप", "dhavap", "to run", "धावणे", "verb", "motion"),
    ("शिकप", "shikap", "to learn", "शिकणे", "verb", "perception"),
    ("वाचप", "vachap", "to read", "वाचणे", "verb", "perception"),
    ("बरोवप", "barovap", "to write", "लिहिणे", "verb", "speech"),
    ("लिहिलेले", "lihilele", "written", "लिहिलेले", "verb", "speech"),
    ("बांदप", "bandap", "to build", "बांधणे", "verb", "motion"),
    ("आणणे", "aanaane", "to bring", "आणणे", "verb", "motion"),
    ("आणणे", "aanaane", "to bring", "आणणे", "verb", "motion"),
    ("अडवू", "adavu", "to stop / block", "अडवणे", "verb", "motion"),
    ("उचला", "uchala", "lift (imperative)", "उचला", "verb", "motion"),
    ("स्नान", "snaan", "bath", "स्नान", "noun", "ritual"),
    ("मिळवा", "milavaa", "earn / obtain (imperative)", "मिळवा", "verb", "motion"),
    ("मेळटा", "melta", "is found / is obtained", "मिळते", "verb", "perception"),
    ("साजरा", "saajraa", "celebrate (imperative)", "साजरा", "verb", "ritual"),
    ("राहील", "rahil", "will remain", "राहील", "verb", "motion"),
    ("कापूस", "kaapus", "cotton", "कापूस", "noun", "food"),
    ("फळ", "fal", "fruit / result", "फळ", "noun", "food"),
    # ── perfective / finite verb forms (high frequency in real proverbs) ────
    ("धुवून", "dhuvun", "having washed", "धुवून", "verb", "motion"),
    ("धुवप", "dhuvap", "to wash", "धुवणे", "verb", "motion"),
    ("गेलो", "gelo", "went (m.)", "गेला", "verb", "motion"),
    ("गेली", "geli", "went (f.)", "गेली", "verb", "motion"),
    ("घेतली", "ghetli", "took (f.)", "घेतली", "verb", "motion"),
    ("घेतलो", "ghetlo", "took (m.)", "घेतला", "verb", "motion"),
    ("घेणे", "ghene", "to take", "घेणे", "verb", "motion"),
    ("केलो", "kelo", "did (m.)", "केला", "verb", "motion"),
    ("केली", "keli", "did (f.)", "केली", "verb", "motion"),
    ("दिलो", "dilo", "gave (m.)", "दिला", "verb", "speech"),
    ("दिली", "dili", "gave (f.)", "दिली", "verb", "speech"),
    ("जालो", "jalo", "became (m.)", "झाला", "verb", "motion"),
    ("जाली", "jali", "became (f.)", "झाली", "verb", "motion"),
    ("म्हणून", "mhanun", "saying / therefore", "म्हणून", "connective", "connective"),
    ("बोलून", "bolun", "having spoken", "बोलून", "verb", "speech"),
    ("आयकून", "aaykun", "having listened", "ऐकून", "verb", "speech"),
    ("चलून", "cholun", "having walked", "चालून", "verb", "motion"),
    ("धांवून", "dhavun", "having run", "धावून", "verb", "motion"),
    ("खाऊन", "khaun", "having eaten", "खाऊन", "verb", "food"),
    ("पळोवून", "palovun", "having seen", "पाहून", "verb", "perception"),
    ("देखून", "dekhun", "having seen", "पाहून", "verb", "perception"),
    ("उठून", "uthun", "having got up", "उठून", "verb", "motion"),
    ("फिरप", "firap", "to wander", "फिरणे", "verb", "motion"),
    ("उरप", "urap", "to remain", "उरणे", "verb", "motion"),
    # ── pronouns (kept out of the imagery pool by pos='pronoun') ───────────
    ("हांव", "hanv", "I", "मी", "pronoun", "person"),
    ("आमी", "aami", "we", "आमी", "pronoun", "person"),
    ("तूं", "tun", "you (singular)", "तू", "pronoun", "person"),
    ("तुमी", "tumi", "you (plural)", "तुम्ही", "pronoun", "person"),
    ("तो", "to", "he / that", "तो", "pronoun", "person"),
    ("ती", "ti", "she / that", "ती", "pronoun", "person"),
    ("ते", "te", "they / those", "ते", "pronoun", "person"),
    ("हो", "ho", "yes / it is", "हो", "auxiliary", "auxiliary"),
    # ── rice-cycle and body-location words (common in food proverbs) ────────
    ("तांदूळ", "tandul", "rice grain", "तांदूळ", "noun", "food"),
    ("शित", "shit", "cooked rice", "भात", "noun", "food"),
    ("पोटांत", "potant", "in the stomach", "पोटात", "noun", "body"),
]


def build_rows():
    rows, seen = [], set()

    def add(dev, rom, eng, mar, pos, tag, source):
        dev = dev.strip()
        if not dev or dev in seen:
            return
        seen.add(dev)
        rows.append({
            "devanagari": dev,
            "roman": rom.strip(),
            "phonetic_key": phonetic_key(rom),
            "english_gloss": clean_gloss(eng),
            "marathi_gloss": mar.strip(),
            "pos": pos,
            "semantic_tag": tag,
            "source": source,
        })

    # 1. pattern bank — inherits the dataset's own trilingual glosses
    for dev, rom, eng in rb.NOUNS:
        add(dev, rom, eng, dev, "noun", "place", "pattern_bank")
    for dev, rom, eng in rb.VERBS:
        add(dev, rom, eng, dev, "verb", "motion", "pattern_bank")
    for dev, rom, mar, eng, polarity in rb.OUTCOMES:
        tag = "emotion" if polarity else "quality"
        for d, r, e in zip(dev.split(), rom.split(), eng.split()):
            if len(d) > 1:
                add(d, r, e, d, "noun", tag, "pattern_bank_outcome")

    # 2. curated hand-written glosses
    for dev, rom, eng, mar, pos, tag in CURATED:
        add(dev, rom, eng, mar, pos, tag, "curated")

    # "म्हणजे ..." connective words always appear literally; make sure the
    # canonical form from OUTCOMES is glossed as a phrase, not split apart.
    for dev, rom, mar, eng, polarity in rb.OUTCOMES:
        key = " ".join(rom.split())
        for row in rows:
            if phonetic_key(row["roman"]) == key:
                row["english_gloss"] = eng
                row["marathi_gloss"] = mar
                break
    return rows


def main():
    rows = build_rows()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print("=" * 70)
    print("KONKAN VANI — WORD GLOSS LEXICON")
    print("=" * 70)
    print(f"saved            : {OUT_PATH}")
    print(f"entries          : {len(rows)}")
    print(f"unique phon keys : {len({r['phonetic_key'] for r in rows if r['phonetic_key']})}")
    from collections import Counter
    print("by source        :", dict(Counter(r["source"] for r in rows)))
    print("by semantic tag  :", dict(Counter(r["semantic_tag"] for r in rows).most_common()))
    assert len(rows) >= 250, "lexicon unexpectedly small"
    assert all(r["devanagari"] and r["english_gloss"] for r in rows), "empty gloss"
    print("\nALL VALIDATION ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
