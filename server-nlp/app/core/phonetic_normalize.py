"""
server-nlp/app/core/phonetic_normalize.py
Spelling-variant normalization into a canonical phonetic key.

Rules (applied in order):
  aa→a, ee→i, oo→u, ph→f, v→w, zh→j
  lowercase, strip non-alpha, collapse whitespace
"""

import re


def devanagari_normalize(text: str) -> str:
    """Normalize Devanagari text by removing punctuation, extra spaces, and diacritic anomalies."""
    if not isinstance(text, str):
        return ""
    # Retain Devanagari characters and whitespace, remove punctuation & symbols
    norm = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return ' '.join(norm.split())


def phonetic_normalize(text: str) -> str:
    """Return a canonical phonetic key for Romanized or Devanagari text."""
    if not isinstance(text, str):
        return ""
    
    # Check if text contains Devanagari characters
    if re.search(r'[\u0900-\u097F]', text):
        return devanagari_normalize(text)

    norm = text.lower()
    norm = re.sub(r'aa', 'a', norm)
    norm = re.sub(r'ee', 'i', norm)
    norm = re.sub(r'oo', 'u', norm)
    norm = re.sub(r'ph', 'f', norm)
    norm = re.sub(r'v',  'w', norm)
    norm = re.sub(r'zh', 'j', norm)
    norm = re.sub(r'[^a-z\s]', '', norm)
    return ' '.join(norm.split())
