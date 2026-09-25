"""
server-nlp/app/core/dynamic_fallback.py

Dynamic AI Fallback Resolver for unindexed or low-confidence Konkani idioms.
Uses LLM reasoning (via Gemini / API or heuristic dictionary parser) when available,
and caches results to data/processed/dynamic_idioms_cache.json.
"""

import os
import json
import logging
import re

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "dynamic_idioms_cache.json")
)


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read dynamic cache: {e}")
    return {}


def _save_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save dynamic cache: {e}")


def resolve_dynamic_fallback(query_text: str) -> dict | None:
    """
    Attempt to resolve an unindexed Konkani idiom dynamically.
    Checks local cache first, then calls AI resolver if GEMINI_API_KEY / LLM is configured.

    NOTE: This stage is OPT-IN. It can produce high-confidence answers for
    idioms that are NOT in the dataset (fabrication risk), so it only runs
    when DYNAMIC_FALLBACK_ENABLED=1 is set in the environment. Set both
    DYNAMIC_FALLBACK_ENABLED=1 and GEMINI_API_KEY=<key> to enable it.
    """
    if os.getenv("DYNAMIC_FALLBACK_ENABLED", "0") != "1":
        return None

    if not query_text or not query_text.strip():
        return None

    clean_query = query_text.strip()
    cache = _load_cache()

    # Check cache first
    if clean_query in cache:
        logger.info(f"Dynamic fallback cache hit for: '{clean_query}'")
        res = cache[clean_query].copy()
        res["match_type"] = "dynamic_ai_cached"
        return res

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY found for dynamic online fallback.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are an expert linguist specializing in Konkani language, idioms, proverbs, and Goan cultural expressions.
Analyze the following Konkani input text (in Devanagari or Romanized script):
"{clean_query}"

Provide a JSON object with EXACTLY the following structure (no markdown formatting, raw JSON only):
{{
    "matched": true,
    "konkani_text": "{clean_query}",
    "literal_meaning": "Word-for-word translation in English",
    "figurative_meaning": "Actual semantic/figurative meaning in English",
    "english_meaning": "Short concise English summary of the meaning",
    "marathi_meaning": "Simple explanation in Marathi",
    "example_sentence": "An example sentence using this idiom in Konkani",
    "cultural_context": "Cultural or historical origin of this expression in Goa/Konkan",
    "category": "Idiom Category (e.g., Deception & Fraud, Character & Integrity, Wisdom & Life Lessons)"
}}
"""
        response = model.generate_content(prompt)
        text_content = response.text.strip()
        text_content = re.sub(r"^```json\s*", "", text_content)
        text_content = re.sub(r"^```\s*", "", text_content)
        text_content = re.sub(r"\s*```$", "", text_content)

        data = json.loads(text_content)
        data["match_type"] = "dynamic_ai"
        data["confidence"] = 0.95

        cache[clean_query] = data
        _save_cache(cache)

        return data
    except Exception as e:
        logger.error(f"Dynamic AI fallback failed for query '{clean_query}': {e}")
        return None
