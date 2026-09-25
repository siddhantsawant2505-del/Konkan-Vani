"""
Konkan Vani — Dataset Scaling & Validation Tool
================================================
Helps expand and validate the gold idiom dataset.

Usage:
    python src/scale_dataset.py validate    — Validate existing dataset schema
    python src/scale_dataset.py stats       — Show detailed statistics
    python src/scale_dataset.py categories  — Show category breakdown
    python src/scale_dataset.py export      — Export validated dataset for model training

The ideal dataset should have 5000-10000 entries with:
  - konkani_text (Devanagari Konkani)
  - romanized_text (Romi Konkani for phonetic matching)
  - marathi_meaning (simple Marathi explanation)
  - english_meaning (English explanation)
  - figurative_meaning (semantic/conceptual meaning)
  - literal_meaning (word-for-word translation)
  - example_sentence (usage in Devanagari)
  - category (thematic grouping)
  - cultural_context (cultural background)
  - script (always "Devanagari" for primary input)
  - phonetic_key (normalized for search)
"""

import os
import sys
import pandas as pd
import re

sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = os.path.join("data", "processed", "idioms_with_phonetic_keys.csv")

REQUIRED_COLUMNS = [
    "konkani_text", "romanized_text", "marathi_meaning", "english_meaning",
    "figurative_meaning", "literal_meaning", "example_sentence",
    "category", "cultural_context", "script", "phonetic_key"
]

IDEAL_MIN_ROWS = 5000
IDEAL_MAX_ROWS = 10000


def validate_dataset():
    """Validate the dataset against the ideal schema."""
    print("=" * 70)
    print("KONKAN VANI — DATASET VALIDATION")
    print("=" * 70)
    
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: Dataset not found at {CSV_PATH}")
        print("   Run src/create_gold_dataset.py first.")
        return False
    
    df = pd.read_csv(CSV_PATH)
    errors = []
    warnings = []
    
    # 1. Check required columns
    print("\n--- Schema Validation ---")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        print(f"  ❌ Missing columns: {missing_cols}")
    else:
        print(f"  ✅ All {len(REQUIRED_COLUMNS)} required columns present")
    
    if extra_cols:
        warnings.append(f"Extra columns (not in schema): {extra_cols}")
        print(f"  ⚠️  Extra columns: {extra_cols}")
    
    # 2. Check row count
    print(f"\n--- Row Count ---")
    print(f"  Current rows: {len(df)}")
    if len(df) < IDEAL_MIN_ROWS:
        warnings.append(f"Only {len(df)} rows — ideal minimum is {IDEAL_MIN_ROWS}")
        print(f"  ⚠️  Below ideal minimum ({IDEAL_MIN_ROWS}). Need {IDEAL_MIN_ROWS - len(df)} more entries.")
    elif len(df) > IDEAL_MAX_ROWS:
        print(f"  ✅ Exceeds ideal maximum ({IDEAL_MAX_ROWS})")
    else:
        print(f"  ✅ Within ideal range ({IDEAL_MIN_ROWS}–{IDEAL_MAX_ROWS})")
    
    # 3. Check for empty/missing values in required fields
    print(f"\n--- Missing Values ---")
    critical_fields = ["konkani_text", "romanized_text", "marathi_meaning", "english_meaning"]
    for field in critical_fields:
        if field in df.columns:
            missing_count = df[field].isna().sum() + (df[field] == "").sum()
            if missing_count > 0:
                errors.append(f"{field} has {missing_count} empty values")
                print(f"  ❌ {field}: {missing_count} empty values")
            else:
                print(f"  ✅ {field}: all values present")
    
    # 4. Check for duplicate konkani_text
    print(f"\n--- Duplicates ---")
    if "konkani_text" in df.columns:
        dupes = df["konkani_text"].duplicated().sum()
        if dupes > 0:
            warnings.append(f"{dupes} duplicate konkani_text entries")
            print(f"  ⚠️  {dupes} duplicate konkani_text entries")
        else:
            print(f"  ✅ No duplicate konkani_text entries")
    
    # 5. Check phonetic key quality
    print(f"\n--- Phonetic Key Quality ---")
    if "phonetic_key" in df.columns:
        empty_keys = (df["phonetic_key"].isna() | (df["phonetic_key"] == "")).sum()
        short_keys = (df["phonetic_key"].str.len() < 3).sum() - empty_keys
        if empty_keys > 0:
            errors.append(f"{empty_keys} entries with empty phonetic keys")
            print(f"  ❌ {empty_keys} entries with empty phonetic keys")
        else:
            print(f"  ✅ All entries have phonetic keys")
        
        if short_keys > 0:
            warnings.append(f"{short_keys} entries with very short phonetic keys (< 3 chars)")
            print(f"  ⚠️  {short_keys} entries with very short phonetic keys")
    
    # 6. Check category distribution
    print(f"\n--- Category Distribution ---")
    if "category" in df.columns:
        for cat, count in df["category"].value_counts().items():
            print(f"  {cat}: {count}")
    
    # Summary
    print(f"\n{'=' * 70}")
    if errors:
        print(f"❌ VALIDATION FAILED — {len(errors)} errors found:")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print(f"✅ VALIDATION PASSED")
        if warnings:
            print(f"   {len(warnings)} warnings:")
            for w in warnings:
                print(f"   - {w}")
        return True


def show_stats():
    """Show detailed dataset statistics."""
    print("=" * 70)
    print("KONKAN VANI — DATASET STATISTICS")
    print("=" * 70)
    
    df = pd.read_csv(CSV_PATH)
    
    print(f"\nTotal entries: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Text length statistics
    print(f"\n--- Text Length Statistics ---")
    for col in ["konkani_text", "marathi_meaning", "english_meaning"]:
        if col in df.columns:
            lengths = df[col].str.len()
            print(f"  {col}:")
            print(f"    Mean: {lengths.mean():.0f} chars")
            print(f"    Min:  {lengths.min()} chars")
            print(f"    Max:  {lengths.max()} chars")
    
    # Script distribution
    if "script" in df.columns:
        print(f"\n--- Script Distribution ---")
        for sc, count in df["script"].value_counts().items():
            print(f"  {sc}: {count}")
    
    # Source distribution
    if "source" in df.columns:
        print(f"\n--- Source Distribution ---")
        for src, count in df["source"].value_counts().items():
            print(f"  {src}: {count}")


def show_categories():
    """Show category breakdown."""
    df = pd.read_csv(CSV_PATH)
    
    print("=" * 70)
    print("KONKAN VANI — CATEGORY BREAKDOWN")
    print("=" * 70)
    
    if "category" in df.columns:
        print(f"\nTotal categories: {df['category'].nunique()}")
        print(f"\n--- Entries per Category ---")
        for cat, count in df["category"].value_counts().items():
            print(f"  {cat}: {count}")
    else:
        print("\nNo 'category' column found in dataset.")


def export_for_training():
    """Export validated dataset for model training."""
    df = pd.read_csv(CSV_PATH)
    
    output_path = os.path.join("data", "processed", "idioms_training_ready.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"✅ Exported {len(df)} entries to {output_path}")
    print(f"   Ready for embedding generation and FAISS index building.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scale_dataset.py [validate|stats|categories|export]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "validate":
        validate_dataset()
    elif command == "stats":
        show_stats()
    elif command == "categories":
        show_categories()
    elif command == "export":
        export_for_training()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python src/scale_dataset.py [validate|stats|categories|export]")
        sys.exit(1)
