import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("KONKAN VANI — DATA SUITABILITY & GAP AUDIT")
print("Project Use Case: Konkani Idiom/Proverb Meaning-Extraction NLP")
print("=" * 80)

def audit_file(fp, desc):
    print(f"\n[{desc}] -> {fp}")
    if not os.path.exists(fp):
        print("Status: NOT FOUND")
        return
    
    size_mb = os.path.getsize(fp) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    try:
        df = pd.read_csv(fp)
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {df.columns.tolist()}")
        print("Sample Data (First 2 rows):")
        for i, row in df.head(2).iterrows():
            print(f"  Row {i}: {dict(row)}")
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")

# 1. Alpaca Konkani
df_alpaca = audit_file("data/raw/alpaca_konkani.csv", "1. Alpaca Konkani (HuggingFace)")

# 2. Goan Data
df_goan = audit_file("data/raw/goan_data.csv", "2. Goan Data (HuggingFace)")

# 3. Konkani Glossary
df_gloss = audit_file("data/raw/konkani_glossary.csv", "3. Konkani Glossary (AI4Bharat Standardized)")

# 4. Wikisource Dictionary
df_ws = audit_file("data/raw/wikisource_dictionary.csv", "4. Wikisource Dictionary (Modern English to Konkani)")
if df_ws is not None:
    sayings = df_ws[df_ws.get("is_saying_related", False) == True]
    print(f"\n  -> Found {len(sayings)} entries flagged as 'is_saying_related':")
    for i, row in sayings.head(5).iterrows():
        print(f"     * English: '{row['english_word']}' -> Konkani: '{row['konkani_translation']}'")

# 5. Wikipedia Extracted
print("\n[5. Wikipedia Extracted Articles Dump] -> data/raw/wiki/extracted/")
wiki_dir = "data/raw/wiki/extracted"
if os.path.exists(wiki_dir):
    art_count = 0
    sample_texts = []
    for root, dirs, files in os.walk(wiki_dir):
        for f in files:
            with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        art_count += 1
                        if len(sample_texts) < 2:
                            try:
                                obj = json.loads(line)
                                sample_texts.append(obj.get("title", "") + ": " + obj.get("text", "")[:120])
                            except Exception:
                                pass
    print(f"Total Extracted Articles: {art_count}")
    for s in sample_texts:
        print(f"  Sample article: {s}...")

# 6. Manual Dataset
df_man = audit_file("data/manual/konkani_idioms_manual.csv", "6. Manual Idioms Dataset")

# 7. Combined Processed
df_comb = audit_file("data/processed/konkani_combined.csv", "7. Processed Combined Dataset")
if df_comb is not None:
    print("\nBreakdown of 'type' in combined dataset:")
    print(df_comb["type"].value_counts().to_string())
    print("\nBreakdown of 'source_dataset' in combined dataset:")
    print(df_comb["source_dataset"].value_counts().to_string())
    print("\nBreakdown of 'script' in combined dataset:")
    print(df_comb["script"].value_counts().to_string())

print("\n" + "=" * 80)
