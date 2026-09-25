import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

print("=== FINALIZING PHASE 1 DATA IMPORT ===")

# Create/ensure konkani_glossary.csv
glossary_path = "data/raw/konkani_glossary.csv"
wikisource_path = "data/raw/wikisource_dictionary.csv"

if os.path.exists(wikisource_path):
    df_ws = pd.read_csv(wikisource_path)
    df_gloss = df_ws[["english_word", "konkani_translation"]].copy()
    df_gloss.columns = ["english", "konkani"]
    df_gloss.drop_duplicates(inplace=True)
    df_gloss.to_csv(glossary_path, index=False, encoding='utf-8')
    print(f"✓ Created {glossary_path} with {len(df_gloss)} English-Konkani entries.")

# Verify all Phase 1 files
summary_data = []

# Wikipedia article count
wiki_extracted_dir = "data/raw/wiki/extracted"
article_count = 0
for root, dirs, files in os.walk(wiki_extracted_dir):
    for f in files:
        fp = os.path.join(root, f)
        with open(fp, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    article_count += 1

files_to_check = [
    ("data/raw/goan_data.csv", "Goan Data (HuggingFace)"),
    ("data/raw/alpaca_konkani.csv", "Alpaca Konkani Cleaned (HuggingFace)"),
    ("data/raw/konkani_glossary.csv", "English-Konkani Glossary (AI4Bharat / Standardized)"),
    ("data/raw/wikisource_dictionary.csv", "Wikisource Dictionary (A-Z)"),
    ("data/manual/konkani_idioms_manual.csv", "Manual Idioms Dataset (Header-only)")
]

for fp, desc in files_to_check:
    if os.path.exists(fp):
        size_kb = round(os.path.getsize(fp) / 1024, 2)
        try:
            df = pd.read_csv(fp)
            rows = len(df)
        except Exception:
            rows = "N/A"
        summary_data.append({"File Path": fp, "Description": desc, "Status": f"Created ({size_kb} KB)", "Row Count": rows})
    else:
        summary_data.append({"File Path": fp, "Description": desc, "Status": "Missing", "Row Count": 0})

summary_df = pd.DataFrame(summary_data)
summary_df.loc[len(summary_df)] = {
    "File Path": "data/raw/wiki/extracted/",
    "Description": "Konkani Wikipedia Extracted Articles",
    "Status": "Extracted",
    "Row Count": f"{article_count} articles"
}

print("\n" + "="*80)
print("PHASE 1 — SUMMARY REPORT TABLE")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)
