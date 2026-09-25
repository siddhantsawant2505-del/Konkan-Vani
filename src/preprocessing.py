import os
import sys
import re
import json
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42
sys.stdout.reconfigure(encoding='utf-8')

print("=== PHASE 2 — PREPROCESSING PIPELINE ===", flush=True)

# ---------------------------------------------------------
# 1. Script Detection Function
# ---------------------------------------------------------
def detect_script(text):
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    
    dev_chars = len(re.findall(r'[\u0900-\u097F]', text))
    kan_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
    rom_chars = len(re.findall(r'[a-zA-Z]', text))
    
    counts = {"Devanagari": dev_chars, "Kannada": kan_chars, "Roman": rom_chars}
    max_script = max(counts, key=counts.get)
    
    if counts[max_script] > 0:
        return max_script
    return "Unknown"

# ---------------------------------------------------------
# 2. Language Purity Filter Function
# ---------------------------------------------------------
def is_genuine_konkani(text, script, is_glossary=False):
    if is_glossary:
        return True, "bilingual_exempt"
    
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
    
    non_space_chars = len(re.findall(r'\S', text))
    if non_space_chars == 0:
        return False, "empty"
    
    if script in ["Devanagari", "Kannada"]:
        if script == "Devanagari":
            target_chars = len(re.findall(r'[\u0900-\u097F]', text))
        else:
            target_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
        
        ratio = target_chars / non_space_chars
        if ratio > 0.6:
            return True, "Konkani"
        else:
            return False, "low_script_ratio"
            
    elif script == "Roman":
        # Check ASCII word ratio
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        total_words = len(text.split())
        ascii_word_ratio = len(words) / max(1, total_words)
        
        # Check langdetect only if ascii_word_ratio > 0.85
        if ascii_word_ratio > 0.85:
            try:
                detected_lang = detect(text)
            except Exception:
                detected_lang = "unknown"
            
            if detected_lang == 'en':
                return False, "English_contamination"
        
        return True, "Konkani"
            
    return False, "Unknown_script"

# ---------------------------------------------------------
# 3. Phonetic Normalization Function
# ---------------------------------------------------------
def phonetic_normalize(text):
    if not isinstance(text, str):
        return ""
    
    norm = text.lower()
    norm = re.sub(r'aa', 'a', norm)
    norm = re.sub(r'ee', 'i', norm)
    norm = re.sub(r'oo', 'u', norm)
    norm = re.sub(r'ph', 'f', norm)
    norm = re.sub(r'v', 'w', norm)
    norm = re.sub(r'zh', 'j', norm)
    
    norm = re.sub(r'[^\w\s]', '', norm)
    norm = " ".join(norm.split())
    return norm

# ---------------------------------------------------------
# 4. Main Preprocessing Execution Function
# ---------------------------------------------------------
def main():
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    all_rows = []
    dropped_stats = {"total_raw": 0, "kept": 0, "dropped": 0, "by_source": {}, "by_reason": {}}
    
    def add_item(text, source_dataset, item_type="general", is_glossary=False):
        nonlocal all_rows, dropped_stats
        dropped_stats["total_raw"] += 1
        dropped_stats["by_source"][source_dataset] = dropped_stats["by_source"].get(source_dataset, 0) + 1
        
        script = detect_script(text)
        is_konkani, lang_class = is_genuine_konkani(text, script, is_glossary=is_glossary)
        
        if is_konkani:
            phon_key = phonetic_normalize(text)
            all_rows.append({
                "text": text.strip(),
                "script": script,
                "language_class": lang_class,
                "phonetic_key": phon_key,
                "source_dataset": source_dataset,
                "type": item_type
            })
            dropped_stats["kept"] += 1
        else:
            dropped_stats["dropped"] += 1
            dropped_stats["by_reason"][lang_class] = dropped_stats["by_reason"].get(lang_class, 0) + 1
            
        if dropped_stats["total_raw"] % 10000 == 0:
            print(f"Processed {dropped_stats['total_raw']} raw items...", flush=True)

    # 1. Goan Data
    goan_fp = os.path.join(raw_data_dir, "goan_data.csv")
    if os.path.exists(goan_fp):
        print("Processing Goan Data...", flush=True)
        df = pd.read_csv(goan_fp)
        for _, row in df.iterrows():
            if pd.notna(row.get("instruction")):
                add_item(str(row["instruction"]), "goan_data", "qa_instruction")
            if pd.notna(row.get("response")):
                add_item(str(row["response"]), "goan_data", "qa_response")

    # 2. Alpaca Konkani
    alpaca_fp = os.path.join(raw_data_dir, "alpaca_konkani.csv")
    if os.path.exists(alpaca_fp):
        print("Processing Alpaca Konkani...", flush=True)
        df = pd.read_csv(alpaca_fp)
        for _, row in df.iterrows():
            if pd.notna(row.get("instruction")):
                add_item(str(row["instruction"]), "alpaca_konkani", "instruction")
            if pd.notna(row.get("output")):
                add_item(str(row["output"]), "alpaca_konkani", "output")

    # 3. Konkani Glossary (Exempt)
    gloss_fp = os.path.join(raw_data_dir, "konkani_glossary.csv")
    if os.path.exists(gloss_fp):
        print("Processing Konkani Glossary (Exempt)...", flush=True)
        df = pd.read_csv(gloss_fp)
        for _, row in df.iterrows():
            if pd.notna(row.get("konkani")):
                add_item(str(row["konkani"]), "konkani_glossary", "glossary_term", is_glossary=True)

    # 4. Wikisource Dictionary
    wiki_dict_fp = os.path.join(raw_data_dir, "wikisource_dictionary.csv")
    if os.path.exists(wiki_dict_fp):
        print("Processing Wikisource Dictionary...", flush=True)
        df = pd.read_csv(wiki_dict_fp)
        for _, row in df.iterrows():
            if pd.notna(row.get("konkani_translation")):
                item_t = "proverb_saying" if row.get("is_saying_related") else "dictionary_entry"
                add_item(str(row["konkani_translation"]), "wikisource_dictionary", item_t)

    # 5. Extracted Wikipedia Articles
    wiki_ext_dir = os.path.join(raw_data_dir, "wiki", "extracted")
    if os.path.exists(wiki_ext_dir):
        print("Processing Wikipedia Extracted Articles...", flush=True)
        for root, dirs, files in os.walk(wiki_ext_dir):
            for f in files:
                fp = os.path.join(root, f)
                with open(fp, "r", encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                body = obj.get("text", "")
                                paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 30]
                                for p in paragraphs[:3]:
                                    add_item(p, "wikipedia", "article_paragraph")
                            except Exception:
                                pass

    # 6. Manual Idioms Dataset
    manual_fp = os.path.join("data", "manual", "konkani_idioms_manual.csv")
    if os.path.exists(manual_fp):
        print("Processing Manual Idiom Dataset...", flush=True)
        df = pd.read_csv(manual_fp)
        for _, row in df.iterrows():
            if pd.notna(row.get("konkani_text")):
                add_item(str(row["konkani_text"]), "manual_idioms", "idiom_proverb")

    df_combined = pd.DataFrame(all_rows)
    print(f"\nExtracted {len(df_combined)} initial valid rows.", flush=True)

    # 5. Deduplication
    print("\nApplying Deduplication...", flush=True)
    initial_count = len(df_combined)
    
    df_combined.drop_duplicates(subset=["text"], inplace=True)
    exact_dedup_count = len(df_combined)
    
    df_combined.drop_duplicates(subset=["phonetic_key", "source_dataset"], inplace=True)
    final_count = len(df_combined)
    
    print(f"✓ Deduplication: {initial_count} -> {exact_dedup_count} (exact) -> {final_count} (phonetic source level)", flush=True)

    output_combined_path = os.path.join(processed_data_dir, "konkani_combined.csv")
    df_combined.to_csv(output_combined_path, index=False, encoding='utf-8')
    print(f"\n✓ Saved final combined dataset to {output_combined_path}", flush=True)

    print("\n" + "="*70, flush=True)
    print("PHASE 2 — PREPROCESSING SUMMARY STATISTICS", flush=True)
    print("="*70, flush=True)
    print(f"Total Raw Rows Processed: {dropped_stats['total_raw']}", flush=True)
    print(f"Total Rows Kept:          {final_count}", flush=True)
    print(f"Total Rows Dropped:       {dropped_stats['dropped']} ({round(dropped_stats['dropped']/max(1, dropped_stats['total_raw'])*100, 2)}%)", flush=True)
    
    print("\n--- Rows Per Source Dataset ---", flush=True)
    source_counts = df_combined["source_dataset"].value_counts()
    for src, cnt in source_counts.items():
        raw_cnt = dropped_stats["by_source"].get(src, cnt)
        drop_cnt = raw_cnt - cnt
        drop_rate = round(drop_cnt / max(1, raw_cnt) * 100, 2)
        print(f" - {src:<25}: {cnt:>6} kept | {drop_cnt:>6} dropped ({drop_rate}% drop rate)", flush=True)

    print("\n--- Rows Per Script ---", flush=True)
    script_counts = df_combined["script"].value_counts()
    for sc, cnt in script_counts.items():
        print(f" - {sc:<15}: {cnt:>6} rows ({round(cnt/len(df_combined)*100, 2)}%)", flush=True)

    print("\n--- Dropped Rows Reason Breakdown ---", flush=True)
    for rsn, cnt in dropped_stats["by_reason"].items():
        print(f" - {rsn:<25}: {cnt:>6} rows", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
