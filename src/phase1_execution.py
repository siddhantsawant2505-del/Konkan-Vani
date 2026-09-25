import os
import sys
import glob
import zipfile
import io
import requests
import pandas as pd
import bz2
import json
import re
from bs4 import BeautifulSoup
from datasets import load_dataset

# Ensure UTF-8 output in Windows console
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/ai4bharat", exist_ok=True)
os.makedirs("data/raw/wiki", exist_ok=True)
os.makedirs("data/raw/wiki/extracted", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/manual", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("src", exist_ok=True)

print("=== PHASE 1 — DATA IMPORT ===")

# ---------------------------------------------------------
# Step 2: HuggingFace Datasets
# ---------------------------------------------------------
print("\n--- 1. Importing HuggingFace Datasets ---")

# Goan_Data
print("Loading konkani/Goan_Data...")
try:
    ds_goan = load_dataset("konkani/Goan_Data")
    split_goan = list(ds_goan.keys())[0] if hasattr(ds_goan, 'keys') else 'train'
    df_goan = ds_goan[split_goan].to_pandas()
    df_goan.to_csv("data/raw/goan_data.csv", index=False, encoding='utf-8')
    print(f"✓ Saved data/raw/goan_data.csv | Shape: {df_goan.shape}")
    print("Goan_Data First 3 rows:")
    print(df_goan.head(3).to_string())
except Exception as e:
    print(f"Error loading konkani/Goan_Data: {e}")

# Alpaca Konkani
print("\nLoading saillab/alpaca-konkani-cleaned...")
try:
    ds_alpaca = load_dataset("saillab/alpaca-konkani-cleaned")
    split_alpaca = list(ds_alpaca.keys())[0] if hasattr(ds_alpaca, 'keys') else 'train'
    df_alpaca = ds_alpaca[split_alpaca].to_pandas()
    df_alpaca.to_csv("data/raw/alpaca_konkani.csv", index=False, encoding='utf-8')
    print(f"✓ Saved data/raw/alpaca_konkani.csv | Shape: {df_alpaca.shape}")
    print("Alpaca_Konkani First 3 rows:")
    print(df_alpaca.head(3).to_string())
except Exception as e:
    print(f"Error loading saillab/alpaca-konkani-cleaned: {e}")


# ---------------------------------------------------------
# Step 3: AI4Bharat Repos & Glossary
# ---------------------------------------------------------
print("\n--- 2. Downloading AI4Bharat Repos & Extracting Glossary ---")

indic_glossaries_dir = os.path.join("data", "raw", "ai4bharat", "Indic-Glossaries")
indicnlp_catalog_dir = os.path.join("data", "raw", "ai4bharat", "indicnlp_catalog")

def download_and_extract_github_zip(repo_url, target_dir):
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"{target_dir} already exists and is not empty.")
        return
    
    headers = {"User-Agent": "Mozilla/5.0"}
    for branch in ["main", "master"]:
        zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
        print(f"Fetching ZIP from {zip_url}...")
        res = requests.get(zip_url, headers=headers)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                # Extract all to parent directory
                parent_dir = os.path.dirname(target_dir)
                z.extractall(parent_dir)
                # Find extracted folder name (usually Repo-branch)
                repo_name = repo_url.split("/")[-1]
                extracted_folder = os.path.join(parent_dir, f"{repo_name}-{branch}")
                if os.path.exists(extracted_folder):
                    if os.path.exists(target_dir):
                        os.rmdir(target_dir)
                    os.rename(extracted_folder, target_dir)
            print(f"✓ Extracted into {target_dir}")
            return
        else:
            print(f"Branch {branch} returned status {res.status_code}")

download_and_extract_github_zip("https://github.com/AI4Bharat/Indic-Glossaries", indic_glossaries_dir)
download_and_extract_github_zip("https://github.com/AI4Bharat/indicnlp_catalog", indicnlp_catalog_dir)

# Find English-Konkani glossary
print("\nSearching for Konkani glossary file inside Indic-Glossaries...")
found_files = []
for root, dirs, files in os.walk(indic_glossaries_dir):
    for file in files:
        f_lower = file.lower()
        if any(term in f_lower for term in ["konkani", "gom", "kok"]):
            found_files.append(os.path.join(root, file))

print(f"Found candidate files matching keywords: {found_files}")

df_glossary = None
if found_files:
    for filepath in found_files:
        try:
            print(f"Inspecting file: {filepath}")
            sep = '\t' if filepath.endswith(('.tsv', '.txt')) else ','
            df_temp = pd.read_csv(filepath, sep=sep, on_bad_lines='skip')
            print(f"Columns in {os.path.basename(filepath)}: {list(df_temp.columns)}")
            
            eng_col = None
            kok_col = None
            for c in df_temp.columns:
                c_lower = str(c).lower()
                if 'eng' in c_lower or 'en' == c_lower or 'english' in c_lower:
                    eng_col = c
                elif 'kok' in c_lower or 'konkani' in c_lower or 'gom' in c_lower or 'tgt' in c_lower or 'target' in c_lower:
                    kok_col = c
            
            if eng_col is None and len(df_temp.columns) >= 2:
                eng_col = df_temp.columns[0]
                kok_col = df_temp.columns[1]

            if eng_col and kok_col:
                df_glossary = df_temp[[eng_col, kok_col]].copy()
                df_glossary.columns = ["english", "konkani"]
                print(f"✓ Successfully extracted glossary from {os.path.basename(filepath)}")
                break
        except Exception as ex:
            print(f"Error reading {filepath}: {ex}")

if df_glossary is None:
    # Exhaustive search across all files in Indic-Glossaries
    print("Performing exhaustive file search in Indic-Glossaries...")
    for root, dirs, files in os.walk(indic_glossaries_dir):
        for file in files:
            if file.endswith(('.csv', '.tsv', '.txt')) and not file.startswith('.'):
                fp = os.path.join(root, file)
                try:
                    df_temp = pd.read_csv(fp, sep=None, engine='python', nrows=5)
                    cols = [str(c).lower() for c in df_temp.columns]
                    if any(term in c for term in ['kok', 'konkani', 'gom'] for c in cols):
                        print(f"Match found in {fp} with columns: {df_temp.columns}")
                        df_full = pd.read_csv(fp, sep=None, engine='python')
                        eng_c = [c for c in df_full.columns if any(k in str(c).lower() for k in ['eng', 'src', 'en'])][0]
                        kok_c = [c for c in df_full.columns if any(k in str(c).lower() for k in ['kok', 'konkani', 'gom', 'tgt'])][0]
                        df_glossary = df_full[[eng_c, kok_c]].copy()
                        df_glossary.columns = ["english", "konkani"]
                        break
                except Exception:
                    pass
        if df_glossary is not None:
            break

if df_glossary is not None:
    df_glossary.dropna(inplace=True)
    df_glossary.to_csv("data/raw/konkani_glossary.csv", index=False, encoding='utf-8')
    print(f"✓ Saved data/raw/konkani_glossary.csv | Shape: {df_glossary.shape}")
    print("Glossary first 3 rows:")
    print(df_glossary.head(3).to_string())
else:
    print("WARNING: Could not locate English-Konkani glossary file automatically. Directory tree of Indic-Glossaries:")
    for root, dirs, files in os.walk(indic_glossaries_dir):
        level = root.replace(indic_glossaries_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:10]:
            print(f"{subindent}{f}")


# ---------------------------------------------------------
# Step 4: Konkani Wikipedia Dump & Wikiextractor
# ---------------------------------------------------------
print("\n--- 3. Downloading Konkani Wikipedia Dump & Extracting ---")

wiki_dir = os.path.join("data", "raw", "wiki")
bz2_path = os.path.join(wiki_dir, "gomwiki-latest-pages-articles.xml.bz2")
xml_path = os.path.join(wiki_dir, "gomwiki-latest-pages-articles.xml")
extracted_dir = os.path.join(wiki_dir, "extracted")

headers_wiki = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KonkanVaniNLP/1.0"}

if not os.path.exists(bz2_path) and not os.path.exists(xml_path):
    # Fetch dump index page to find active bz2 link
    index_url = "https://dumps.wikimedia.org/gomwiki/latest/"
    print(f"Checking Wikipedia dump index: {index_url}...")
    target_filename = None
    try:
        r_index = requests.get(index_url, headers=headers_wiki, timeout=15)
        if r_index.status_code == 200:
            soup_idx = BeautifulSoup(r_index.text, "html.parser")
            links = [a['href'] for a in soup_idx.find_all('a', href=True)]
            # Look for pages-articles.xml.bz2 or pages-articles-multistream.xml.bz2
            article_dumps = [l for l in links if 'pages-articles' in l and l.endswith('.bz2')]
            if article_dumps:
                # Prefer pages-articles.xml.bz2 over multistream if available
                target_filename = article_dumps[0]
                for l in article_dumps:
                    if 'pages-articles.xml.bz2' in l and 'multistream' not in l:
                        target_filename = l
                        break
                print(f"Found active article dump file: {target_filename}")
    except Exception as e:
        print(f"Could not fetch dump index: {e}")
    
    if target_filename:
        download_url = f"https://dumps.wikimedia.org/gomwiki/latest/{target_filename}"
    else:
        download_url = "https://dumps.wikimedia.org/gomwiki/latest/gomwiki-latest-pages-articles.xml.bz2"
    
    print(f"Downloading from: {download_url}...")
    res = requests.get(download_url, stream=True, headers=headers_wiki)
    if res.status_code == 200:
        with open(bz2_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print(f"✓ Downloaded {bz2_path} ({round(os.path.getsize(bz2_path)/(1024*1024), 2)} MB)")
    else:
        print(f"Failed to download wiki dump. HTTP Status: {res.status_code}")
else:
    print(f"Wiki dump archive already present.")

if os.path.exists(bz2_path) and not os.path.exists(xml_path):
    print("Decompressing bz2 archive...")
    with bz2.open(bz2_path, "rb") as source, open(xml_path, "wb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
    print(f"✓ Decompressed to {xml_path} ({round(os.path.getsize(xml_path)/(1024*1024), 2)} MB)")

if os.path.exists(xml_path):
    print("Running wikiextractor with --json output...")
    try:
        from wikiextractor.extract import main as wikiextractor_main
        # Run wikiextractor via Python API if available
        import sys
        sys_argv_orig = sys.argv
        sys.argv = ["wikiextractor", xml_path, "--json", "-o", extracted_dir]
        try:
            wikiextractor_main()
        except SystemExit:
            pass
        sys.argv = sys_argv_orig
        print("✓ Wikiextractor processing completed.")
    except Exception as ex:
        print(f"Executing wikiextractor module via subprocess: {ex}")
        import subprocess
        res_we = subprocess.run([sys.executable, "-m", "wikiextractor.extract", xml_path, "--json", "-o", extracted_dir], capture_output=True, text=True)
        print(f"wikiextractor return code: {res_we.returncode}")

# Count extracted articles
article_count = 0
for root, dirs, files in os.walk(extracted_dir):
    for f in files:
        if f.startswith("wiki_"):
            fp = os.path.join(root, f)
            with open(fp, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        article_count += 1

print(f"✓ Total extracted Wikipedia articles: {article_count}")


# ---------------------------------------------------------
# Step 5: Scrape Wikisource Konkani-English Dictionary
# ---------------------------------------------------------
print("\n--- 4. Scraping Wikisource Konkani-English Dictionary ---")

dictionary_entries = []
high_value_keywords = ["saying", "mhonni", "proverb", "opar"]

alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]

for letter in alphabet:
    url = f"https://wikisource.org/wiki/Modern_English_to_Konkani_Dictionary/{letter}"
    try:
        r = requests.get(url, headers=headers_wiki, timeout=15)
        if r.status_code != 200:
            print(f"Letter {letter}: HTTP {r.status_code}")
            continue
        
        soup = BeautifulSoup(r.content, "html.parser")
        content_div = soup.find("div", {"class": "mw-parser-output"})
        if not content_div:
            continue
        
        b_tags = content_div.find_all("b")
        letter_count = 0
        for b in b_tags:
            word = b.get_text(strip=True)
            if not word or len(word) > 100 or word.lower() in ["table of contents", "chapters"]:
                continue
            
            sibling_text = []
            curr = b.next_sibling
            while curr:
                if curr.name == "b" or curr.name == "h2" or curr.name == "h1":
                    break
                if isinstance(curr, str):
                    sibling_text.append(str(curr))
                elif curr.name in ["i", "span", "a", "em"]:
                    sibling_text.append(curr.get_text())
                elif curr.name in ["br", "p", "div"]:
                    sibling_text.append(curr.get_text())
                    break
                curr = curr.next_sibling
            
            translation = " ".join("".join(sibling_text).split()).strip()
            if not translation:
                continue
            
            full_entry_text = f"{word} {translation}".lower()
            is_saying = any(kw in full_entry_text for kw in high_value_keywords)
            
            dictionary_entries.append({
                "english_word": word,
                "konkani_translation": translation,
                "is_saying_related": is_saying
            })
            letter_count += 1
        
        print(f"Letter {letter}: {letter_count} entries extracted.")
    except Exception as e:
        print(f"Error scraping letter {letter}: {e}")

df_wikisource = pd.DataFrame(dictionary_entries)
if not df_wikisource.empty:
    df_wikisource.drop_duplicates(subset=["english_word", "konkani_translation"], inplace=True)
    df_wikisource.to_csv("data/raw/wikisource_dictionary.csv", index=False, encoding='utf-8')
    saying_count = df_wikisource["is_saying_related"].sum()
    print(f"✓ Saved data/raw/wikisource_dictionary.csv | Total: {len(df_wikisource)} | Saying-related: {saying_count}")
else:
    print("WARNING: No Wikisource dictionary entries were extracted.")


# ---------------------------------------------------------
# Step 6: Create Header-Only Manual Dataset
# ---------------------------------------------------------
print("\n--- 5. Creating Header-Only Manual Dataset ---")

manual_csv_path = "data/manual/konkani_idioms_manual.csv"
headers_manual = ["konkani_text", "romanized_text", "script", "literal_meaning", "figurative_meaning", "english_equivalent", "example_sentence", "source"]

df_manual = pd.DataFrame(columns=headers_manual)
df_manual.to_csv(manual_csv_path, index=False, encoding='utf-8')
print(f"✓ Created {manual_csv_path} with headers only.")


# ---------------------------------------------------------
# Phase 1 Summary Report Table
# ---------------------------------------------------------
print("\n" + "="*75)
print("PHASE 1 — DATA IMPORT SUMMARY TABLE")
print("="*75)

summary_data = []

def get_file_info(filepath):
    if os.path.exists(filepath):
        size_kb = round(os.path.getsize(filepath) / 1024, 2)
        try:
            df = pd.read_csv(filepath)
            rows = len(df)
        except Exception:
            rows = "N/A"
        return f"Created ({size_kb} KB)", rows
    return "Missing", 0

files_to_check = [
    ("data/raw/goan_data.csv", "Goan Data (HuggingFace)"),
    ("data/raw/alpaca_konkani.csv", "Alpaca Konkani Cleaned (HuggingFace)"),
    ("data/raw/konkani_glossary.csv", "English-Konkani Glossary (AI4Bharat)"),
    ("data/raw/wikisource_dictionary.csv", "Wikisource Dictionary (A-Z)"),
    ("data/manual/konkani_idioms_manual.csv", "Manual Idioms Dataset (Header-only)")
]

for fp, desc in files_to_check:
    status, row_cnt = get_file_info(fp)
    summary_data.append({"File Path": fp, "Description": desc, "Status": status, "Row Count": row_cnt})

summary_df = pd.DataFrame(summary_data)
summary_df.loc[len(summary_df)] = {
    "File Path": "data/raw/wiki/extracted/",
    "Description": "Konkani Wikipedia Extracted Articles",
    "Status": "Extracted",
    "Row Count": f"{article_count} articles"
}

print(summary_df.to_string(index=False))
print("="*75)
