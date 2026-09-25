import os
import sys
import json
import xml.etree.ElementTree as ET
import re

sys.stdout.reconfigure(encoding='utf-8')

xml_path = "data/raw/wiki/gomwiki-latest-pages-articles.xml"
output_dir = "data/raw/wiki/extracted"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "wiki_00")

print(f"Parsing {xml_path}...")

def clean_wikitext(text):
    if not text:
        return ""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[(File|Image|फायल):.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref.*?>', '', text)
    text = re.sub(r'\[\[[^\]]*?\|([^\]]*?)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*?)\]\]', r'\1', text)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'==+.*?==+', '', text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

articles = []
count = 0

context = ET.iterparse(xml_path, events=('end',))
_, root = next(context)

prefix = "{http://www.mediawiki.org/xml/export-0.11/}"
if not root.tag.startswith("{"):
    prefix = ""

for event, elem in context:
    if elem.tag == f"{prefix}page":
        title_elem = elem.find(f"{prefix}title")
        ns_elem = elem.find(f"{prefix}ns")
        id_elem = elem.find(f"{prefix}id")
        rev_elem = elem.find(f"{prefix}revision")
        
        title = title_elem.text if title_elem is not None else ""
        ns = ns_elem.text if ns_elem is not None else ""
        art_id = id_elem.text if id_elem is not None else ""
        
        if ns == "0":
            text_elem = rev_elem.find(f"{prefix}text") if rev_elem is not None else None
            raw_text = text_elem.text if text_elem is not None else ""
            
            if raw_text and not raw_text.strip().lower().startswith("#redirect") and not raw_text.strip().lower().startswith("#पुनर्निर्देशन"):
                cleaned = clean_wikitext(raw_text)
                if len(cleaned) > 20:
                    article_obj = {
                        "id": art_id,
                        "url": f"https://gom.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "title": title,
                        "text": f"{title}\n\n{cleaned}"
                    }
                    articles.append(article_obj)
                    count += 1
        
        elem.clear()
        root.clear()

print(f"Extracted {count} articles.")

with open(output_file, "w", encoding="utf-8") as f:
    for art in articles:
        f.write(json.dumps(art, ensure_ascii=False) + "\n")

print(f"Saved extracted articles to {output_file}")
