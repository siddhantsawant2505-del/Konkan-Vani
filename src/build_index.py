"""
Konkan Vani — Index Builder
=============================
Loads the gold idiom CSV and pre-computed embeddings (built by
build_embeddings.py), then builds a FAISS IndexFlatIP for fast cosine-
similarity search at query time.

The validation queries at the end are encoded with the same model the
runtime server uses (model_registry → custom model or LaBSE fallback).

Usage:
    python -X utf8 src/build_index.py

Output:
    data/processed/index/konkani.index    (FAISS index file)
    data/processed/index/konkani_meta.json (parallel metadata)
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDIOMS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "idioms_with_phonetic_keys.csv")
EMBEDDINGS_NPY = os.path.join(PROJECT_ROOT, "data", "idiom_embeddings.npy")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "index")

# Make model_registry importable regardless of cwd
_SERVER_NLP = os.path.join(PROJECT_ROOT, "server-nlp")
if _SERVER_NLP not in sys.path:
    sys.path.insert(0, _SERVER_NLP)


def main():
    print("=" * 70)
    print("KONKAN VANI — FAISS INDEX BUILDER")
    print("=" * 70)

    # 1. Load the gold idiom dataset
    if not os.path.exists(IDIOMS_CSV):
        print(f"ERROR: {IDIOMS_CSV} not found.")
        sys.exit(1)

    print(f"\nLoading {IDIOMS_CSV} ...")
    df = pd.read_csv(IDIOMS_CSV)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Filter out empty rows
    df = df.dropna(subset=["konkani_text"])
    df = df[df["konkani_text"].str.strip().str.len() > 0]
    print(f"  Rows after cleanup: {len(df)}")

    # 2. Load pre-computed embeddings
    if not os.path.exists(EMBEDDINGS_NPY):
        print(f"\nERROR: {EMBEDDINGS_NPY} not found. Run build_embeddings.py first.")
        sys.exit(1)

    print(f"\nLoading embeddings from {EMBEDDINGS_NPY} ...")
    embeddings = np.load(EMBEDDINGS_NPY)
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dtype: {embeddings.dtype}")

    assert embeddings.shape[0] == len(df), (
        f"Row count ({len(df)}) != embedding count ({embeddings.shape[0]}). "
        f"Rebuild embeddings after updating the CSV."
    )
    print(f"  ✓ Row count matches embedding count")

    # 3. Build metadata from CSV rows
    metadata = []
    for _, row in df.iterrows():
        meta = {
            "konkani_text": str(row.get("konkani_text", "")),
            "romanized_text": str(row.get("romanized_text", "")),
            "literal_meaning": str(row.get("literal_meaning", "")),
            "figurative_meaning": str(row.get("figurative_meaning", "")),
            "english_meaning": str(row.get("english_meaning", "")),
            "marathi_meaning": str(row.get("marathi_meaning", "")),
            "example_sentence": str(row.get("example_sentence", "")),
            "phonetic_key": str(row.get("phonetic_key", "")),
            "category": str(row.get("category", "")),
            "cultural_context": str(row.get("cultural_context", "")),
            "source": str(row.get("source", "")),
            "script": str(row.get("script", "")),
        }
        metadata.append(meta)

    # 4. L2-normalize embeddings for cosine similarity via inner product
    print("\nL2-normalizing embeddings ...")
    faiss_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    faiss_norms = np.maximum(faiss_norms, 1e-8)
    normalized = (embeddings / faiss_norms).astype(np.float32)

    # 5. Build FAISS index (IndexFlatIP = inner product on normalized vectors = cosine similarity)
    import faiss

    print("Building FAISS IndexFlatIP ...")
    t0 = time.time()
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    elapsed = time.time() - t0
    print(f"  Added {index.ntotal} vectors (dim={dim}) in {elapsed:.3f}s")

    # 6. Save index + metadata
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "konkani.index")
    meta_path = os.path.join(INDEX_DIR, "konkani_meta.json")

    print(f"\nSaving FAISS index to {index_path} ...")
    faiss.write_index(index, index_path)
    index_size = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  ✓ Index saved ({index_size:.1f} MB)")

    print(f"Saving metadata to {meta_path} ...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    meta_size = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"  ✓ Metadata saved ({meta_size:.1f} MB)")

    # 7. Validation search — encode with the same model as the runtime server
    print("\n--- Validation Search ---")
    test_queries = [
        "haat dakhvun ayaak",
        "udaak pioone wisor",
        "modde maarpe",
    ]

    from app.core import model_registry
    for query in test_queries:
        q_emb = model_registry.encode([query]).astype(np.float32)
        scores, idx_list = index.search(q_emb, 3)
        print(f"\n  Query: '{query}'")
        for rank, (idx, score) in enumerate(zip(idx_list[0], scores[0]), 1):
            meta = metadata[idx]
            print(f"    {rank}. [{score:.4f}] {meta['konkani_text'][:60]}  ({meta['category']})")

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print(f"  Total entries indexed: {index.ntotal}")
    print(f"  Embedding dimension: {dim}")
    print(f"  Index location: {INDEX_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
