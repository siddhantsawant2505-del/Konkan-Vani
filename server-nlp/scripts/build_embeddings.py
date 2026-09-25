"""
Konkan Vani — Embedding Builder
================================
Loads sentence-transformers/LaBSE, encodes each idiom row as
(konkani_text + " " + romanized_text), and saves vectors to
data/idiom_embeddings.npy.

Usage:
    python server-nlp/scripts/build_embeddings.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("KONKAN VANI — EMBEDDING BUILDER")
    print("=" * 60)

    # 1. Load the idioms CSV
    csv_path = os.path.join("data", "processed", "idioms_with_phonetic_keys.csv")
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # 2. Prepare texts: konkani_text + " " + romanized_text
    print("\nPreparing texts for encoding...")
    texts = (
        df["konkani_text"].fillna("").astype(str) + " " +
        df["romanized_text"].fillna("").astype(str)
    ).tolist()
    print(f"  Sample text [0]: '{texts[0]}'")
    print(f"  Total texts: {len(texts)}")

    # 3. Load LaBSE model
    print("\nLoading LaBSE model (sentence-transformers/LaBSE)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    print(f"  Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # 4. Encode
    print("\nEncoding texts...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Encoding complete.")

    # 5. Save to data/idiom_embeddings.npy
    output_path = os.path.join("data", "idiom_embeddings.npy")
    os.makedirs("data", exist_ok=True)
    np.save(output_path, embeddings)

    # 6. Report
    print(f"\n{'=' * 60}")
    print(f"EMBEDDING BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Number of embeddings: {len(embeddings)}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
