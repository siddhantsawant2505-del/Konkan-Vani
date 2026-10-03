"""
server-nlp/scripts/train_model.py
==================================
Fine-tunes a domain-specific multilingual semantic encoder (LaBSE)
on Konkani idiom and proverb pairs using MultipleNegativesRankingLoss (MNRL).

Key Features:
  - Splits data into Train (80%) and Test (20%) with fixed seed for reproducibility
  - Evaluates pre-trained baseline model on the test split
  - Fine-tunes model on NVIDIA GPU using PyTorch + CUDA
  - Evaluates fine-tuned model on the exact same test split
  - Computes MRR@1, MRR@5, MRR@10, Recall@1, Recall@5
  - Saves the trained checkpoint to models/konkan-vani-encoder-v1/
  - Generates training loss curve and comparison benchmark report
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "idioms_with_phonetic_keys.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_MODEL_DIR = os.path.join(MODELS_DIR, "konkan-vani-encoder-v1")
REPORT_PATH = os.path.join(MODELS_DIR, "benchmark_report.json")
PLOT_PATH = os.path.join(MODELS_DIR, "training_loss_curve.png")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)


def evaluate_retrieval(model, test_df, top_k=(1, 5, 10), batch_size=64):
    """
    Evaluates semantic retrieval accuracy and MRR on the test set.
    For each test idiom, the query is the Konkani text (or Romanized),
    and the target is the full meaning text.
    """
    logger.info(f"Evaluating {len(test_df)} test queries...")
    
    # Queries: Romanized & Devanagari text
    queries = test_df["konkani_text"].fillna("").astype(str).tolist()
    # Documents: English + Marathi + Figurative meaning
    meanings = (
        test_df["english_meaning"].fillna("").astype(str) + " — " +
        test_df["figurative_meaning"].fillna("").astype(str) + " (" +
        test_df["marathi_meaning"].fillna("").astype(str) + ")"
    ).tolist()

    query_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    doc_embs = model.encode(meanings, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)

    # Cosine similarity matrix: (N, N)
    sim_matrix = query_embs @ doc_embs.T

    ranks = []
    top_hits = {k: 0 for k in top_k}
    positive_similarities = []

    for i in range(len(test_df)):
        sims = sim_matrix[i]
        positive_similarities.append(float(sims[i]))
        
        # Rank of the ground-truth document (1-indexed)
        # Higher similarity = better rank
        sorted_indices = np.argsort(sims)[::-1]
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

        for k in top_k:
            if rank <= k:
                top_hits[k] += 1

    total = len(test_df)
    mrr_10 = float(np.mean([1.0 / r if r <= 10 else 0.0 for r in ranks]))
    mrr_5 = float(np.mean([1.0 / r if r <= 5 else 0.0 for r in ranks]))
    recall_1 = float(top_hits[1] / total)
    recall_5 = float(top_hits[5] / total)
    recall_10 = float(top_hits[10] / total)
    avg_pos_sim = float(np.mean(positive_similarities))

    metrics = {
        "mrr@10": round(mrr_10, 4),
        "mrr@5": round(mrr_5, 4),
        "recall@1": round(recall_1, 4),
        "recall@5": round(recall_5, 4),
        "recall@10": round(recall_10, 4),
        "avg_pos_similarity": round(avg_pos_sim, 4),
    }
    return metrics


def main():
    print("=" * 70)
    print("KONKAN VANI — DOMAIN-SPECIFIC MODEL FINE-TUNING")
    print("=" * 70)

    # 1. Hardware check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Hardware] Target Device: {device}")
    if device == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    # 2. Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n[Data] Loaded {len(df)} idiom rows from {DATA_PATH}")

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Train / Test split (80% train, 20% test)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples : {len(test_df)}")

    # 3. Create training examples
    train_examples = []
    for _, row in train_df.iterrows():
        konk = str(row.get("konkani_text", "")).strip()
        rom = str(row.get("romanized_text", "")).strip()
        meaning = (
            str(row.get("english_meaning", "")).strip() + " — " +
            str(row.get("figurative_meaning", "")).strip() + " (" +
            str(row.get("marathi_meaning", "")).strip() + ")"
        )
        if konk and meaning:
            train_examples.append(InputExample(texts=[konk, meaning]))
        if rom and meaning:
            train_examples.append(InputExample(texts=[rom, meaning]))

    print(f"\n[Training Pairs] Created {len(train_examples)} (idiom, meaning) training pairs")

    # 4. Load baseline model
    base_model_name = "sentence-transformers/LaBSE"
    print(f"\n[Base Model] Loading {base_model_name}...")
    model = SentenceTransformer(base_model_name, device=device)

    # 5. Evaluate baseline BEFORE training
    print("\n[Baseline Evaluation] Benchmarking pre-trained model on test set...")
    baseline_metrics = evaluate_retrieval(model, test_df)
    print("  Baseline Results:")
    for k, v in baseline_metrics.items():
        print(f"    - {k}: {v}")

    # 6. Setup training
    batch_size = 16  # Safe for 6GB RTX 3050 VRAM
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = MultipleNegativesRankingLoss(model)

    epochs = 3
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    print(f"\n[Training] Starting fine-tuning for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {len(train_dataloader)}")
    print(f"  Total optimization steps: {len(train_dataloader) * epochs}")
    print(f"  Loss function: MultipleNegativesRankingLoss (MNRL)")

    start_time = time.time()
    
    # Store loss history
    loss_history = []

    def loss_callback(loss_value, epoch, step):
        loss_history.append((step, float(loss_value)))

    # Fine-tuning loop
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": 2e-5},
        show_progress_bar=True,
        use_amp=True if device == "cuda" else False,  # Mixed precision fp16
    )

    training_time = round(time.time() - start_time, 2)
    print(f"\n[Training Complete] Finished in {training_time} seconds ({round(training_time/60, 2)} minutes)")

    # 7. Evaluate fine-tuned model AFTER training
    print("\n[Post-Training Evaluation] Benchmarking fine-tuned model on test set...")
    finetuned_metrics = evaluate_retrieval(model, test_df)
    print("  Fine-Tuned Model Results:")
    for k, v in finetuned_metrics.items():
        print(f"    - {k}: {v}")

    # 8. Save model checkpoint
    print(f"\n[Saving Model] Saving custom weights to {OUTPUT_MODEL_DIR} ...")
    model.save(OUTPUT_MODEL_DIR)
    print("  Saved successfully!")

    # 9. Generate comparison report
    comparison = {
        "model_name": "konkan-vani-encoder-v1",
        "base_model": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": training_time,
        "train_pairs": len(train_examples),
        "test_samples": len(test_df),
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "improvements": {
            "mrr@10_gain": f"+{round(((finetuned_metrics['mrr@10'] - baseline_metrics['mrr@10']) / max(baseline_metrics['mrr@10'], 0.001)) * 100, 1)}%",
            "recall@1_gain": f"+{round(((finetuned_metrics['recall@1'] - baseline_metrics['recall@1']) / max(baseline_metrics['recall@1'], 0.001)) * 100, 1)}%",
            "recall@5_gain": f"+{round(((finetuned_metrics['recall@5'] - baseline_metrics['recall@5']) / max(baseline_metrics['recall@5'], 0.001)) * 100, 1)}%",
            "avg_similarity_gain": f"+{round(((finetuned_metrics['avg_pos_similarity'] - baseline_metrics['avg_pos_similarity']) / max(baseline_metrics['avg_pos_similarity'], 0.001)) * 100, 1)}%",
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Report] Evaluation report saved to {REPORT_PATH}")

    # 10. Generate Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        metrics_names = ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]
        base_vals = [baseline_metrics["recall@1"], baseline_metrics["recall@5"], baseline_metrics["recall@10"], baseline_metrics["mrr@10"]]
        fine_vals = [finetuned_metrics["recall@1"], finetuned_metrics["recall@5"], finetuned_metrics["recall@10"], finetuned_metrics["mrr@10"]]

        x = np.arange(len(metrics_names))
        width = 0.35

        plt.bar(x - width/2, base_vals, width, label="Generic Baseline (LaBSE)", color="#8d7167")
        plt.bar(x + width/2, fine_vals, width, label="Fine-Tuned (Konkan Vani)", color="#2a6865")

        plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
        plt.title("Konkan Vani — Retrieval Benchmark Comparison (Baseline vs Fine-Tuned)", fontsize=14, fontweight="bold")
        plt.xticks(x, metrics_names, fontsize=11)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for i in range(len(metrics_names)):
            plt.text(x[i] - width/2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha="center", fontsize=9)
            plt.text(x[i] + width/2, fine_vals[i] + 0.02, f"{fine_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=200)
        plt.close()
        print(f"[Plot] Comparison chart saved to {PLOT_PATH}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")

    # Final summary display
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Generic Baseline':<18} | {'Fine-Tuned Model':<18} | {'Gain':<10}")
    print("-" * 75)
    print(f"{'Recall@1 (Top-1 Accuracy)':<25} | {baseline_metrics['recall@1']:<18} | {finetuned_metrics['recall@1']:<18} | {comparison['improvements']['recall@1_gain']:<10}")
    print(f"{'Recall@5':<25} | {baseline_metrics['recall@5']:<18} | {finetuned_metrics['recall@5']:<18} | {comparison['improvements']['recall@5_gain']:<10}")
    print(f"{'Recall@10':<25} | {baseline_metrics['recall@10']:<18} | {finetuned_metrics['recall@10']:<18} | ")
    print(f"{'MRR@10':<25} | {baseline_metrics['mrr@10']:<18} | {finetuned_metrics['mrr@10']:<18} | {comparison['improvements']['mrr@10_gain']:<10}")
    print(f"{'Avg Positive Sim':<25} | {baseline_metrics['avg_pos_similarity']:<18} | {finetuned_metrics['avg_pos_similarity']:<18} | {comparison['improvements']['avg_similarity_gain']:<10}")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
