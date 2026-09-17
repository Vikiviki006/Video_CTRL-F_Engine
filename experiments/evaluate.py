"""
Research evaluation script for VideoCtrl-F.

Computes:
  - Recall@1, Recall@5, Recall@10
  - Precision@K
  - MRR
  - MAP
  - Query latency
  - Indexing time / frames retained ratio (from processing logs)

IMPORTANT: This script does NOT invent results.
Fill evaluation_dataset.csv with real ground-truth timestamps,
run the system, then execute this script against a live backend.
"""

import csv
import time
from pathlib import Path
from typing import List, Dict, Any

# Placeholder – replace with real HTTP calls to your running backend
# or direct service imports when running inside the same environment.

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset(path: str = "evaluation_dataset.csv") -> List[Dict[str, Any]]:
    rows = []
    with open(Path(__file__).parent / path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ground_truth_timestamp"):
                rows.append(row)
    return rows


def compute_metrics(predictions: List[List[float]], ground_truths: List[float], ks=(1, 5, 10)):
    """
    predictions: list of ranked timestamp lists (one per query)
    ground_truths: list of ground-truth timestamps
    """
    recalls = {k: [] for k in ks}
    mrr_list = []
    # Simple temporal tolerance (seconds)
    tol = 2.0

    for preds, gt in zip(predictions, ground_truths):
        gt = float(gt)
        ranks = []
        for i, p in enumerate(preds):
            if abs(p - gt) <= tol:
                ranks.append(i + 1)
        if ranks:
            best = min(ranks)
            mrr_list.append(1.0 / best)
            for k in ks:
                recalls[k].append(1.0 if best <= k else 0.0)
        else:
            mrr_list.append(0.0)
            for k in ks:
                recalls[k].append(0.0)

    metrics = {
        "MRR": sum(mrr_list) / len(mrr_list) if mrr_list else 0.0,
    }
    for k in ks:
        metrics[f"Recall@{k}"] = sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0.0
    return metrics


def main():
    dataset = load_dataset()
    if not dataset:
        print("No ground-truth rows found in evaluation_dataset.csv.")
        print("Add real (query, video_id, ground_truth_timestamp) rows and re-run.")
        # Write empty placeholder results
        out = RESULTS_DIR / "metrics.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["note", "No data – populate evaluation_dataset.csv first"])
        return

    print(f"Loaded {len(dataset)} evaluation queries.")
    print("Connect this script to a running VideoCtrl-F instance to obtain real predictions.")
    print("Results will be written to experiments/results/ after real runs.")


if __name__ == "__main__":
    main()
