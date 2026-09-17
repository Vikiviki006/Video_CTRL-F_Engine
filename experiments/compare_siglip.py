"""
Fair comparison script: SigLIP vs SigLIP-2

Uses identical:
  - videos
  - queries
  - ground truth
  - keyframe policy
  - top-K
  - evaluation metrics

Configure MODEL_NAME via environment and re-run the full pipeline
for each checkpoint. Aggregate results into CSV for plotting.
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("SigLIP vs SigLIP-2 comparison scaffold.")
    print("1. Run full indexing + evaluation with MODEL_NAME=google/siglip-so400m-patch14-384")
    print("2. Run again with MODEL_NAME=google/siglip2-so400m-patch14-384")
    print("3. Collect metrics CSVs and plot with plot_results.py")
    print("No fabricated numbers are written.")


if __name__ == "__main__":
    main()
