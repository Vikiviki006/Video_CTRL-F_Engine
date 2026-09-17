"""
Generate research figures from measured CSV data only.
Never invent values.
"""

from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    print("plot_results.py – Matplotlib figure generator")
    print("Expected input CSVs (produced by real experiments):")
    print("  results/recall_comparison.csv")
    print("  results/latency.csv")
    print("  results/frames_retained.csv")
    print("  results/indexing_time.csv")
    print("  results/storage_reduction.csv")
    print("")
    print("Figures to produce (once data exists):")
    print("  Figure 1 – Recall@K (SigLIP vs SigLIP-2)")
    print("  Figure 2 – Query latency")
    print("  Figure 3 – Frames retained before vs after adaptive extraction")
    print("  Figure 4 – Indexing time")
    print("  Figure 5 – Storage reduction")
    print("")
    print("No plots are generated until real measured data is present.")


if __name__ == "__main__":
    main()
