"""Figure 4 - NDCG@K: vector-only baseline vs. vector + graph.

Input : experiments/results/retrieval_metrics.csv  (columns ``ndcg@1`` ... ``ndcg@20``)
Output: experiments/results/figures/ndcg_at_k.png

Binary relevance is used, since no dataset in this project carries graded
relevance. See ``experiments/evaluation/metrics.py`` for the exact formula.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visualization._common import (  # noqa: E402
    add_results_dir_arg,
    plot_metric_comparison,
    results_dir_from,
)


def main() -> int:
    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_ndcg", description=__doc__)
    )
    args = parser.parse_args()

    path = plot_metric_comparison(
        csv_name="retrieval_metrics.csv",
        prefix="ndcg",
        title="NDCG@K (binary relevance) - vector-only vs. vector + graph",
        ylabel="NDCG@K (mean over queries)",
        filename="ndcg_at_k.png",
        results_dir=results_dir_from(args),
    )
    print(f"wrote {path}" if path else "N/A - no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
