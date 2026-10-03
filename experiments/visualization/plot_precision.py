"""Figure 2 - Precision@K: vector-only baseline vs. vector + graph.

Input : experiments/results/retrieval_metrics.csv  (columns ``precision@1`` ... ``precision@20``)
Output: experiments/results/figures/precision_at_k.png
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
        argparse.ArgumentParser(prog="plot_precision", description=__doc__)
    )
    args = parser.parse_args()

    path = plot_metric_comparison(
        csv_name="retrieval_metrics.csv",
        prefix="precision",
        title="Precision@K - vector-only baseline vs. vector + graph",
        ylabel="Precision@K (mean over queries)",
        filename="precision_at_k.png",
        results_dir=results_dir_from(args),
    )
    print(f"wrote {path}" if path else "N/A - no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
