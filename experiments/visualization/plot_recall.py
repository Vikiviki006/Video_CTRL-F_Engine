"""Figure 1 - Recall@K: vector-only baseline vs. vector + graph.

Input : experiments/results/retrieval_metrics.csv  (columns ``recall@1`` ... ``recall@20``)
Output: experiments/results/figures/recall_at_k.png

No value is hard-coded. When the CSV is missing the figure is written with an
explicit ``N/A`` caption so it can never be mistaken for a measurement.
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
        argparse.ArgumentParser(prog="plot_recall", description=__doc__)
    )
    args = parser.parse_args()

    path = plot_metric_comparison(
        csv_name="retrieval_metrics.csv",
        prefix="recall",
        title="Recall@K - vector-only baseline vs. vector + graph",
        ylabel="Recall@K (mean over queries)",
        filename="recall_at_k.png",
        results_dir=results_dir_from(args),
    )
    print(f"wrote {path}" if path else "N/A - no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
