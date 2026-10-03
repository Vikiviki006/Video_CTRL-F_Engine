"""Figure 5 - Latency breakdown per pipeline stage.

Input : experiments/results/latency_results.csv
Output: experiments/results/figures/latency_breakdown.png

Stages: query embedding / vector retrieval / graph expansion / graph re-ranking /
total. Mean with a P95 overlay, both measured. A stage with no samples is left
off the chart rather than drawn as zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visualization._common import (  # noqa: E402
    add_results_dir_arg,
    plot_latency_breakdown,
    results_dir_from,
)


def main() -> int:
    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_latency", description=__doc__)
    )
    args = parser.parse_args()

    path = plot_latency_breakdown(results_dir=results_dir_from(args))
    print(f"wrote {path}" if path else "N/A - no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
