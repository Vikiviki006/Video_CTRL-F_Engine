"""Figure 6 - Adaptive keyframe extraction: original / candidate / retained.

Input : experiments/results/indexing_results.csv
Output: experiments/results/figures/frame_reduction.png

The reduction percentage shown on each bar is read from the measured
``frame_reduction_percent`` column, itself computed as
``(1 - retained / original) * 100`` by ``metrics.frame_reduction_percent``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visualization._common import (  # noqa: E402
    add_results_dir_arg,
    plot_frame_reduction,
    results_dir_from,
)


def main() -> int:
    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_frame_reduction", description=__doc__)
    )
    args = parser.parse_args()

    path = plot_frame_reduction(results_dir=results_dir_from(args))
    print(f"wrote {path}" if path else "N/A - no data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
