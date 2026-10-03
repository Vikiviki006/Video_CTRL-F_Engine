"""Figure 3 - MRR: vector-only baseline vs. vector + graph.

Input : experiments/results/retrieval_metrics.csv  (column ``mrr``)
Output: experiments/results/figures/mrr.png

MRR is a single number per condition rather than a curve over K, so this figure
is a grouped bar chart. ``Mrr = mean_q 1 / rank_of_first_relevant(q)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visualization._common import (  # noqa: E402
    COLORS,
    add_results_dir_arg,
    finish,
    is_na,
    new_figure,
    read_csv,
    results_dir_from,
    to_float,
)


def main() -> int:
    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_mrr", description=__doc__)
    )
    args = parser.parse_args()
    rdir = results_dir_from(args)

    rows = read_csv("retrieval_metrics.csv", rdir)
    fig, ax = new_figure(figsize=(7.0, 4.5))

    usable = [r for r in rows if not is_na(r.get("mrr"))]
    if not usable:
        path = finish(
            fig,
            "mrr.png",
            title="MRR - mean reciprocal rank of the first relevant result",
            xlabel="configuration",
            ylabel="MRR (higher is better)",
            results_dir=rdir,
            na_reason="retrieval_metrics.csv has no measured 'mrr' column. "
            "Run the evaluation first.",
        )
        print(f"wrote {path} (N/A caption)")
        return 0

    labels = [str(r.get("description") or r.get("condition") or "run") for r in usable]
    values = [to_float(r.get("mrr")) or 0.0 for r in usable]
    colors = [
        COLORS.get(str(r.get("condition")), "#0ea5e9") for r in usable
    ]

    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    path = finish(
        fig,
        "mrr.png",
        title="MRR - mean reciprocal rank of the first relevant result",
        xlabel="configuration",
        ylabel="MRR (higher is better)",
        results_dir=rdir,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
