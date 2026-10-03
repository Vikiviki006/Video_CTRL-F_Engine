"""Figure 8 - SigLIP vs SigLIP-2.

Input : experiments/results/model_comparison.csv
Output: experiments/results/figures/model_comparison.png

Both checkpoints must be indexed over identical frames and scored on the same
queries, ground truth, top-K and metrics for this comparison to mean anything.
When only one checkpoint has been evaluated, the figure is written with an
explicit ``N/A`` caption and the CSV records the reason - it is never filled
with a single model plus an invented second series.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visualization._common import (  # noqa: E402
    COLORS,
    add_results_dir_arg,
    finish,
    is_na,
    read_csv,
    results_dir_from,
    to_float,
)

METRICS = [
    ("recall@10", "Recall@10"),
    ("precision@10", "Precision@10"),
    ("mrr", "MRR"),
    ("ndcg@10", "NDCG@10"),
    ("latency_avg_ms", "Latency avg (ms)"),
]


def main() -> int:
    import argparse

    import matplotlib.pyplot as plt

    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_comparison", description=__doc__)
    )
    args = parser.parse_args()
    rdir = results_dir_from(args)

    rows = read_csv("model_comparison.csv", rdir)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    def measured(r):
        return any(not is_na(r.get(k)) and to_float(r.get(k)) is not None for k, _ in METRICS)

    usable = [r for r in rows if measured(r)]
    if not usable:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        path = finish(
            fig,
            "model_comparison.png",
            title="SigLIP vs SigLIP-2 (identical frames, queries, top-K, metrics)",
            xlabel="metric",
            ylabel="value",
            results_dir=rdir,
            na_reason="model_comparison.csv has no measured values. Both "
            "checkpoints must be indexed and scored before this comparison exists.",
        )
        print(f"wrote {path} (N/A caption)")
        return 0

    models = [str(r.get("model")) for r in usable]
    positions = list(range(len(METRICS)))
    width = 0.8 / max(1, len(usable))

    for index, row in enumerate(usable):
        model = str(row.get("model"))
        xs = [p + (index - (len(usable) - 1) / 2) * width for p in positions]
        ys = []
        for key, _label in METRICS:
            value = to_float(row.get(key))
            ys.append(value if value is not None else 0.0)
        bars = ax.bar(
            xs,
            ys,
            width=width,
            label=f"{model} ({row.get('checkpoint') or 'n/a'})",
            color=COLORS.get(model, "#0ea5e9"),
            alpha=0.9,
        )
        for bar, (key, _label) in zip(bars, METRICS):
            raw = row.get(key)
            if is_na(raw) or to_float(raw) is None:
                bar.set_color("#1e293b")
                bar.set_label("")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#94a3b8",
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{to_float(raw):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _k, label in METRICS], rotation=12, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    path = finish(
        fig,
        "model_comparison.png",
        title="SigLIP vs SigLIP-2 (identical frames, queries, top-K, metrics)",
        xlabel="metric",
        ylabel="value",
        results_dir=rdir,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
