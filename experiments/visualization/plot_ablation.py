"""Figure 7 - Ablation study (conditions A/B/C/D).

Input : experiments/results/ablation_results.csv
Output: experiments/results/figures/ablation.png

    A  vector only
    B  vector + temporal
    C  vector + graph
    D  vector + graph + temporal

Quality panel  : Recall@10 / Precision@10 / NDCG@10 / MRR / MAP
Latency panel  : mean and P95 end-to-end latency per condition

Every value is read from the CSV. A metric that was not measured renders as
``N/A`` in the bar label and contributes no bar, so the figure cannot imply a
zero result.
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
    ("ndcg@10", "NDCG@10"),
    ("mrr", "MRR"),
    ("map", "MAP"),
]


def main() -> int:
    import argparse

    import matplotlib.pyplot as plt

    parser = add_results_dir_arg(
        argparse.ArgumentParser(prog="plot_ablation", description=__doc__)
    )
    args = parser.parse_args()
    rdir = results_dir_from(args)

    rows = read_csv("ablation_results.csv", rdir)
    fig, (ax_q, ax_l) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax in (ax_q, ax_l):
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    usable = [r for r in rows if not is_na(r.get("condition"))]
    if not usable:
        for ax in (ax_q, ax_l):
            ax.set_axis_off()
        path = finish(
            fig,
            "ablation.png",
            title="Ablation: vector / +temporal / +graph / +graph+temporal",
            xlabel="condition",
            ylabel="metric",
            results_dir=rdir,
            na_reason="ablation_results.csv not present. Re-run with --ablation.",
        )
        print(f"wrote {path} (N/A caption)")
        return 0

    usable.sort(key=lambda r: str(r.get("condition")))
    conditions = [str(r.get("condition")) for r in usable]
    colors = [COLORS.get(c, "#0ea5e9") for c in conditions]
    positions = list(range(len(usable)))
    width = 0.8 / max(1, len(METRICS))

    for index, (key, label) in enumerate(METRICS):
        values = [to_float(r.get(key)) for r in usable]
        if all(v is None for v in values):
            continue
        xs = [p + (index - (len(METRICS) - 1) / 2) * width for p in positions]
        ys = [v if v is not None else 0.0 for v in values]
        ax_q.bar(xs, ys, width=width, label=label, color=plt.cm.viridis(index / max(1, len(METRICS) - 1)))
        for x, y, value in zip(xs, ys, values):
            if value is not None:
                ax_q.text(x, y, f"{value:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax_q.set_xticks(positions)
    ax_q.set_xticklabels(conditions)
    ax_q.set_ylim(0.0, 1.1)
    ax_q.set_xlabel("ablation condition")
    ax_q.set_ylabel("metric value (higher is better)")
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax_q.set_axisbelow(True)

    # Latency panel
    lat_means = [to_float(r.get("latency_avg_ms")) for r in usable]
    lat_p95 = [to_float(r.get("latency_p95_ms")) for r in usable]
    if any(v is not None for v in lat_means):
        ax_l.bar(
            [p - 0.2 for p in positions],
            [v if v is not None else 0.0 for v in lat_means],
            width=0.38,
            label="mean",
            color=colors,
            alpha=0.9,
        )
        ax_l.bar(
            [p + 0.2 for p in positions],
            [v if v is not None else 0.0 for v in lat_p95],
            width=0.38,
            label="P95",
            color=colors,
            alpha=0.4,
        )
        ax_l.legend(fontsize=8)
    else:
        ax_l.text(
            0.5,
            0.5,
            "N/A - no latency was measured",
            ha="center",
            va="center",
            transform=ax_l.transAxes,
            color="#94a3b8",
            fontsize=12,
        )
    ax_l.set_xticks(positions)
    ax_l.set_xticklabels(conditions)
    ax_l.set_xlabel("ablation condition")
    ax_l.set_ylabel("end-to-end latency (ms, lower is better)")
    ax_l.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax_l.set_axisbelow(True)

    path = finish(
        fig,
        "ablation.png",
        title="Ablation: A vector | B +temporal | C +graph | D +graph+temporal",
        xlabel="condition",
        ylabel="metric",
        results_dir=rdir,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
