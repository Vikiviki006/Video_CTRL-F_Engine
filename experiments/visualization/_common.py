"""Shared helpers for the VideoCtrl-F figure scripts.

Two hard rules enforced here:

1. **Figures are drawn only from CSV/JSON files on disk.** Every plot script
   reads ``experiments/results/*.csv`` through :func:`read_csv` and renders what
   it finds. No script contains a hard-coded metric, a hard-coded latency, or a
   hard-coded frame count.

2. **A missing or unmeasured series is shown as "N/A", not as zero.** A zero
   line on a chart reads as "measured, and it was zero", which is a different
   and false claim. :func:`series_or_na` keeps the distinction visible.
"""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

# Headless backend: these scripts run on CI and in Docker with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

#: Value written by ``reporting.normalise`` for anything not measured.
NA_TOKENS = {"N/A", "n/a", "", "NA", "None", "nan", "NaN"}

#: Shared colour language, so a series keeps its colour across every figure.
COLORS: Dict[str, str] = {
    "vector": "#64748b",  # slate - the baseline
    "graph": "#0ea5e9",  # sky   - vector + graph
    "temporal": "#f59e0b",  # amber - temporal
    "A": "#64748b",
    "B": "#f59e0b",
    "C": "#0ea5e9",
    "D": "#22c55e",
    "siglip": "#a855f7",
    "siglip2": "#0ea5e9",
}


def is_na(value: Any) -> bool:
    """True when a cell represents "not measured"."""
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return str(value).strip() in NA_TOKENS


def to_float(value: Any) -> Optional[float]:
    """Parse a CSV cell into a float, or ``None`` when it is not measured."""
    if is_na(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def read_csv(name: str, results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read a results CSV.

    Returns ``[]`` when the file is absent, which is the signal for a figure
    script to report "N/A - run the evaluation first" rather than to plot an
    empty axis that looks like a real measurement of nothing.
    """
    path = (results_dir or RESULTS_DIR) / name
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("#"):
        # reporting.write_csv writes a '#' comment for an empty result set.
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series_or_na(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    where: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[Optional[float]]]:
    """Extract an (x, y) series, preserving gaps as ``None``.

    Gaps stay ``None`` so matplotlib draws a break instead of interpolating
    across missing measurements.
    """
    xs: List[float] = []
    ys: List[Optional[float]] = []
    for row in rows:
        if where and not all(str(row.get(k)) == str(v) for k, v in where.items()):
            continue
        x = to_float(row.get(x_key))
        if x is None:
            continue
        xs.append(x)
        ys.append(to_float(row.get(y_key)))
    return xs, ys


def grouped_series(
    rows: Sequence[Dict[str, Any]],
    group_key: str,
    x_key: str,
    y_key: str,
) -> Dict[str, Tuple[List[float], List[Optional[float]]]]:
    """Extract one series per distinct value of ``group_key``."""
    out: Dict[str, Tuple[List[float], List[Optional[float]]]] = {}
    for row in rows:
        label = str(row.get(group_key))
        out.setdefault(label, ([], []))
    for label in out:
        xs, ys = series_or_na(rows, x_key, y_key, where={group_key: label})
        out[label] = (xs, ys)
    return out


def finish(
    fig: plt.Figure,
    filename: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    results_dir: Optional[Path] = None,
    na_reason: Optional[str] = None,
) -> Optional[Path]:
    """Label, save and close a figure. Returns the path, or ``None`` for N/A.

    When ``na_reason`` is given the axes are cleared and the figure carries a
    wrapped ``N/A`` caption instead. Leaving a bare, labelled-but-empty axis
    would be worse than useless: it looks like a real measurement of nothing.
    """
    if na_reason:
        for ax in fig.get_axes():
            ax.set_axis_off()
        message = f"N/A - {na_reason}"
        wrapped = textwrap.fill(message, width=58)
        fig.text(
            0.5,
            0.5,
            wrapped,
            ha="center",
            va="center",
            fontsize=13,
            color="#64748b",
            transform=fig.transFigure,
            bbox={"facecolor": "#f1f5f9", "edgecolor": "#cbd5e1", "pad": 14},
        )
    else:
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))

    target_dir = (results_dir or RESULTS_DIR) / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def new_figure(figsize: Tuple[float, float] = (7.0, 4.5)) -> plt.Figure:
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def ks_from_metric_name(name: str) -> Optional[int]:
    """Extract K from a column name such as ``recall@10``. Returns None if absent."""
    if not name or "@" not in name:
        return None
    try:
        return int(name.split("@", 1)[1])
    except ValueError:
        return None


def metric_columns(rows: Sequence[Dict[str, Any]], prefix: str) -> List[str]:
    """All column names like ``<prefix>@K``, ordered by K."""
    if not rows:
        return []
    found = []
    for key in rows[0].keys():
        if key and key.startswith(prefix + "@"):
            k = ks_from_metric_name(key)
            if k is not None:
                found.append((k, key))
    return [key for _k, key in sorted(found)]


def add_results_dir_arg(parser: Any) -> Any:
    """Add the shared ``--results-dir`` option to a plot script's parser."""
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Directory holding the results CSVs. Defaults to "
            "experiments/results/. Useful for rendering a figure from an "
            "alternative results directory without touching the real one."
        ),
    )
    return parser


def results_dir_from(args: Any):
    """Resolve ``--results-dir`` into a Path, or None for the default."""
    value = getattr(args, "results_dir", None)
    return Path(value) if value else None


def banner(text: str) -> None:
    """Print a short status line for CLI use."""
    print(text)


def plot_metric_comparison(
    *,
    csv_name: str,
    prefix: str,
    title: str,
    ylabel: str,
    filename: str,
    baseline_condition: str = "A",
    graph_condition: str = "D",
    results_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Shared implementation for the four Recall/Precision/MRR/NDCG figures.

    Each of those figures has identical shape - one metric at K = 1, 5, 10, 20,
    comparing the vector-only baseline against vector + graph - so they share
    this function rather than four near-identical copies.

    Reads ``<prefix>@K`` columns from ``experiments/results/<csv_name>`` and
    plots one line per condition. Gaps stay broken. If the CSV is absent the
    figure is still written, but captioned ``N/A`` so a reader is never left
    staring at an empty axis that could be mistaken for a measurement.
    """
    rows = read_csv(csv_name, results_dir)
    columns = metric_columns(rows, prefix)

    fig, ax = new_figure()
    if not rows or not columns:
        return finish(
            fig,
            filename,
            title=title,
            xlabel="K (cut-off)",
            ylabel=ylabel,
            results_dir=results_dir,
            na_reason=f"{csv_name} not present or has no {prefix}@K columns. "
            f"Run: python -m experiments.evaluation.evaluate_retrieval",
        )

    condition_key = "condition" if "condition" in rows[0] else None
    grouped = (
        grouped_series(rows, condition_key, "condition", "unused")
        if condition_key
        else {}
    )

    if condition_key:
        labels = sorted({str(r.get(condition_key)) for r in rows})
        for label in labels:
            xs, ys = [], []
            for col in columns:
                k = ks_from_metric_name(col)
                values = [to_float(r.get(col)) for r in rows if str(r.get(condition_key)) == label]
                # A condition is measured only if every K is present.
                if any(v is None for v in values) or not values:
                    xs = []
                    break
                xs.append(float(k))
                ys.append(values[0])
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=_condition_label(label, rows, condition_key),
                color=COLORS.get(label),
            )
    else:
        xs = [float(ks_from_metric_name(c)) for c in columns]
        ys = [to_float(r.get(c)) for r in rows]
        if all(y is not None for y in ys):
            ax.plot(xs, ys, marker="o", linewidth=2, label="measured")

    ax.set_xlabel("K (cut-off)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([float(ks_from_metric_name(c)) for c in columns])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=9)
    return finish(
        fig,
        filename,
        title=title,
        xlabel="K (cut-off)",
        ylabel=ylabel,
        results_dir=results_dir,
    )


def _condition_label(
    label: str, rows: Sequence[Dict[str, Any]], condition_key: str
) -> str:
    """Use the human-readable description when the CSV provides one."""
    for row in rows:
        if str(row.get(condition_key)) == label:
            desc = row.get("description")
            if desc and not is_na(desc):
                return str(desc)
    return f"condition {label}"


def plot_latency_breakdown(
    *, csv_name: str = "latency_results.csv", results_dir: Optional[Path] = None
) -> Optional[Path]:
    """Latency per pipeline stage: embedding / vector / expansion / rerank / total.

    Mean with a P95 overlay, read from the measured CSV. Both bars use only
    measured values; a stage with no samples is omitted rather than drawn as 0.
    """
    rows = read_csv(csv_name, results_dir)
    fig, ax = new_figure(figsize=(8.0, 4.5))
    if not rows:
        return finish(
            fig,
            "latency_breakdown.png",
            title="Query latency by pipeline stage",
            xlabel="mean milliseconds (lower is better)",
            ylabel="Pipeline stage",
            results_dir=results_dir,
            na_reason=f"{csv_name} not present. Re-run the evaluation with a live database.",
        )

    stage_order = [
        "embedding",
        "vector_retrieval",
        "graph_expansion",
        "graph_reranking",
        "total",
    ]
    by_stage: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stage = str(row.get("stage"))
        by_stage.setdefault(stage, row)

    stages, means, p95s = [], [], []
    for stage in stage_order:
        row = by_stage.get(stage)
        if row is None:
            continue
        mean = to_float(row.get("avg_ms"))
        p95 = to_float(row.get("p95_ms"))
        if mean is None:
            continue
        stages.append(stage)
        means.append(mean)
        p95s.append(p95)

    if not stages:
        return finish(
            fig,
            "latency_breakdown.png",
            title="Query latency by pipeline stage",
            xlabel="mean milliseconds (lower is better)",
            ylabel="Pipeline stage",
            results_dir=results_dir,
            na_reason="no stage in the CSV carried a measured average",
        )

    positions = range(len(stages))
    ax.barh(list(positions), means, color="#0ea5e9", alpha=0.85, label="mean")
    ax.barh(
        list(positions),
        [p if p is not None else 0.0 for p in p95s],
        color="#0f172a",
        alpha=0.35,
        height=0.32,
        label="P95",
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(stages)
    ax.legend(fontsize=9)
    for index, (mean, p95) in enumerate(zip(means, p95s)):
        label = f"{mean:.1f} ms"
        if p95 is not None:
            label += f"  (p95 {p95:.1f})"
        ax.text(mean, index, f"  {label}", va="center", fontsize=8)
    return finish(
        fig,
        "latency_breakdown.png",
        title="Query latency by pipeline stage",
        xlabel="milliseconds",
        ylabel="Pipeline stage",
        results_dir=results_dir,
    )


def plot_frame_reduction(
    *, csv_name: str = "indexing_results.csv", results_dir: Optional[Path] = None
) -> Optional[Path]:
    """Original / candidate / retained frame counts plus the reduction %.

    ``candidate`` and ``retained`` come from the measured indexing CSV. The
    original count is the source video's true decoded frame count, and the
    reduction percentage is computed from those two measured numbers by
    ``metrics.frame_reduction_percent`` - never supplied by hand.
    """
    rows = read_csv(csv_name, results_dir)
    fig, ax = new_figure(figsize=(8.0, 4.5))
    if not rows:
        return finish(
            fig,
            "frame_reduction.png",
            title="Adaptive keyframe extraction: frames indexed vs. decoded",
            xlabel="video",
            ylabel="frame count",
            results_dir=results_dir,
            na_reason=f"{csv_name} not present. Re-run with --index-metrics.",
        )

    usable = [
        r
        for r in rows
        if to_float(r.get("original_frames")) is not None
        and to_float(r.get("retained_frames")) is not None
    ]
    if not usable:
        return finish(
            fig,
            "frame_reduction.png",
            title="Adaptive keyframe extraction: frames indexed vs. decoded",
            xlabel="video",
            ylabel="frame count",
            results_dir=results_dir,
            na_reason="no row carried both an original and a retained frame count",
        )

    labels = [str(r.get("filename") or r.get("video_id"))[:28] for r in usable]
    originals = [to_float(r.get("original_frames")) or 0.0 for r in usable]
    retained = [to_float(r.get("retained_frames")) or 0.0 for r in usable]
    candidates = [to_float(r.get("candidate_frames")) for r in usable]

    positions = list(range(len(labels)))
    width = 0.38
    ax.bar(
        [p - width / 2 for p in positions],
        originals,
        width,
        label="original (decoded)",
        color="#334155",
    )
    if any(c is not None for c in candidates):
        values = [c if c is not None else 0.0 for c in candidates]
        ax.bar(
            positions,
            values,
            width,
            label="candidate",
            color="#94a3b8",
        )
        ax.bar(
            [p + width / 2 for p in positions],
            retained,
            width,
            label="retained (indexed)",
            color="#22c55e",
        )
    else:
        ax.bar(
            positions,
            retained,
            width,
            label="retained (indexed)",
            color="#22c55e",
        )

    for index, row in enumerate(usable):
        pct = to_float(row.get("frame_reduction_percent"))
        if pct is not None:
            ax.annotate(
                f"-{pct:.1f}%",
                (index, originals[index]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#16a34a",
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.legend(fontsize=9)
    return finish(
        fig,
        "frame_reduction.png",
        title="Adaptive keyframe extraction: frames indexed vs. decoded",
        xlabel="video",
        ylabel="frame count",
        results_dir=results_dir,
    )
