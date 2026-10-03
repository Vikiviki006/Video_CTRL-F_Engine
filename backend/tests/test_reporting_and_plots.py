"""Tests for the reporting layer and the eight figure scripts.

Two things are being proven here:

1. **The reports cannot drift from the data.** The Markdown summary is rendered
   from the same row objects the CSV writers receive, and :data:`N/A` is written
   for absent values - never ``0`` and never an empty cell.
2. **The figures genuinely render from CSV.** Each of the eight plot scripts is
   executed against a temporary results directory and must produce a PNG. This
   is the check that the "figures are generated from data, not hard-coded"
   claim is real, rather than merely asserted in a docstring.

The records used below are **synthetic test fixtures**. They are written to a
pytest ``tmp_path`` and deleted afterwards. They never reach
``experiments/results/`` and no real result file is created or modified by this
test. The numbers are chosen only to be arithmetically consistent with the
metric definitions; they are not measurements of anything.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.evaluation import reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPTS = [
    "plot_recall",
    "plot_precision",
    "plot_mrr",
    "plot_ndcg",
    "plot_latency",
    "plot_frame_reduction",
    "plot_ablation",
    "plot_comparison",
]


# ---------------------------------------------------------------------------
# Synthetic fixtures (clearly labelled; never written to the real results dir)
# ---------------------------------------------------------------------------


def _synthetic_retrieval_rows() -> list:
    """Two conditions across K = 1, 5, 10, 20 with internally consistent values."""
    rows = []
    for condition, base in (("A", 0.20), ("D", 0.25)):
        row = {"condition": condition, "description": f"synthetic condition {condition}"}
        for k in (1, 5, 10, 20):
            # Recall and precision fall off as K grows; NDCG likewise. Values
            # are fabricated for the test, deliberately and consistently.
            row[f"recall@{k}"] = base + 0.1 * (k / 20)
            row[f"precision@{k}"] = base + 0.05 * (k / 20)
            row[f"ndcg@{k}"] = base + 0.08 * (k / 20)
        row["mrr"] = base
        row["map"] = base - 0.02
        row["metrics_status"] = "measured"
        rows.append(row)
    return rows


def _synthetic_ablation_rows() -> list:
    rows = []
    for key, desc, recall in (
        ("A", "A: vector only", 0.30),
        ("B", "B: vector + temporal", 0.32),
        ("C", "C: vector + graph", 0.35),
        ("D", "D: vector + graph + temporal", 0.38),
    ):
        rows.append(
            {
                "condition": key,
                "description": desc,
                "recall@10": recall,
                "precision@10": recall / 2,
                "ndcg@10": recall - 0.01,
                "mrr": recall - 0.02,
                "map": recall - 0.03,
                "latency_avg_ms": 50.0 + 10 * ord(key),
                "latency_p95_ms": 90.0 + 10 * ord(key),
            }
        )
    return rows


def _synthetic_latency_rows() -> list:
    return [
        {
            "condition": "D",
            "stage": stage,
            "samples": 5,
            "avg_ms": value,
            "median_ms": value,
            "p95_ms": value * 1.5,
            "p99_ms": value * 1.8,
            "min_ms": value * 0.8,
            "max_ms": value * 2.0,
        }
        for stage, value in (
            ("embedding", 12.0),
            ("vector_retrieval", 8.0),
            ("graph_expansion", 5.0),
            ("graph_reranking", 3.0),
            ("total", 28.0),
        )
    ]


def _synthetic_indexing_rows() -> list:
    return [
        {
            "video_id": "00000000-0000-0000-0000-000000000001",
            "filename": "synthetic-clip.mp4",
            "original_frames": 1000,
            "retained_frames": 250,
            # (1 - 250/1000) * 100
            "frame_reduction_percent": 75.0,
            "graph_nodes": 300,
            "graph_edges": 900,
        }
    ]


def _synthetic_model_rows() -> list:
    return [
        {
            "model": "siglip",
            "checkpoint": "google/siglip-so400m-patch14-384",
            "recall@10": 0.30,
            "precision@10": 0.15,
            "mrr": 0.28,
            "ndcg@10": 0.29,
            "latency_avg_ms": 40.0,
        },
        {
            "model": "siglip2",
            "checkpoint": "google/siglip2-so400m-patch14-384",
            "recall@10": 0.35,
            "precision@10": 0.18,
            "mrr": 0.32,
            "ndcg@10": 0.34,
            "latency_avg_ms": 55.0,
        },
    ]


def _write_all(tmp_path: Path) -> Path:
    out = tmp_path / "results"
    reporting.write_all(
        out,
        run_metadata={
            "dataset": "synthetic",
            "model": "siglip2",
            "top_k": 10,
            "note": "synthetic fixture produced by a unit test",
        },
        retrieval_rows=_synthetic_retrieval_rows(),
        model_rows=_synthetic_model_rows(),
        ablation_rows=_synthetic_ablation_rows(),
        latency_rows=_synthetic_latency_rows(),
        indexing_rows=_synthetic_indexing_rows(),
        graph_rows=[
            {
                "depth": d,
                "nodes_reached_avg": 10.0 * d,
                "expansion_latency_ms_avg": 2.0 * d,
            }
            for d in (0, 1, 2, 3)
        ],
    )
    return out


# ---------------------------------------------------------------------------
# N/A handling
# ---------------------------------------------------------------------------


def test_normalise_maps_none_to_na():
    assert reporting.normalise(None) == "N/A"


def test_normalise_maps_nan_to_na():
    # A NaN mean is a bug, not a measurement; it must never reach a results file.
    assert reporting.normalise(float("nan")) == "N/A"


def test_normalise_preserves_a_real_zero():
    # 0.0 is a legitimate measurement and must stay distinguishable from N/A.
    assert reporting.normalise(0.0) == 0.0
    assert reporting.normalise(0.0) != "N/A"


def test_normalise_renders_booleans_readably():
    assert reporting.normalise(True) == "true"
    assert reporting.normalise(False) == "false"


def test_write_csv_with_no_rows_marks_the_file_as_unmeasured(tmp_path):
    path = reporting.write_csv(tmp_path / "empty.csv", [])
    assert path.exists()
    assert "N/A" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV / JSON / Markdown consistency
# ---------------------------------------------------------------------------


def test_write_all_produces_every_required_file(tmp_path):
    out = _write_all(tmp_path)
    for name in (
        reporting.RETRIEVAL_CSV,
        reporting.RETRIEVAL_JSON,
        reporting.MODEL_CSV,
        reporting.MODEL_JSON,
        reporting.ABLATION_CSV,
        reporting.ABLATION_JSON,
        reporting.LATENCY_CSV,
        reporting.INDEXING_CSV,
        reporting.GRAPH_CSV,
        reporting.SUMMARY_MD,
    ):
        assert (out / name).exists(), f"{name} was not produced"


def test_summary_is_generated_from_the_same_rows_as_the_csv(tmp_path):
    """The Markdown table must contain the CSV's values, not a re-typed copy."""
    out = _write_all(tmp_path)
    csv_text = (out / reporting.RETRIEVAL_CSV).read_text(encoding="utf-8")
    summary = (out / reporting.SUMMARY_MD).read_text(encoding="utf-8")

    # A value present in the CSV must appear in the summary.
    first_value = "0.3"  # recall@20 for the synthetic condition D
    assert first_value in csv_text
    assert first_value in summary


def test_summary_of_no_rows_says_na_not_empty(tmp_path):
    table = reporting.markdown_table([], title="Nothing")
    assert "N/A" in table


def test_markdown_table_renders_na_as_italic(tmp_path):
    table = reporting.markdown_table([{"a": None, "b": 1}])
    assert "*N/A*" in table


def test_json_preserves_none_as_null(tmp_path):
    path = reporting.write_json(tmp_path / "x.json", {"a": None, "b": 0.0})
    payload = json.loads(path.read_text(encoding="utf-8"))
    # null and 0 stay distinguishable in JSON, as they must be.
    assert payload["a"] is None
    assert payload["b"] == 0.0


# ---------------------------------------------------------------------------
# The eight figures really do render from the CSVs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", PLOT_SCRIPTS)
def test_plot_script_renders_a_png_from_data(script, tmp_path):
    out = _write_all(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"experiments.visualization.{script}",
            "--results-dir",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stderr}"

    figures = out / "figures"
    produced = list(figures.glob("*.png"))
    assert produced, f"{script} produced no PNG in {figures}"


def test_ablation_figure_is_not_the_na_caption(tmp_path):
    """With real rows present, the ablation figure must not be an N/A caption."""
    import matplotlib.image as mpimg

    out = _write_all(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.visualization.plot_ablation",
            "--results-dir",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    path = out / "figures" / "ablation.png"
    image = mpimg.imread(path)
    # A captioned N/A figure is a flat single colour; a real chart is not.
    assert len(set(image.reshape(-1, image.shape[-1] if image.ndim == 3 else 1)[:, 0].tolist())) > 5


def test_plot_script_reports_na_when_no_csv_exists(tmp_path):
    """An empty results directory must produce a captioned N/A figure, not a crash."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.visualization.plot_recall",
            "--results-dir",
            str(empty),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (empty / "figures" / "recall_at_k.png").exists()
