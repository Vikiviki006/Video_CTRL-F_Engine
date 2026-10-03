"""CSV / JSON / Markdown output for the evaluation study.

One rule governs this module: **every table is generated from the same
in-memory records that get written to CSV and JSON.** There is no path by which
a number can reach the Markdown summary without also being in a CSV. That
makes hand-editing a result impossible by construction, which is the
mechanism by which fabricated numbers usually enter a report.

Absent measurements are written as the literal string ``N/A``, never as 0 and
never as an empty cell. A reader can always tell "not measured" from "measured
as zero".
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

NA = "N/A"

#: Canonical output filenames required by the specification.
RETRIEVAL_CSV = "retrieval_metrics.csv"
RETRIEVAL_JSON = "retrieval_metrics.json"
MODEL_CSV = "model_comparison.csv"
MODEL_JSON = "model_comparison.json"
ABLATION_CSV = "ablation_results.csv"
ABLATION_JSON = "ablation_results.json"
LATENCY_CSV = "latency_results.csv"
INDEXING_CSV = "indexing_results.csv"
GRAPH_CSV = "graph_results.csv"
SUMMARY_MD = "SUMMARY.md"


def normalise(value: Any) -> Any:
    """Make a value safe for CSV/JSON and honest about absence.

    * ``None``            -> ``"N/A"``
    * ``float('nan')``    -> ``"N/A"`` (a NaN mean is a bug, not a result)
    * ``bool``            -> ``"true"`` / ``"false"``
    * ``dict`` / ``list`` -> compact JSON, so a column never silently loses
      structure
    """
    if value is None:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return NA
        return round(value, 6)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _ordered_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Union of keys, first-seen order preserved, so column order is stable."""
    seen: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Write records to CSV. An empty row list still produces a headerless,
    clearly-marked file rather than an empty file that looks like success."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        target.write_text(
            "# N/A - no measured records were produced for this experiment.\n",
            encoding="utf-8",
        )
        return target

    fieldnames = _ordered_fieldnames(rows)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: normalise(row.get(k)) for k in fieldnames})
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON payload with ``None`` preserved as ``null``.

    JSON keeps the difference between "measured as zero" and "not measured"
    (``0`` vs ``null``); CSV renders both through :data:`NA` for the same
    reason.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "value"):  # enum
        return value.value
    return str(value)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def markdown_table(
    rows: Sequence[Dict[str, Any]],
    columns: Optional[Sequence[str]] = None,
    title: str = "",
) -> str:
    """Render records as a GitHub-flavoured Markdown table.

    Generated purely from the passed records, which are the same objects that
    were written to CSV. ``N/A`` cells are rendered as the italic ``N/A`` so a
    missing measurement is visually obvious in a rendered table.
    """
    if not rows:
        heading = f"### {title}\n\n" if title else ""
        return f"{heading}_N/A - no measured records were produced for this experiment._\n"

    fieldnames = list(columns) if columns else _ordered_fieldnames(rows)
    lines: List[str] = []
    if title:
        lines.append(f"### {title}\n")

    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        cells: List[str] = []
        for key in fieldnames:
            raw = normalise(row.get(key))
            if raw == NA:
                cells.append("*N/A*")
            else:
                cells.append(f"`{raw}`" if isinstance(raw, (int, float)) else str(raw))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_summary(
    *,
    run_metadata: Dict[str, Any],
    retrieval_rows: Sequence[Dict[str, Any]],
    model_rows: Sequence[Dict[str, Any]],
    ablation_rows: Sequence[Dict[str, Any]],
    latency_rows: Sequence[Dict[str, Any]],
    indexing_rows: Sequence[Dict[str, Any]],
    graph_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Assemble the auto-generated Markdown summary.

    Every table below is rendered from the same row objects that the CSV
    writers received, so the summary cannot disagree with the data files.
    """
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parts: List[str] = []
    parts.append("# VideoCtrl-F Evaluation Summary\n")
    parts.append(
        f"_Generated {generated} by "
        "`experiments.evaluation.reporting.build_summary`._\n\n"
        "Every table on this page is rendered from the same records that were "
        "written to the CSV files in this directory. No value here is typed by "
        "hand. A cell reading **N/A** means the measurement was not taken - it "
        "is not a zero.\n"
    )

    parts.append("## Run conditions\n")
    parts.append(markdown_table([run_metadata], title="Run metadata"))

    parts.append("\n## Retrieval metrics\n")
    parts.append(
        markdown_table(
            retrieval_rows,
            title="Vector vs. vector + graph",
        )
    )

    parts.append("\n## Ablation (A/B/C/D)\n")
    parts.append(markdown_table(ablation_rows, title="Ablation conditions"))

    parts.append("\n## Graph depth sweep\n")
    parts.append(markdown_table(graph_rows, title="Graph depth"))

    parts.append("\n## Latency by pipeline stage\n")
    parts.append(markdown_table(latency_rows, title="Latency"))

    parts.append("\n## Indexing metrics\n")
    parts.append(markdown_table(indexing_rows, title="Indexing"))

    parts.append("\n## Model comparison\n")
    parts.append(markdown_table(model_rows, title="SigLIP vs SigLIP-2"))

    parts.append("\n## Files\n")
    for name in (
        RETRIEVAL_CSV,
        RETRIEVAL_JSON,
        MODEL_CSV,
        MODEL_JSON,
        ABLATION_CSV,
        ABLATION_JSON,
        LATENCY_CSV,
        INDEXING_CSV,
        GRAPH_CSV,
    ):
        exists = (output_dir / name).exists()
        parts.append(f"- `{name}` - {'present' if exists else 'NOT PRODUCED'}")
    parts.append("")

    return "\n".join(parts)


def write_all(
    output_dir: str | Path,
    *,
    run_metadata: Dict[str, Any],
    retrieval_rows: Sequence[Dict[str, Any]],
    model_rows: Sequence[Dict[str, Any]],
    ablation_rows: Sequence[Dict[str, Any]],
    latency_rows: Sequence[Dict[str, Any]],
    indexing_rows: Sequence[Dict[str, Any]],
    graph_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    """Write every CSV/JSON artefact and the generated summary.

    Returns a mapping of logical name -> written path, so the CLI can report
    exactly what it produced (and, by omission, what it did not).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {
        RETRIEVAL_CSV: write_csv(out / RETRIEVAL_CSV, retrieval_rows),
        RETRIEVAL_JSON: write_json(
            out / RETRIEVAL_JSON,
            {"run": run_metadata, "rows": list(retrieval_rows)},
        ),
        MODEL_CSV: write_csv(out / MODEL_CSV, model_rows),
        MODEL_JSON: write_json(
            out / MODEL_JSON,
            {"run": run_metadata, "rows": list(model_rows)},
        ),
        ABLATION_CSV: write_csv(out / ABLATION_CSV, ablation_rows),
        ABLATION_JSON: write_json(
            out / ABLATION_JSON,
            {"run": run_metadata, "rows": list(ablation_rows)},
        ),
        LATENCY_CSV: write_csv(out / LATENCY_CSV, latency_rows),
        INDEXING_CSV: write_csv(out / INDEXING_CSV, indexing_rows),
        GRAPH_CSV: write_csv(out / GRAPH_CSV, graph_rows),
    }

    summary = build_summary(
        run_metadata=run_metadata,
        retrieval_rows=retrieval_rows,
        model_rows=model_rows,
        ablation_rows=ablation_rows,
        latency_rows=latency_rows,
        indexing_rows=indexing_rows,
        graph_rows=graph_rows,
        output_dir=out,
    )
    summary_path = out / SUMMARY_MD
    summary_path.write_text(summary, encoding="utf-8")
    written[SUMMARY_MD] = summary_path
    return written
