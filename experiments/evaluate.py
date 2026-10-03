"""Reproducible baseline-vs-proposed evaluation for VideoCtrl-F.

The evaluator deliberately does not call a model or invent measurements.  A
system adapter (or a captured predictions file) supplies measured outputs for
the same examples and LLM conditions.  The resulting JSON is suitable for
tables and figures.

Prediction files are JSON arrays (or JSONL) with one object per example::

    {"id": "q1", "retrieved": [{"id": "frame-1", "score": .9}],
     "answer_faithfulness": 1, "evidence_ids": ["frame-1"],
     "contradiction_detected": true, "relationship_labels": ["before"],
     "latency_ms": 42, "input_tokens": 100, "output_tokens": 20}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

RESULTS_DIR = Path(__file__).parent / "results"


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def load_dataset(path: str | Path = "evaluation_dataset.csv") -> list[dict[str, Any]]:
    """Load CSV annotations, retaining optional research metric columns.

    Supported aliases include ``ground_truth_ids``/``relevant_ids`` and
    ``ground_truth_timestamp``/``relevant_timestamps``.  List values may be
    JSON arrays or ``|``-separated strings.
    """
    source = Path(path)
    if not source.is_absolute():
        source = Path(__file__).parent / source
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [_normalise_example(row, index) for index, row in enumerate(rows)]


def _split(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    text = str(value)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [item for item in text.split("|") if item]


def _normalise_example(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    relevant_ids = _split(row.get("relevant_ids", row.get("ground_truth_ids")))
    timestamps = _split(
        row.get("relevant_timestamps", row.get("ground_truth_timestamp"))
    )
    if not relevant_ids and row.get("ground_truth_timestamp") not in (None, ""):
        timestamps = [_float(item) for item in timestamps]
    return {
        **dict(row),
        "id": str(row.get("id") or row.get("query_id") or index),
        "query": str(row.get("query", "")),
        "relevant_ids": [str(item) for item in relevant_ids],
        "relevant_timestamps": [float(item) for item in timestamps],
        "tolerance_seconds": float(row.get("tolerance_seconds") or 2.0),
        "expected_answer": row.get("expected_answer") or None,
        "expected_contradiction": _optional_bool(row.get("expected_contradiction")),
        "relevant_relationships": [
            str(item) for item in _split(row.get("relevant_relationships"))
        ],
    }


def _optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return str(value).strip().lower() in {"1", "true", "yes"}


def _retrieved_ids(result: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("id", item.get("frame_id", item.get("document_id"))))
        if isinstance(item, Mapping)
        else str(item)
        for item in result.get("retrieved", [])
    ]


def _retrieved_timestamps(result: Mapping[str, Any]) -> list[float]:
    values = []
    for item in result.get("retrieved", []):
        value = item.get("timestamp_seconds") if isinstance(item, Mapping) else None
        if value is not None:
            values.append(float(value))
    return values


def _relevant(result: Mapping[str, Any], example: Mapping[str, Any]) -> list[bool]:
    ids = set(example.get("relevant_ids", []))
    timestamps = example.get("relevant_timestamps", [])
    tolerance = float(example.get("tolerance_seconds", 2.0))
    output = []
    for item in result.get("retrieved", []):
        item_id = str(item.get("id", item.get("frame_id", ""))) if isinstance(item, Mapping) else str(item)
        timestamp = item.get("timestamp_seconds") if isinstance(item, Mapping) else None
        output.append(item_id in ids or any(
            timestamp is not None and abs(float(timestamp) - gt) <= tolerance
            for gt in timestamps
        ))
    return output


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _retrieval_metrics(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for result, example in rows:
        hits = _relevant(result, example)
        relevant_count = len(example.get("relevant_ids", [])) or len(example.get("relevant_timestamps", []))
        precisions.append(sum(hits) / len(hits) if hits else 0.0)
        recalls.append(sum(hits) / relevant_count if relevant_count else 0.0)
        first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum((2**int(hit) - 1) / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(relevant_count, len(hits))))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "retrieval_precision": _mean(precisions),
        "retrieval_recall": _mean(recalls),
        "mrr": _mean(reciprocal_ranks),
        "ndcg": _mean(ndcgs),
    }


def _accuracy(values: Iterable[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return _mean(values) if values else None


def evaluate_system(
    name: str,
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    """Score predictions and preserve per-query measurements for analysis."""
    by_id = {str(row.get("id")): row for row in predictions}
    missing = [str(example["id"]) for example in examples if str(example["id"]) not in by_id]
    if missing:
        raise ValueError(f"{name} predictions missing example ids: {', '.join(missing)}")
    rows = [(by_id[str(example["id"])], example) for example in examples]
    metrics = _retrieval_metrics(rows)
    faithfulness = _accuracy(
        _float(result.get("answer_faithfulness"))
        for result, _ in rows
    )
    coverage = _accuracy(
        _evidence_coverage(result, example) for result, example in rows
    )
    contradiction = _contradiction_accuracy(rows)
    relationship = _relationship_accuracy(rows)
    latency = [float(result["latency_ms"]) for result, _ in rows if result.get("latency_ms") is not None]
    tokens = [
        float(result.get("input_tokens", 0)) + float(result.get("output_tokens", 0))
        for result, _ in rows
        if result.get("input_tokens") is not None or result.get("output_tokens") is not None
    ]
    metrics.update({
        "answer_faithfulness": faithfulness,
        "evidence_coverage": coverage,
        "contradiction_detection_accuracy": contradiction,
        "relationship_classification_accuracy": relationship,
        "latency_ms_mean": _mean(latency),
        "latency_ms_p50": statistics.median(latency) if latency else 0.0,
        "token_consumption_mean": _mean(tokens),
    })
    return {"name": name, "condition": dict(condition), "metrics": metrics, "queries": list(rows_to_records(rows))}


def _evidence_coverage(result: Mapping[str, Any], example: Mapping[str, Any]) -> float | None:
    expected = set(example.get("relevant_ids", []))
    if not expected:
        return None
    actual = {str(item) for item in result.get("evidence_ids", [])}
    return len(expected & actual) / len(expected)


def _contradiction_accuracy(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> float | None:
    values = []
    for result, example in rows:
        expected = example.get("expected_contradiction")
        actual = result.get("contradiction_detected")
        if expected is not None and actual is not None:
            values.append(float(bool(expected) == bool(actual)))
    return _accuracy(values)


def _relationship_accuracy(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> float | None:
    values = []
    for result, example in rows:
        expected = set(example.get("relevant_relationships", []))
        actual = set(str(item) for item in result.get("relationship_labels", []))
        if expected:
            values.append(float(expected == actual))
    return _accuracy(values)


def rows_to_records(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Iterable[dict[str, Any]]:
    for result, example in rows:
        yield {"id": example["id"], "query": example.get("query", ""), "prediction": dict(result)}


def compare_systems(
    examples: Sequence[Mapping[str, Any]],
    baseline_predictions: Sequence[Mapping[str, Any]],
    proposed_predictions: Sequence[Mapping[str, Any]],
    *,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate both systems under one explicit, shared experimental condition."""
    expected_ids = {str(example["id"]) for example in examples}
    baseline_ids = {str(row.get("id")) for row in baseline_predictions}
    proposed_ids = {str(row.get("id")) for row in proposed_predictions}
    if baseline_ids != expected_ids or proposed_ids != expected_ids:
        raise ValueError(
            "Baseline and proposed predictions must contain exactly the dataset example ids"
        )
    baseline = evaluate_system("baseline", examples, baseline_predictions, condition=condition)
    proposed = evaluate_system("proposed", examples, proposed_predictions, condition=condition)
    return {
        "schema_version": "1.0",
        "generated_at_epoch_seconds": time.time(),
        "dataset_size": len(examples),
        "condition": dict(condition),
        "systems": {"baseline": baseline, "proposed": proposed},
    }


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Prediction file must contain a JSON array or JSONL records")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation_dataset.csv")
    parser.add_argument("--baseline", required=True, help="Measured baseline JSON/JSONL predictions")
    parser.add_argument("--proposed", required=True, help="Measured proposed JSON/JSONL predictions")
    parser.add_argument("--output", default=str(RESULTS_DIR / "evaluation.json"))
    parser.add_argument("--condition", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    condition = dict(item.split("=", 1) for item in args.condition if "=" in item)
    report = compare_systems(
        load_dataset(args.dataset),
        load_predictions(args.baseline),
        load_predictions(args.proposed),
        condition=condition,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
