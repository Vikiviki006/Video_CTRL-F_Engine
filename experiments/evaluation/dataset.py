"""Dataset loading and ground-truth validation.

The single most important behaviour in this module: **a query with no
ground-truth annotation yields ``None`` metrics, not a zero.** Reporting 0.0
Recall for an unanswerable question is a fabricated number - it says "the
system retrieved nothing relevant" when the truth is "nobody said what
relevant was". :class:`Query` therefore carries an explicit
``has_ground_truth`` flag, and :func:`validate_dataset` reports exactly how many
queries are scorable so the report can print ``N/A`` for the rest.

Accepted CSV columns
--------------------
Required
    ``query``            the natural-language query
Ground truth (at least one of the following per row)
    ``video_id``         restrict the search to one video
    ``ground_truth_timestamp``  a single relevant timestamp in seconds
    ``ground_truth_timestamps``  several, ``|``-separated or a JSON array
    ``relevant_ids``     explicit frame UUIDs, ``|``-separated or JSON array
Optional
    ``id``               stable query identifier (defaults to row index)
    ``tolerance_seconds`` how far from the annotated timestamp still counts
                         as a hit (default 2.0)
    ``relevant_relationships``  ground-truth relationship names, for the
                         relationship-accuracy measurement
    ``notes``            free text, never used in scoring

Multi-value cells accept a JSON array (``["10.0","12.5"]``) or a
pipe-separated list (``10.0|12.5``); pipes are rejected when a value would be
ambiguous, so JSON is preferred.
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Default timestamp tolerance, in seconds. Chosen to be wider than the adaptive
# extractor's maximum forced-keep interval (MAX_FRAME_INTERVAL_SECONDS = 3.0) is
# NOT the intent: the tolerance must be narrower than the spacing between
# distinct shots or unrelated frames would count as hits. 2.0 s is a
# deliberately conservative default and is overridable per row.
DEFAULT_TOLERANCE_SECONDS = 2.0


@dataclass
class Query:
    """One annotated evaluation query."""

    id: str
    query: str
    video_id: Optional[uuid.UUID] = None
    relevant_timestamps: List[float] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS
    relevant_relationships: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def has_ground_truth(self) -> bool:
        """True when at least one relevant moment or frame is declared."""
        return bool(self.relevant_ids or self.relevant_timestamps)

    def to_run(self, ranked: Sequence[Any]) -> Dict[str, Any]:
        """Convert a ranked result list into the ``runs`` entry metrics expect."""
        return {
            "ranked": list(ranked),
            "relevant_ids": list(self.relevant_ids),
            "relevant_timestamps": list(self.relevant_timestamps),
            "tolerance_seconds": self.tolerance_seconds,
        }


@dataclass
class Dataset:
    """A loaded evaluation dataset plus its validation report."""

    queries: List[Query]
    path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def scorable(self) -> List[Query]:
        """Queries that have ground truth and can therefore be scored."""
        return [q for q in self.queries if q.has_ground_truth]

    @property
    def unscorable(self) -> List[Query]:
        """Queries with no ground truth. These must produce N/A, never 0.0."""
        return [q for q in self.queries if not q.has_ground_truth]

    @property
    def is_scorable(self) -> bool:
        return bool(self.scorable)

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "num_queries": len(self.queries),
            "num_scorable": len(self.scorable),
            "num_unscorable": len(self.unscorable),
            "warnings": list(self.warnings),
        }


class DatasetError(ValueError):
    """Raised when a dataset file cannot be used at all."""


def _split_multi(value: Any) -> List[str]:
    """Split a multi-value cell accepting either a JSON array or ``|``-separated."""
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON array in cell: {text!r} ({exc})")
        if not isinstance(parsed, list):
            raise DatasetError(f"expected a JSON array, got {text!r}")
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [part.strip() for part in text.split("|") if part.strip()]


def _optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_dataset(path: str | Path) -> Dataset:
    """Load and validate an evaluation dataset CSV.

    Raises
    ------
    DatasetError
        If the file is missing, has no header, has no ``query`` column, or
        contains no data rows. A dataset that cannot be parsed has no metrics
        at all, and that is a hard error rather than a warning.
    """
    source = Path(path)
    if not source.exists():
        raise DatasetError(
            f"Dataset not found: {source}. Copy "
            f"experiments/datasets/evaluation_dataset.template.csv and fill in "
            f"the ground-truth columns before running an evaluation."
        )

    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"Dataset has no header row: {source}")
        if "query" not in reader.fieldnames:
            raise DatasetError(
                f"Dataset is missing the required 'query' column. "
                f"Found: {', '.join(reader.fieldnames)}"
            )
        rows = list(reader)

    if not rows:
        raise DatasetError(f"Dataset contains a header but no data rows: {source}")

    warnings: List[str] = []
    queries: List[Query] = []
    seen_ids: Dict[str, int] = {}

    for index, row in enumerate(rows):
        query_text = (row.get("query") or "").strip()
        if not query_text:
            warnings.append(f"row {index + 2}: empty 'query', skipped")
            continue

        query_id = (row.get("id") or f"q{index + 1}").strip()
        if query_id in seen_ids:
            warnings.append(
                f"row {index + 2}: duplicate id '{query_id}' "
                f"(first seen on row {seen_ids[query_id] + 2}); disambiguated"
            )
            query_id = f"{query_id}#{index + 1}"
        seen_ids[query_id] = index

        video_id: Optional[uuid.UUID] = None
        raw_video = (row.get("video_id") or "").strip()
        if raw_video:
            try:
                video_id = uuid.UUID(raw_video)
            except ValueError:
                warnings.append(
                    f"query '{query_id}': video_id '{raw_video}' is not a valid "
                    f"UUID; the query will be evaluated across all videos"
                )

        timestamps: List[float] = []
        for key in ("ground_truth_timestamps", "ground_truth_timestamp"):
            raw = row.get(key)
            if raw in (None, ""):
                continue
            for token in _split_multi(raw):
                try:
                    timestamps.append(float(token))
                except ValueError:
                    warnings.append(
                        f"query '{query_id}': '{token}' in {key} is not a number; skipped"
                    )
        if any(t < 0 for t in timestamps):
            warnings.append(
                f"query '{query_id}': negative timestamp(s) ignored "
                f"(timestamps are seconds from the start of the video)"
            )
            timestamps = [t for t in timestamps if t >= 0]

        relevant_ids = _split_multi(row.get("relevant_ids"))
        tolerance_raw = (row.get("tolerance_seconds") or "").strip()
        try:
            tolerance = (
                float(tolerance_raw) if tolerance_raw else DEFAULT_TOLERANCE_SECONDS
            )
        except ValueError:
            warnings.append(
                f"query '{query_id}': tolerance_seconds '{tolerance_raw}' is not a "
                f"number; using default {DEFAULT_TOLERANCE_SECONDS}"
            )
            tolerance = DEFAULT_TOLERANCE_SECONDS
        if tolerance <= 0:
            warnings.append(
                f"query '{query_id}': tolerance_seconds must be > 0; using default "
                f"{DEFAULT_TOLERANCE_SECONDS}"
            )
            tolerance = DEFAULT_TOLERANCE_SECONDS

        queries.append(
            Query(
                id=query_id,
                query=query_text,
                video_id=video_id,
                relevant_timestamps=timestamps,
                relevant_ids=relevant_ids,
                tolerance_seconds=tolerance,
                relevant_relationships=_split_multi(
                    row.get("relevant_relationships")
                ),
                notes=(row.get("notes") or "").strip(),
            )
        )

    if not queries:
        raise DatasetError(f"No usable rows in {source}")

    dataset = Dataset(queries=queries, path=source, warnings=warnings)
    if not dataset.is_scorable:
        dataset.warnings.append(
            f"{len(dataset.queries)} query/queries loaded but NONE have ground "
            f"truth (no video_id + ground_truth_timestamp, no ground_truth_timestamps, "
            f"no relevant_ids). Retrieval metrics are undefined and will be reported "
            f"as N/A. This is a missing-annotation problem, not a zero score."
        )
    return dataset


def validate_dataset(dataset: Dataset) -> Dict[str, Any]:
    """Produce the report block describing what can and cannot be scored."""
    report = dataset.summary()
    report["scorable_query_ids"] = [q.id for q in dataset.scorable]
    report["unscorable_query_ids"] = [q.id for q in dataset.unscorable]
    report["metrics_status"] = (
        "measured" if dataset.is_scorable else "N/A - no ground truth available"
    )
    return report
