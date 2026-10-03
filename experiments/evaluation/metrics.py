"""Information-retrieval metrics for VideoCtrl-F, with exact definitions.

Every function here is a pure function of *already-ranked* results and
*already-declared* ground truth. Nothing in this module runs a model, touches a
database, or invents a relevance judgement. Given a ranking and a ground-truth
annotation, the answer is deterministic and hand-checkable — which is why
``backend/tests/test_metrics.py`` verifies each one against a value computed by
hand.

Conventions
-----------
A *run* for one query is a list of retrieved items, already ordered best-first.
Each item is a dict that may carry:

``id`` or ``frame_id``
    Frame identity. Matched against ground-truth ids when the dataset provides
    them.
``timestamp_seconds``
    Frame position. Matched against ground-truth timestamps within a tolerance
    when the dataset provides them instead of ids.

Relevance
~~~~~~~~~
**Binary relevance is used throughout.** A retrieved item is relevant if it
matches any ground-truth id, or falls within ``tolerance_seconds`` of any
ground-truth timestamp. Graded relevance is *not* used, because the datasets
this project can realistically build carry no relevance grades; inventing a
grade would fabricate information. If a future dataset supplies real grades,
:func:`ndcg_at_k` accepts them directly (see its docstring) and the other
metrics ignore them, which is the correct behaviour for binary metrics.

Formulas
--------

.. code-block:: text

    Precision@K(q)  = |{i in top-K : relevant(i)}| / K
                      (the denominator is K, not |top-K|, so a run that returns
                       fewer than K results is correctly penalised)

    Recall@K(q)     = |{i in top-K : relevant(i)}| / |relevant(q)|
                      undefined when the query has no relevant items -> 0.0

    HitRate@K(q)    = 1.0 if any item in top-K is relevant else 0.0

    MRR(q)          = 1 / rank_of_first_relevant_result          (0 if none)

    AP(q)           = (1 / R) * sum_{i in 1..ranked} P@i * relevant(i)
                      R = total number of relevant items for q
                      sum runs over the FULL ranked list, not just top-K

    MAP             = mean over queries of AP(q)

    DCG@K           = sum_{i=1..K} rel_i / log2(i + 1)
    IDCG@K          = sum_{i=1..min(K, R)} 1 / log2(i + 1)   (ideal ordering)
    NDCG@K          = DCG@K / IDCG@K, or 0.0 when IDCG@K == 0

The log2 discount is the standard one; using ``log2(i+1)`` rather than
``log2(i)`` avoids division by zero at the first position.

Latency
~~~~~~~
Percentiles use **linear interpolation between closest ranks** (the
"inclusive"/NumPy default method) on the sorted sample. With ``n`` samples and
``p`` in ``[0, 100]`` the target position is ``(n - 1) * p / 100``, floored and
ceiled, and the value interpolated. This is stated explicitly because percentile
conventions differ between libraries and silently mixing them makes P95
numbers incomparable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Default cut-offs required by the evaluation specification.
DEFAULT_KS: Tuple[int, ...] = (1, 5, 10, 20)
DEFAULT_HITRATE_KS: Tuple[int, ...] = (1, 5, 10)

# Guards against a degenerate dataset silently producing meaningless numbers.
MIN_QUERIES_FOR_STATS = 1


# ---------------------------------------------------------------------------
# Relevance judgement
# ---------------------------------------------------------------------------


def _item_id(item: Any) -> Optional[str]:
    if isinstance(item, Mapping):
        for key in ("id", "frame_id", "document_id"):
            value = item.get(key)
            if value is not None:
                return str(value)
        return None
    return str(item)


def _item_timestamp(item: Any) -> Optional[float]:
    if isinstance(item, Mapping):
        value = item.get("timestamp_seconds")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def is_relevant(
    item: Any,
    relevant_ids: Sequence[str],
    relevant_timestamps: Sequence[float],
    tolerance_seconds: float,
) -> bool:
    """Binary relevance test for one retrieved item.

    Matching rule, stated once and used everywhere:

    * If the dataset supplies ground-truth **ids**, an item matches on id.
    * If the dataset supplies ground-truth **timestamps**, an item matches when
      ``|item.timestamp - gt| <= tolerance_seconds``. This is the rule that
      makes a single-ground-truth-timestamp dataset well defined: a query
      annotated with one timestamp is satisfied by any retrieved frame within
      the tolerance of it.
    * An item with no id and no timestamp is never relevant.
    """
    item_id = _item_id(item)
    if item_id is not None and relevant_ids:
        if item_id in set(str(x) for x in relevant_ids):
            return True

    ts = _item_timestamp(item)
    if ts is not None and relevant_timestamps:
        for gt in relevant_timestamps:
            if abs(ts - float(gt)) <= tolerance_seconds:
                return True
    return False


def relevance_vector(
    ranked: Sequence[Any],
    relevant_ids: Sequence[str],
    relevant_timestamps: Sequence[float],
    tolerance_seconds: float,
    grades: Optional[Mapping[str, float]] = None,
) -> List[bool]:
    """Binary relevance for every item of a ranked list, in order."""
    return [
        is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds)
        for item in ranked
    ]


def _relevant_count(relevant_ids: Sequence[str], relevant_timestamps: Sequence[float]) -> int:
    return len(relevant_ids) or len(relevant_timestamps)


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def precision_at_k(
    ranked: Sequence[Any],
    k: int,
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
) -> float:
    """``Precision@K = |relevant in top-K| / K``.

    The denominator is ``K`` and not ``min(K, len(ranked))``, so a system that
    returns fewer than K results is penalised for the missing ones. A query with
    no relevant item scores 0.0 at every K.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    hits = sum(
        1
        for item in list(ranked)[:k]
        if is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds)
    )
    return hits / float(k)


def recall_at_k(
    ranked: Sequence[Any],
    k: int,
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
) -> float:
    """``Recall@K = |relevant in top-K| / |relevant|``.

    Returns 0.0 when the query has no ground-truth relevant item, because the
    ratio is undefined and silently returning 1.0 would reward a system for
    being asked an unanswerable question.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    total = _relevant_count(relevant_ids, relevant_timestamps)
    if total == 0:
        return 0.0
    hits = sum(
        1
        for item in list(ranked)[:k]
        if is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds)
    )
    return hits / float(total)


def hit_rate_at_k(
    ranked: Sequence[Any],
    k: int,
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
) -> float:
    """``HitRate@K = 1.0 if any item in top-K is relevant else 0.0``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for item in list(ranked)[:k]:
        if is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds):
            return 1.0
    return 0.0


def reciprocal_rank(
    ranked: Sequence[Any],
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
) -> float:
    """``1 / rank_of_first_relevant_result``; 0.0 when no result is relevant.

    Rank is 1-based, so a correct first hit gives 1.0.
    """
    for index, item in enumerate(ranked, start=1):
        if is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds):
            return 1.0 / index
    return 0.0


def average_precision(
    ranked: Sequence[Any],
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
) -> float:
    """``AP = (1/R) * sum_i P@i * relevant(i)`` over the **full** ranked list.

    Supports any number of relevant items, which is what makes MAP meaningful
    on datasets where several moments of a video answer a query. ``R`` is the
    total number of ground-truth relevant items, not the number returned.
    """
    total = _relevant_count(relevant_ids, relevant_timestamps)
    if total == 0:
        return 0.0

    hits = 0
    score = 0.0
    for index, item in enumerate(ranked, start=1):
        if is_relevant(item, relevant_ids, relevant_timestamps, tolerance_seconds):
            hits += 1
            score += (hits / float(index))
    return score / float(total)


def dcg_at_k(
    ranked: Sequence[Any],
    k: int,
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
    grades: Optional[Mapping[str, float]] = None,
) -> float:
    """``DCG@K = sum_{i=1..K} gain_i / log2(i + 1)``.

    With ``grades=None`` (the default, and what every dataset in this project
    uses) the gain is binary: 1 for relevant, 0 otherwise. With a ``grades``
    mapping keyed by item id the gain becomes that item's grade, which is the
    only case in which this project would use graded relevance.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    total = 0.0
    for index, item in enumerate(list(ranked)[:k], start=1):
        if grades is not None:
            item_id = _item_id(item)
            gain = float(grades.get(item_id, 0.0)) if item_id else 0.0
        else:
            gain = 1.0 if is_relevant(
                item, relevant_ids, relevant_timestamps, tolerance_seconds
            ) else 0.0
        total += gain / math.log2(index + 1)
    return total


def ndcg_at_k(
    ranked: Sequence[Any],
    k: int,
    relevant_ids: Sequence[str] = (),
    relevant_timestamps: Sequence[float] = (),
    tolerance_seconds: float = 2.0,
    grades: Optional[Mapping[str, float]] = None,
) -> float:
    """``NDCG@K = DCG@K / IDCG@K``.

    ``IDCG@K`` assumes the ideal ranking puts ``min(K, R)`` relevant items
    first, where ``R`` is the total number of relevant items. Returns 0.0 when
    ``IDCG@K == 0`` (nothing relevant exists, or ``K`` is 0).

    Binary relevance unless a ``grades`` mapping is supplied; see the module
    docstring.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    dcg = dcg_at_k(ranked, k, relevant_ids, relevant_timestamps, tolerance_seconds, grades)
    if grades is not None:
        r = len(grades)
    else:
        r = _relevant_count(relevant_ids, relevant_timestamps)
    ideal_positions = min(k, r)
    if ideal_positions <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_positions + 1))
    return dcg / idcg


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------


def evaluate_run(
    runs: Sequence[Mapping[str, Any]],
    ks: Iterable[int] = DEFAULT_KS,
    hitrate_ks: Iterable[int] = DEFAULT_HITRATE_KS,
) -> Dict[str, float]:
    """Aggregate per-query metrics into corpus-level means.

    Parameters
    ----------
    runs:
        One entry per query: ``{"ranked": [...], "relevant_ids": [...],
        "relevant_timestamps": [...], "tolerance_seconds": 2.0}``.
    ks:
        Cut-offs for Precision@K / Recall@K / NDCG@K.
    hitrate_ks:
        Cut-offs for HitRate@K.

    Returns
    -------
    dict
        ``{"precision@K": v, "recall@K": v, "ndcg@K": v, "hitrate@K": v,
        "mrr": v, "map": v, "num_queries": n, "num_relevant_items": r}``.

    Raises
    ------
    ValueError
        When ``runs`` is empty. Returning an empty dict or a set of zeros for
        an empty run would be indistinguishable from "scored and got 0", which
        is exactly the kind of ambiguity that turns into a fabricated number
        later on. Callers that genuinely have nothing to score must handle it
        explicitly.
    """
    runs = list(runs)
    if not runs:
        raise ValueError(
            "evaluate_run() requires at least one query run. An empty run has "
            "no defined metrics; report N/A instead of zero."
        )

    ks = list(ks)
    hitrate_ks = list(hitrate_ks)

    out: Dict[str, float] = {"num_queries": float(len(runs))}
    total_relevant = 0

    for k in ks:
        precisions, recalls, ndcgs = [], [], []
        for run in runs:
            ranked = run.get("ranked", [])
            ids = run.get("relevant_ids", []) or []
            ts = run.get("relevant_timestamps", []) or []
            tol = float(run.get("tolerance_seconds", 2.0))
            precisions.append(precision_at_k(ranked, k, ids, ts, tol))
            recalls.append(recall_at_k(ranked, k, ids, ts, tol))
            ndcgs.append(ndcg_at_k(ranked, k, ids, ts, tol))
        out[f"precision@{k}"] = _mean(precisions)
        out[f"recall@{k}"] = _mean(recalls)
        out[f"ndcg@{k}"] = _mean(ndcgs)

    for k in hitrate_ks:
        hits = [
            hit_rate_at_k(
                run.get("ranked", []),
                k,
                run.get("relevant_ids", []) or [],
                run.get("relevant_timestamps", []) or [],
                float(run.get("tolerance_seconds", 2.0)),
            )
            for run in runs
        ]
        out[f"hitrate@{k}"] = _mean(hits)

    rr = [
        reciprocal_rank(
            run.get("ranked", []),
            run.get("relevant_ids", []) or [],
            run.get("relevant_timestamps", []) or [],
            float(run.get("tolerance_seconds", 2.0)),
        )
        for run in runs
    ]
    ap = [
        average_precision(
            run.get("ranked", []),
            run.get("relevant_ids", []) or [],
            run.get("relevant_timestamps", []) or [],
            float(run.get("tolerance_seconds", 2.0)),
        )
        for run in runs
    ]
    for run in runs:
        total_relevant += _relevant_count(
            run.get("relevant_ids", []) or [], run.get("relevant_timestamps", []) or []
        )

    out["mrr"] = _mean(rr)
    out["map"] = _mean(ap)
    out["num_relevant_items"] = float(total_relevant)
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, ``p`` in ``[0, 100]``.

    Position is ``(n - 1) * p / 100``. Explicitly documented because the
    nearest-rank convention used by some tools gives visibly different P95
    values on small samples, and mixing the two makes latency numbers
    incomparable across tables.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (p / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return data[int(pos)]
    return data[low] + (data[high] - data[low]) * (pos - low)


def latency_stats(values: Sequence[float]) -> Dict[str, float]:
    """avg / median / p95 / p99 / min / max / count for one sample.

    Returns ``count=0`` and all other keys ``None`` for an empty sample, which
    renders as ``N/A`` in the reports instead of a misleading 0 ms.
    """
    data = [float(v) for v in values]
    if not data:
        return {
            "count": 0,
            "avg_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(data)
    n = len(ordered)
    return {
        "count": n,
        "avg_ms": sum(ordered) / n,
        "median_ms": (
            ordered[n // 2]
            if n % 2 == 1
            else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        ),
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


# ---------------------------------------------------------------------------
# Indexing / storage
# ---------------------------------------------------------------------------


def frame_reduction_percent(original_frames: int, retained_frames: int) -> Optional[float]:
    """``Frame Reduction % = (1 - retained / original) * 100``.

    Returns ``None`` when ``original_frames <= 0``, because the ratio is
    undefined; ``None`` renders as ``N/A`` rather than a fabricated 0 %.
    """
    if original_frames <= 0:
        return None
    return (1.0 - (retained_frames / float(original_frames))) * 100.0


def storage_overhead_ratio(
    raw_bytes: int, derived_bytes: int
) -> Optional[float]:
    """``derived / raw`` - how many bytes the index adds per byte of video.

    ``None`` when ``raw_bytes <= 0``.
    """
    if raw_bytes <= 0:
        return None
    return derived_bytes / float(raw_bytes)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Convenience wrapper returning only avg/median/p95/p99."""
    stats = latency_stats(values)
    return {
        "avg": stats["avg_ms"],
        "median": stats["median_ms"],
        "p95": stats["p95_ms"],
        "p99": stats["p99_ms"],
    }
