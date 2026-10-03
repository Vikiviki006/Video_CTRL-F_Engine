"""Ranking configurations for the evaluation study.

The ablation is expressed purely as a set of :class:`RankingConfig` values
re-exported from ``app.services.graph_ranking_service``, so the experiment and
production share one implementation of the scoring formula. There is no second
copy of the ranking logic here, which is what stops the published ablation
numbers from describing code the system does not actually run.

Conditions
==========
=====  ==============================  =======  ======  =======  ========
Key    Description                     vector   graph   temporal  confidence
=====  ==============================  =======  ======  =======  ========
``A``  vector only (the baseline)     1.00     0.00    0.00      0.00
``B``  vector + temporal              0.65     0.00    0.35      0.00
``C``  vector + graph                 0.80     0.20    0.00      0.00
``D``  vector + graph + temporal      0.65     0.20    0.15      0.00
=====  ==============================  =======  ======  =======  ========

Within each condition the active weights are renormalised to sum to 1.0, so
the comparison is about *which signals participate*, not about their raw
magnitudes.

Condition ``D`` leaves ``confidence_weight`` at 0 on purpose. With no extractor
registered, every node confidence is exactly 1.0, so the term is
ranking-neutral; including it would add a constant 0.05 to every candidate's
final score and make the numbers look like the confidence signal is doing work
when it demonstrably is not. It is exercised separately as a measured fact in
:func:`confidence_term_diagnostic`.

Graph-depth sweep
=================
Depths 0/1/2/3. Depth 0 disables expansion entirely, so it collapses to the
vector-only ordering and is the natural control row.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.services.graph_ranking_service import (
    RankingConfig,
    ablation_configs,
    rank_candidates,
)
from experiments.evaluation.metrics import evaluate_run, latency_stats

#: Depth values swept by the graph-depth ablation.
DEPTH_SWEEP: tuple = (0, 1, 2, 3)


def get_configs() -> Dict[str, RankingConfig]:
    """The A/B/C/D ablation configurations."""
    return ablation_configs()


def config_rows() -> List[Dict[str, Any]]:
    """One flat row per condition, for writing straight to CSV."""
    rows: List[Dict[str, Any]] = []
    for key, cfg in sorted(get_configs().items()):
        rows.append(
            {
                "condition": key,
                "description": cfg.name,
                "vector_weight": cfg.vector_weight,
                "graph_weight": cfg.graph_weight,
                "temporal_weight": cfg.temporal_weight,
                "confidence_weight": cfg.confidence_weight,
            }
        )
    return rows


def confidence_term_diagnostic(
    per_candidate_confidence: Sequence[float],
) -> Dict[str, Any]:
    """Report whether the ConfidenceScore term can affect ranking.

    Returns ``{"ranking_neutral": bool, ...}``. When every candidate's
    confidence score is identical the term is a constant, so it cannot change
    any ordering, and saying so is more useful than quietly reporting a number
    that looks meaningful but is not.
    """
    values = [float(v) for v in per_candidate_confidence]
    if not values:
        return {
            "measured": False,
            "status": "N/A",
            "reason": "no candidates were scored",
        }
    distinct = sorted(set(round(v, 12) for v in values))
    return {
        "measured": True,
        "status": "measured",
        "num_candidates": len(values),
        "distinct_values": len(distinct),
        "ranking_neutral": len(distinct) <= 1,
        "min": min(values),
        "max": max(values),
        "note": (
            "All candidates share the same confidence score, so this term is a "
            "constant and cannot change the ranking. This is the expected state "
            "while no concept/object extractor is registered."
            if len(distinct) <= 1
            else "Confidence scores differ across candidates, so the term can "
            "affect ranking."
        ),
    }


def run_condition(
    condition_key: str,
    per_query_ranked: Sequence[Sequence[Any]],
    per_query_latency_ms: Sequence[float],
    *,
    dataset,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one ablation condition end to end.

    Parameters
    ----------
    condition_key:
        ``"A"``, ``"B"``, ``"C"`` or ``"D"``.
    per_query_ranked:
        For each scorable query, its already-ranked result list, in the order
        that condition produced.
    per_query_latency_ms:
        Measured end-to-end latency for the same queries, in the same order.
    dataset:
        The loaded :class:`~experiments.evaluation.dataset.Dataset`, used to
        attach ground truth and to decide whether metrics are scorable at all.

    Returns
    -------
    dict
        Either real metrics, or an explicit ``N/A`` block naming the reason.
    """
    configs = get_configs()
    if condition_key not in configs:
        raise KeyError(
            f"unknown condition '{condition_key}'. Available: {', '.join(sorted(configs))}"
        )
    cfg = configs[condition_key]

    row: Dict[str, Any] = {
        "condition": condition_key,
        "description": cfg.name,
        "vector_weight": cfg.vector_weight,
        "graph_weight": cfg.graph_weight,
        "temporal_weight": cfg.temporal_weight,
        "confidence_weight": cfg.confidence_weight,
    }
    if extra:
        row.update(extra)

    if len(per_query_ranked) != len(per_query_latency_ms):
        raise ValueError(
            f"condition {condition_key}: got {len(per_query_ranked)} ranked lists "
            f"but {len(per_query_latency_ms)} latency samples; they must correspond"
        )

    if not dataset.is_scorable:
        row["metrics_status"] = "N/A"
        row["metrics_reason"] = (
            "no ground truth in dataset: recall/precision/mrr/map/ndcg are undefined"
        )
    elif not per_query_ranked:
        row["metrics_status"] = "N/A"
        row["metrics_reason"] = "no queries were scored for this condition"
    else:
        runs = [
            query.to_run(ranked)
            for query, ranked in zip(dataset.scorable, per_query_ranked)
        ]
        row["metrics_status"] = "measured"
        row.update(evaluate_run(runs))

    latency = latency_stats(per_query_latency_ms)
    row["latency_count"] = latency["count"]
    row["latency_avg_ms"] = latency["avg_ms"]
    row["latency_median_ms"] = latency["median_ms"]
    row["latency_p95_ms"] = latency["p95_ms"]
    row["latency_p99_ms"] = latency["p99_ms"]
    return row
