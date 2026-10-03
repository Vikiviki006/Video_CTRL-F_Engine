"""Graph-specific measurements for the evaluation study.

These answer questions the plain IR metrics cannot:

* How many nodes did a bounded expansion actually touch?
* How many edges did it traverse, and of which types?
* Did the traversal hit ``max_depth`` or ``max_nodes``? (i.e. was the bound
  load-bearing, or was the graph simply small?)
* How does expanded-node count and latency vary with depth?

Every value here is measured during a real run. Nothing is estimated, and
:func:`empty_graph_report` exists precisely so an unrun configuration is
reported as ``N/A`` instead of being quietly filled with zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from experiments.evaluation.metrics import latency_stats


@dataclass
class ExpansionObservation:
    """What one real bounded expansion did.

    Recorded by the retrieval harness for every query, at every depth.
    """

    query_id: str
    depth_requested: int
    nodes_reached: int
    edges_traversed: int
    seed_count: int
    max_depth_reached: int
    truncated_depth: bool
    truncated_nodes: bool
    expansion_latency_ms: float
    edge_type_counts: Dict[str, int] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "depth_requested": self.depth_requested,
            "nodes_reached": self.nodes_reached,
            "edges_traversed": self.edges_traversed,
            "seed_count": self.seed_count,
            "max_depth_reached": self.max_depth_reached,
            "truncated_depth": self.truncated_depth,
            "truncated_nodes": self.truncated_nodes,
            "expansion_latency_ms": self.expansion_latency_ms,
            **{
                f"edges_{k.lower()}": v
                for k, v in sorted(self.edge_type_counts.items())
            },
        }


def empty_graph_report(reason: str) -> Dict[str, Any]:
    """A graph report that explicitly says nothing was measured.

    Returned instead of a dict of zeros when there is no graph data, so the
    downstream CSV contains the string ``N/A`` and a reader cannot mistake an
    unrun experiment for a measured result.
    """
    return {
        "measured": False,
        "status": "N/A",
        "reason": reason,
        "nodes_reached_avg": None,
        "edges_traversed_avg": None,
        "expansion_latency_ms_avg": None,
        "expansion_latency_ms_p95": None,
        "depths_measured": [],
    }


def aggregate_expansions(observations: Sequence[ExpansionObservation]) -> Dict[str, Any]:
    """Aggregate expansions into one row per requested depth.

    Returns an ``N/A`` report when there are no observations.
    """
    if not observations:
        return empty_graph_report("no bounded graph expansion was executed")

    by_depth: Dict[int, List[ExpansionObservation]] = {}
    for obs in observations:
        by_depth.setdefault(obs.depth_requested, []).append(obs)

    depths: List[Dict[str, Any]] = []
    for depth in sorted(by_depth):
        group = by_depth[depth]
        latencies = [g.expansion_latency_ms for g in group]
        stats = latency_stats(latencies)
        edge_totals: Dict[str, int] = {}
        for g in group:
            for key, value in g.edge_type_counts.items():
                edge_totals[key] = edge_totals.get(key, 0) + value
        depths.append(
            {
                "depth": depth,
                "queries": len(group),
                "nodes_reached_avg": sum(g.nodes_reached for g in group) / len(group),
                "nodes_reached_max": max(g.nodes_reached for g in group),
                "edges_traversed_avg": sum(g.edges_traversed for g in group)
                / len(group),
                "edges_traversed_max": max(g.edges_traversed for g in group),
                "expansion_latency_ms_avg": stats["avg_ms"],
                "expansion_latency_ms_p95": stats["p95_ms"],
                "depth_limit_hit_queries": sum(1 for g in group if g.truncated_depth),
                "node_limit_hit_queries": sum(1 for g in group if g.truncated_nodes),
                "edge_type_totals": dict(sorted(edge_totals.items())),
            }
        )

    all_latencies = [o.expansion_latency_ms for o in observations]
    all_stats = latency_stats(all_latencies)

    return {
        "measured": True,
        "status": "measured",
        "num_observations": len(observations),
        "depths_measured": [d["depth"] for d in depths],
        "by_depth": depths,
        "nodes_reached_avg": sum(o.nodes_reached for o in observations)
        / len(observations),
        "edges_traversed_avg": sum(o.edges_traversed for o in observations)
        / len(observations),
        "expansion_latency_ms_avg": all_stats["avg_ms"],
        "expansion_latency_ms_p95": all_stats["p95_ms"],
    }


def depth_effect_report(
    per_depth: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarise how quality and cost move as depth increases.

    The summary deliberately makes **no claim** that deeper is better. It
    reports the measured direction of each metric with depth and leaves the
    interpretation to the reader, because whether depth 2 beats depth 1 is an
    empirical question this project has not pre-decided.

    ``per_depth`` entries need ``depth`` plus at least one of the quality keys
    (``recall@10``, ``ndcg@10``, ``mrr``, ``map``) and one of the cost keys
    (``nodes_reached_avg``, ``expansion_latency_ms_avg``).
    """
    if not per_depth:
        return empty_graph_report("no depth sweep was executed")

    ordered = sorted(per_depth, key=lambda r: r["depth"])
    report: Dict[str, Any] = {
        "measured": True,
        "status": "measured",
        "depths": [r["depth"] for r in ordered],
        "rows": list(ordered),
    }

    quality_keys = [
        k
        for k in ("recall@10", "ndcg@10", "mrr", "map", "precision@10")
        if any(k in r for r in ordered)
    ]
    for key in quality_keys:
        series = [(r["depth"], r.get(key)) for r in ordered if r.get(key) is not None]
        report[f"{key}_by_depth"] = dict(series)
        if len(series) >= 2:
            first, last = series[0], series[-1]
            delta = last[1] - first[1]
            report[f"{key}_delta_shallow_to_deepest"] = delta
            # Stated as an observation, not a verdict.
            report[f"{key}_direction"] = (
                "increased" if delta > 0 else ("decreased" if delta < 0 else "flat")
            )

    for key in ("nodes_reached_avg", "expansion_latency_ms_avg"):
        series = [(r["depth"], r.get(key)) for r in ordered if r.get(key) is not None]
        if series:
            report[f"{key}_by_depth"] = dict(series)

    # Identify the depth that maximised each quality metric, when measurable.
    for key in quality_keys:
        scored = [
            (r["depth"], r.get(key)) for r in ordered if r.get(key) is not None
        ]
        if scored:
            best_depth, best_value = max(scored, key=lambda kv: kv[1])
            report[f"best_{key}_depth"] = best_depth
            report[f"best_{key}_value"] = best_value
    return report


def graph_size_report(db: Any, video_id: Any) -> Dict[str, Any]:
    """Count nodes/edges for one video, grouped by type.

    Used to report the storage cost of the graph layer as a real measured
    number rather than an estimate.
    """
    from sqlalchemy import func, select

    from app.models.graph import GraphEdge, GraphNode

    node_rows = (
        db.execute(
            select(GraphNode.node_type, func.count())
            .where(GraphNode.video_id == video_id)
            .group_by(GraphNode.node_type)
        )
        .all()
    )
    edge_rows = (
        db.execute(
            select(GraphEdge.relationship, func.count())
            .join(GraphNode, GraphNode.id == GraphEdge.source_node_id)
            .where(GraphNode.video_id == video_id)
            .group_by(GraphEdge.relationship)
        )
        .all()
    )

    node_counts = {str(t): int(c) for t, c in node_rows}
    edge_counts = {str(r): int(c) for r, c in edge_rows}
    return {
        "video_id": str(video_id),
        "node_count": sum(node_counts.values()),
        "edge_count": sum(edge_counts.values()),
        "node_type_counts": dict(sorted(node_counts.items())),
        "relationship_counts": dict(sorted(edge_counts.items())),
    }
