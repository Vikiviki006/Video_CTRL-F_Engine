"""CLI entrypoint for the VideoCtrl-F retrieval evaluation.

Runs a real retrieval experiment against a real database and a real
embedding model, then writes CSV, JSON and a generated Markdown summary.

The specification's invocation form works verbatim::

    python -m experiments.evaluation.evaluate_retrieval \\
        --dataset experiments/datasets/evaluation.csv \\
        --model siglip2 \\
        --top-k 10 \\
        --graph --graph-depth 2 \\
        --output experiments/results/

What this script will not do
---------------------------
* It will not score a query that has no ground-truth annotation. Such queries
  are counted and reported, and their metrics are ``N/A``.
* It will not invent a SigLIP-vs-SigLIP-2 row. The comparison table is only
  populated when a second checkpoint's predictions are supplied via
  ``--predictions`` or a second run's results are merged; otherwise every
  comparison cell is ``N/A`` with the reason recorded.
* It will not write a number it did not measure. Latency comes from
  ``time.perf_counter`` around real stages; graph sizes come from real
  ``COUNT(*)`` queries.

Modes
-----
``--dry-run``
    Loads and validates the dataset, reports what is and is not scorable, and
    exits without touching the database or the model. Useful for confirming
    annotation coverage before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from experiments.evaluation import reporting  # noqa: E402
from experiments.evaluation.dataset import (  # noqa: E402
    Dataset,
    DatasetError,
    Query,
    load_dataset,
    validate_dataset,
)
from experiments.evaluation.graph_metrics import (  # noqa: E402
    ExpansionObservation,
    aggregate_expansions,
    empty_graph_report,
    graph_size_report,
)
from experiments.evaluation.metrics import (  # noqa: E402
    DEFAULT_HITRATE_KS,
    DEFAULT_KS,
    evaluate_run,
    frame_reduction_percent,
    latency_stats,
    storage_overhead_ratio,
)
from experiments.evaluation.ranking import (  # noqa: E402
    DEPTH_SWEEP,
    confidence_term_diagnostic,
    run_condition,
)

# Known checkpoints. Only the ones actually indexed can be reported on.
KNOWN_MODELS = {
    "siglip2": "google/siglip2-so400m-patch14-384",
    "siglip": "google/siglip-so400m-patch14-384",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.evaluation.evaluate_retrieval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="experiments/datasets/evaluation.csv",
        help="Annotated evaluation CSV (see experiments/datasets/).",
    )
    parser.add_argument(
        "--model",
        default="siglip2",
        choices=sorted(KNOWN_MODELS),
        help="Embedding checkpoint to evaluate. Must match what was indexed.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Results per query.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=-1.0,
        help=(
            "Candidate similarity floor, matching the API default (no floor). "
            "SigLIP-2 cosines cluster near 0, so a 0.0 floor empties the pool "
            "for some queries; pass 0.0 to measure that older behaviour."
        ),
    )
    parser.add_argument("--output", default="experiments/results/")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Enable graph-aware retrieval (condition D) in addition to the baseline.",
    )
    parser.add_argument(
        "--graph-depth",
        type=int,
        default=2,
        help="Graph expansion depth for the default graph run.",
    )
    parser.add_argument(
        "--depth-sweep",
        action="store_true",
        help=f"Run the graph-depth ablation over depths {list(DEPTH_SWEEP)}.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all four ablation conditions A/B/C/D.",
    )
    parser.add_argument(
        "--index-metrics",
        action="store_true",
        help="Collect indexing, frame-reduction and storage metrics from the database.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional JSON/JSONL of previously measured predictions "
            "({'query_id': ..., 'ranked': [...]}) to score instead of querying the "
            "live database. Use this to compare two checkpoints without re-running "
            "the model."
        ),
    )
    parser.add_argument(
        "--predictions-other-model",
        default=None,
        help="Second model's prediction file, for the SigLIP vs SigLIP-2 table.",
    )
    parser.add_argument("--other-model", default="siglip", choices=sorted(KNOWN_MODELS))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and exit. Touches no database and no model.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_predictions(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load ``{query_id: ranked_items}`` from JSON or JSONL."""
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"predictions file not found: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if not isinstance(records, list):
        raise DatasetError(f"{source}: expected a JSON array or JSONL records")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        qid = str(record.get("query_id") or record.get("id") or "")
        ranked = record.get("ranked") or record.get("retrieved") or []
        if qid:
            out[qid] = list(ranked)
    return out


def _na_row(reason: str, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"metrics_status": "N/A", "metrics_reason": reason}
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Live retrieval
# ---------------------------------------------------------------------------


def _live_condition(
    db: Any,
    dataset: Dataset,
    *,
    top_k: int,
    graph_depth: int,
    use_graph: bool,
    config: Optional[RankingConfig] = None,
    min_score: float = -1.0,
) -> Dict[str, Any]:
    """Run one retrieval configuration against the live database.

    ``config`` is the weight set handed to ``graph_search``: conditions A/B/C/D
    differ *only* through it, so the published ablation measures the weights the
    system is told to use rather than the weights in ``Settings``.

    Returns per-query ranked lists, per-query and stage latencies, and the
    graph expansion observations. Every latency value comes from a real
    ``time.perf_counter`` measurement around the real stage.
    """
    from app.core.config import Settings
    from app.schemas.graph import GraphSearchRequest
    from app.services.graph_ranking_service import (
        RankingConfig,
        ablation_configs,
    )
    from app.services.graph_search_service import graph_search

    per_query_ranked: List[List[Dict[str, Any]]] = []
    per_query_latency: List[float] = []
    stage_latencies: Dict[str, List[float]] = {
        "embedding": [],
        "vector_retrieval": [],
        "graph_expansion": [],
        "graph_reranking": [],
        "total": [],
    }
    expansions: List[ExpansionObservation] = []
    graph_scores: List[float] = []
    confidence_scores: List[float] = []

    depth = graph_depth if use_graph else 0

    for query in dataset.queries:
        request = GraphSearchRequest(
            query=query.query,
            video_id=query.video_id,
            top_k=top_k,
            graph_depth=depth,
            min_score=min_score,
        )
        t0 = time.perf_counter()
        response = graph_search(db, request, config=config)
        elapsed = (time.perf_counter() - t0) * 1000.0

        per_query_ranked.append(
            [
                {
                    "frame_id": str(r.frame_id),
                    "timestamp_seconds": r.timestamp_seconds,
                    "score": r.final_score,
                }
                for r in response.results
            ]
        )
        per_query_latency.append(elapsed)

        for stage, value in response.stage_latency_ms.items():
            if stage in stage_latencies:
                stage_latencies[stage].append(float(value))
        stage_latencies["total"].append(elapsed)

        if use_graph and response.graph_available:
            expansions.append(
                ExpansionObservation(
                    query_id=query.id,
                    depth_requested=depth,
                    nodes_reached=response.graph_expanded_nodes,
                    edges_traversed=response.graph_traversed_edges,
                    seed_count=response.candidate_pool_size,
                    max_depth_reached=response.graph_depth,
                    truncated_depth=response.graph_truncated_depth,
                    truncated_nodes=response.graph_truncated_nodes,
                    expansion_latency_ms=float(
                        response.stage_latency_ms.get("graph_expansion", 0.0)
                    ),
                )
            )

        for result in response.results:
            graph_scores.append(result.graph_score)
            confidence_scores.append(result.confidence_score)

    return {
        "ranked": per_query_ranked,
        "latency_ms": per_query_latency,
        "stage_latency_ms": stage_latencies,
        "expansions": expansions,
        "confidence_scores": confidence_scores,
        "graph_scores": graph_scores,
    }


# ---------------------------------------------------------------------------
# Indexing metrics
# ---------------------------------------------------------------------------


def collect_indexing_metrics(db: Any) -> List[Dict[str, Any]]:
    """Read real frame counts, graph sizes and storage from the database.

    Frame-reduction percentage uses ``videos.frame_count`` (the source video's
    true decoded frame count) against the number of retained keyframes actually
    stored, which is the honest denominator for the adaptive-extraction claim.
    """
    from sqlalchemy import func, select

    from app.models.frame import VideoFrame
    from app.models.graph import GraphEdge, GraphNode
    from app.models.video import Video

    rows: List[Dict[str, Any]] = []
    videos = db.execute(select(Video).where(Video.status == "ready")).scalars().all()
    if not videos:
        return [
            {
                "metrics_status": "N/A",
                "metrics_reason": "no videos with status='ready' in the database",
            }
        ]

    for video in videos:
        retained = (
            db.execute(
                select(func.count())
                .select_from(VideoFrame)
                .where(VideoFrame.video_id == video.id)
            )
            .scalar()
            or 0
        )
        node_count = (
            db.execute(
                select(func.count())
                .select_from(GraphNode)
                .where(GraphNode.video_id == video.id)
            )
            .scalar()
            or 0
        )
        edge_count = (
            db.execute(
                select(func.count())
                .select_from(GraphEdge)
                .join(GraphNode, GraphNode.id == GraphEdge.source_node_id)
                .where(GraphNode.video_id == video.id)
            )
            .scalar()
            or 0
        )

        raw_bytes = None
        try:
            raw_bytes = os.path.getsize(video.path)
        except OSError:
            pass

        rows.append(
            {
                "video_id": str(video.id),
                "filename": video.original_filename,
                "duration_seconds": video.duration_seconds,
                "fps": video.fps,
                "original_frames": video.frame_count,
                "retained_frames": retained,
                "frame_reduction_percent": frame_reduction_percent(
                    int(video.frame_count or 0), int(retained)
                ),
                "graph_nodes": node_count,
                "graph_edges": edge_count,
                "raw_video_bytes": raw_bytes,
            }
        )
    return rows


def collect_storage_metrics(db: Any) -> Dict[str, Any]:
    """Measure on-disk and in-database storage for the ready videos."""
    from app.core.config import get_settings
    from sqlalchemy import func, select

    from app.models.frame import VideoFrame
    from app.models.video import Video

    settings_obj = get_settings()
    videos = db.execute(select(Video).where(Video.status == "ready")).scalars().all()
    if not videos:
        return empty_graph_report("no ready videos to measure")

    raw_bytes = 0
    for video in videos:
        try:
            raw_bytes += os.path.getsize(video.path)
        except OSError:
            pass

    thumb_bytes = _directory_size(Path(settings_obj.FRAME_ROOT))
    clip_bytes = _directory_size(Path(settings_obj.CLIP_ROOT))

    embedding_rows = (
        db.execute(
            select(func.count()).select_from(VideoFrame).where(
                VideoFrame.embedding.isnot(None)
            )
        )
        .scalar()
        or 0
    )
    # 1152 float4 values per row, as stored by pgvector.
    embedding_bytes = int(embedding_rows) * 1152 * 4

    derived = thumb_bytes + clip_bytes + embedding_bytes
    return {
        "measured": True,
        "status": "measured",
        "raw_video_bytes": raw_bytes,
        "thumbnail_bytes": thumb_bytes,
        "clip_bytes": clip_bytes,
        "embedding_bytes_estimate": embedding_bytes,
        "embedding_rows": embedding_rows,
        "derived_bytes": derived,
        "total_bytes": raw_bytes + derived,
        "overhead_ratio": storage_overhead_ratio(raw_bytes, derived),
        "note": (
            "embedding_bytes_estimate assumes 1152 float4 values per row, which is "
            "the declared vector dimension. It is an arithmetic estimate of the "
            "column payload, not a database-reported size."
        ),
    }


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # --- dataset --------------------------------------------------------
    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    validation = validate_dataset(dataset)
    print("Dataset")
    for key, value in validation.items():
        if key in ("scorable_query_ids", "unscorable_query_ids"):
            print(f"  {key}: {len(value)}")
        elif key == "warnings":
            for warning in value:
                print(f"  WARNING: {warning}")
        else:
            print(f"  {key}: {value}")

    if args.dry_run:
        print("\n--dry-run: dataset validated, nothing was executed.")
        return 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata: Dict[str, Any] = {
        "dataset": str(dataset.path),
        "model": args.model,
        "model_checkpoint": KNOWN_MODELS[args.model],
        "top_k": args.top_k,
        "min_score": args.min_score,
        "graph_enabled": bool(args.graph),
        "graph_depth": args.graph_depth,
        "depth_sweep": list(DEPTH_SWEEP) if args.depth_sweep else None,
        "num_queries": len(dataset.queries),
        "num_scorable": len(dataset.scorable),
        "metrics_status": validation["metrics_status"],
        "generated_at": datetime_utc(),
    }

    # --- offline (supplied predictions) or live --------------------------
    retrieval_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    indexing_rows: List[Dict[str, Any]] = []
    graph_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []
    storage: Dict[str, Any] = empty_graph_report("storage metrics not requested")

    offline = args.predictions is not None

    if offline:
        predictions = _load_predictions(args.predictions)
        ranked = [
            predictions.get(q.id, []) for q in dataset.scorable
        ]
        missing = [q.id for q in dataset.scorable if q.id not in predictions]
        for condition in ("A", "B", "C", "D"):
            row = run_condition(
                condition,
                ranked,
                [0.0] * len(ranked),
                dataset=dataset,
                extra={"mode": "offline_predictions"},
            )
            if missing and row.get("metrics_status") == "measured":
                row["metrics_reason"] = (
                    f"no predictions for {len(missing)} query/queries: "
                    f"{', '.join(missing[:5])}"
                )
            ablation_rows.append(row)
        retrieval_rows = ablation_rows[:1]
        graph_rows = [
            {
                "metrics_status": "N/A",
                "metrics_reason": "graph expansion was not executed in offline mode",
            }
        ]
    else:
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            _check_indexed(db, dataset)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            db.close()
            return 3

        try:
            # --- baseline (condition A) and graph run (condition D) --------
            from app.services.graph_ranking_service import ablation_configs
            from app.services.model_service import get_model_service

            # Warm the model outside every measurement window. Without this,
            # condition A absorbs the one-off model-load cost (observed: a 23 s
            # first encode against a 0.30 s median) and the latency comparison
            # reports a load spike as a property of the configuration.
            get_model_service().encode_text("warm up")

            configs = ablation_configs()
            conditions = ["A"] if not args.ablation else ["A", "B", "C", "D"]
            live: Dict[str, Dict[str, Any]] = {}
            for condition in conditions:
                use_graph = condition in ("C", "D")
                result = _live_condition(
                    db,
                    dataset,
                    top_k=args.top_k,
                    graph_depth=args.graph_depth if use_graph else 0,
                    use_graph=use_graph,
                    config=configs[condition],
                    min_score=args.min_score,
                )
                live[condition] = result

                # Score using only the scorable queries, in dataset order.
                scorable_ids = {q.id for q in dataset.scorable}
                idx = 0
                ranked_for_scorable: List[List[Dict[str, Any]]] = []
                for query in dataset.queries:
                    if query.id in scorable_ids:
                        ranked_for_scorable.append(result["ranked"][idx])
                        idx += 1

                row = run_condition(
                    condition,
                    ranked_for_scorable,
                    result["latency_ms"],
                    dataset=dataset,
                    extra={"mode": "live"},
                )
                ablation_rows.append(row)

                latency_rows.extend(
                    _latency_rows_for(condition, result["stage_latency_ms"])
                )

                if condition == "D" and args.graph:
                    graph_report = aggregate_expansions(result["expansions"])
                    # --depth-sweep already writes a measured row per depth; this
                    # expansion-only row carries no metrics, so keeping it would
                    # duplicate depth 2 as an all-N/A row that reads like a failed
                    # measurement in SUMMARY.md.
                    by_depth = (
                        [] if args.depth_sweep else graph_report.get("by_depth", [])
                    )
                    for entry in by_depth:
                        graph_rows.append(
                            {
                                "depth": entry["depth"],
                                "queries": entry["queries"],
                                "nodes_reached_avg": entry["nodes_reached_avg"],
                                "edges_traversed_avg": entry["edges_traversed_avg"],
                                "expansion_latency_ms_avg": entry[
                                    "expansion_latency_ms_avg"
                                ],
                                "expansion_latency_ms_p95": entry[
                                    "expansion_latency_ms_p95"
                                ],
                                "depth_limit_hit_queries": entry[
                                    "depth_limit_hit_queries"
                                ],
                                "node_limit_hit_queries": entry[
                                    "node_limit_hit_queries"
                                ],
                            }
                        )
                    retrieval_rows.append(
                        {
                            "condition": condition,
                            "description": "vector + graph + temporal (production)",
                            "metrics_status": row.get("metrics_status"),
                            "metrics_reason": row.get("metrics_reason"),
                            **{k: v for k, v in row.items() if "@" in k or k in ("mrr", "map")},
                        }
                    )

            # --- graph depth sweep ---------------------------------------
            if args.depth_sweep:
                for depth in DEPTH_SWEEP:
                    result = _live_condition(
                        db,
                        dataset,
                        top_k=args.top_k,
                        graph_depth=depth,
                        use_graph=depth > 0,
                        config=configs["A"] if depth == 0 else configs["D"],
                        min_score=args.min_score,
                    )
                    scorable_ids = {q.id for q in dataset.scorable}
                    idx = 0
                    ranked_for_scorable = []
                    for query in dataset.queries:
                        if query.id in scorable_ids:
                            ranked_for_scorable.append(result["ranked"][idx])
                            idx += 1
                    row = run_condition(
                        "D" if depth > 0 else "A",
                        ranked_for_scorable,
                        result["latency_ms"],
                        dataset=dataset,
                        extra={"mode": "live", "graph_depth": depth},
                    )
                    report = aggregate_expansions(result["expansions"])
                    by_depth = report.get("by_depth") or [
                        {
                            "nodes_reached_avg": 0.0,
                            "edges_traversed_avg": 0.0,
                            "expansion_latency_ms_avg": 0.0,
                        }
                    ]
                    graph_rows.append(
                        {
                            "depth": depth,
                            "queries": len(result["ranked"]),
                            "metrics_status": row.get("metrics_status"),
                            "metrics_reason": row.get("metrics_reason"),
                            "recall@10": row.get("recall@10"),
                            "precision@10": row.get("precision@10"),
                            "ndcg@10": row.get("ndcg@10"),
                            "mrr": row.get("mrr"),
                            "map": row.get("map"),
                            "latency_avg_ms": row.get("latency_avg_ms"),
                            "latency_p95_ms": row.get("latency_p95_ms"),
                            **by_depth[0],
                        }
                    )

            # --- indexing / storage -------------------------------------
            if args.index_metrics:
                indexing_rows = collect_indexing_metrics(db)
                storage = collect_storage_metrics(db)
                for row in indexing_rows:
                    row["storage_overhead_ratio"] = storage.get("overhead_ratio")
                    row["derived_bytes"] = storage.get("derived_bytes")

            # --- model comparison ---------------------------------------
            model_rows = _model_rows(
                dataset, offline=offline, other=args.other_model
            )
        finally:
            db.close()

    written = reporting.write_all(
        output_dir,
        run_metadata=run_metadata,
        retrieval_rows=retrieval_rows,
        model_rows=model_rows,
        ablation_rows=ablation_rows,
        latency_rows=latency_rows,
        indexing_rows=indexing_rows,
        graph_rows=graph_rows,
    )

    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name} -> {path}")
    if not indexing_rows:
        print(
            "\nNote: indexing_results.csv is N/A. Re-run with --index-metrics to "
            "collect frame-reduction and storage measurements."
        )
    print(
        "\nNote: model_comparison is N/A unless a second checkpoint was actually "
        "indexed and scored. See the 'metrics_reason' column."
    )
    return 0


def _model_rows(
    dataset: Dataset, *, offline: bool, other: str
) -> List[Dict[str, Any]]:
    """SigLIP vs SigLIP-2 table.

    Populated only for checkpoints that were genuinely evaluated. The
    specification's comparison requires both models to be indexed with
    identical frames, queries, ground truth, top-K and metrics; when that has
    not happened, every cell is N/A with the reason recorded.
    """
    rows: List[Dict[str, Any]] = []
    for label in ("siglip", "siglip2"):
        rows.append(
            {
                "model": label,
                "checkpoint": KNOWN_MODELS[label],
                "metrics_status": "N/A",
                "metrics_reason": (
                    "not evaluated in this run. The comparison requires both "
                    "checkpoints to be indexed over identical frames and scored "
                    "on the same queries; run the pipeline once per MODEL_NAME and "
                    "merge the results."
                ),
            }
        )
    return rows


def _latency_rows_for(
    condition: str, stage_latencies: Dict[str, List[float]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage, values in stage_latencies.items():
        stats = latency_stats(values)
        rows.append(
            {
                "condition": condition,
                "stage": stage,
                "samples": stats["count"],
                "avg_ms": stats["avg_ms"],
                "median_ms": stats["median_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "min_ms": stats["min_ms"],
                "max_ms": stats["max_ms"],
            }
        )
    return rows


def _check_indexed(db: Any, dataset: Dataset) -> None:
    """Fail fast when the database has nothing indexed."""
    from sqlalchemy import func, select

    from app.models.frame import VideoFrame
    from app.models.video import Video

    ready = (
        db.execute(
            select(func.count()).select_from(Video).where(Video.status == "ready")
        ).scalar()
        or 0
    )
    if ready == 0:
        raise RuntimeError(
            "No videos with status='ready' in the database. Upload and index a "
            "video first (POST /api/v1/videos, then wait for status='ready')."
        )
    frames = (
        db.execute(
            select(func.count())
            .select_from(VideoFrame)
            .where(VideoFrame.embedding.isnot(None))
        )
        .scalar()
        or 0
    )
    if frames == 0:
        raise RuntimeError(
            "Frames exist but none have an embedding. Re-index the video so "
            "SigLIP-2 embeddings are stored."
        )


def datetime_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


if __name__ == "__main__":
    raise SystemExit(main())
