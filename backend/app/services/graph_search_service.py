"""Graph-aware retrieval pipeline (Phase 2).

This is a *new* code path. ``app/services/search_service.py`` is left
byte-identical, so ``POST /api/v1/search`` behaves exactly as it always has. The
graph pipeline reuses the same pgvector candidate query, so the vector-only
baseline and the graph pipeline are fed identical input and any difference in
output is attributable to the graph, not to a different candidate generator.

Flow::

    query text
      -> SigLIP-2 text embedding
      -> pgvector top-N candidate frames          (identical SQL to /search)
      -> bounded graph expansion                  (max_depth AND max_nodes)
      -> graph + temporal + confidence scoring
      -> weighted re-ranking
      -> ranked results with full score breakdown

Every stage is timed, and the per-stage timings are returned in the response so
the Phase 3 latency breakdown is measured from production code rather than
reconstructed afterwards.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.schemas.graph import (
    GraphSearchRequest,
    GraphSearchResponse,
    GraphSearchResult,
    RelatedNode,
    RelationshipEvidence,
    VideoGraphResponse,
)
from app.services.graph_ranking_service import RankingConfig, rank_candidates
from app.services.graph_service import expand_from_seeds, serialize_graph
from app.services.model_service import get_model_service

logger = setup_logging()
settings = get_settings()


# The candidate query is byte-identical in shape to the one in search_service,
# differing only in the LIMIT (a fixed pool rather than top_k * 3) and in
# selecting the embedding back so the vector stage can be re-scored locally.
# Keeping one SQL text for both paths is what makes the ablation a controlled
# comparison.
_CANDIDATE_SQL = """
    SELECT
        f.id AS frame_id,
        f.video_id,
        v.original_filename AS video_name,
        f.timestamp_seconds,
        f.thumbnail_path,
        1 - (f.embedding <=> CAST(:emb AS vector)) AS vector_score
    FROM video_frames f
    JOIN videos v ON v.id = f.video_id
    WHERE v.status = 'ready'
      AND 1 - (f.embedding <=> CAST(:emb AS vector)) >= :min_score
      AND f.embedding IS NOT NULL
"""

_CANDIDATE_TAIL = """
    ORDER BY f.embedding <=> CAST(:emb AS vector) ASC
    LIMIT :limit
"""


def _fetch_candidates(
    db: Session, query_embedding: List[float], request: GraphSearchRequest
) -> List[Dict[str, Any]]:
    """Vector candidate retrieval. Same operator, same filter as /search."""
    pool_size = request.top_k * settings.GRAPH_CANDIDATE_MULTIPLIER
    params: Dict[str, Any] = {
        "emb": str(query_embedding),
        "limit": pool_size,
        "min_score": request.min_score,
    }
    sql = _CANDIDATE_SQL
    if request.video_id:
        sql += " AND f.video_id = :video_id"
        params["video_id"] = str(request.video_id)
    sql += _CANDIDATE_TAIL

    rows = db.execute(text(sql), params).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "frame_id": row["frame_id"],
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "thumbnail_url": _thumbnail_url(row["video_id"], row["thumbnail_path"]),
                "vector_score": float(row["vector_score"]),
            }
        )
    return out


def _thumbnail_url(video_id: Any, thumbnail_path: str) -> str:
    """Rebuild the public thumbnail URL.

    Mirrors ``search_service.search``: the stored absolute path is discarded
    and reconstructed from the video id and the basename, which is what the
    static mount under ``/media/frames`` expects.
    """
    return f"/media/frames/{video_id}/{thumbnail_path.split('/')[-1]}"


def graph_search(
    db: Session,
    request: GraphSearchRequest,
    *,
    config: Optional[RankingConfig] = None,
) -> GraphSearchResponse:
    """Execute the graph-aware retrieval pipeline.

    Parameters
    ----------
    request:
        Validated request. ``graph_depth=0`` is honoured as "vector only",
        which is condition A of the ablation and needs no separate code path.
    config:
        Weight override, used by the evaluation harness to evaluate conditions
        A/B/C/D without reimplementing any of them.
    """
    stage_ms: Dict[str, float] = {}

    # --- stage 1: query embedding -----------------------------------------
    t0 = time.perf_counter()
    model = get_model_service()
    query_embedding = model.encode_text(request.query)[0].tolist()
    stage_ms["embedding"] = (time.perf_counter() - t0) * 1000.0

    # --- stage 2: vector candidate retrieval ------------------------------
    t0 = time.perf_counter()
    candidates = _fetch_candidates(db, query_embedding, request)
    stage_ms["vector_retrieval"] = (time.perf_counter() - t0) * 1000.0

    if not candidates:
        return GraphSearchResponse(
            query=request.query,
            results=[],
            total=0,
            latency_ms=round(sum(stage_ms.values()), 2),
            graph_available=False,
            graph_depth=0,
            candidate_pool_size=0,
            stage_latency_ms={k: round(v, 2) for k, v in stage_ms.items()},
        )

    # --- stage 3: bounded graph expansion ---------------------------------
    t0 = time.perf_counter()
    depth = min(request.graph_depth, settings.GRAPH_MAX_DEPTH)
    expansion = None
    if depth > 0 and settings.GRAPH_ENABLED:
        expansion = expand_from_seeds(
            db,
            [c["frame_id"] for c in candidates],
            max_depth=depth,
            max_nodes=settings.GRAPH_MAX_NODES,
        )
    stage_ms["graph_expansion"] = (time.perf_counter() - t0) * 1000.0

    graph_available = bool(expansion and expansion.edges)

    # --- stage 4: scoring and re-ranking ----------------------------------
    t0 = time.perf_counter()
    scored = rank_candidates(
        candidates,
        expansion,
        config=config,
        window_seconds=settings.TEMPORAL_WINDOW_SECONDS,
        top_k=request.top_k,
    )
    stage_ms["graph_reranking"] = (time.perf_counter() - t0) * 1000.0

    results = [
        GraphSearchResult(
            frame_id=item.frame_id,
            video_id=item.video_id,
            video_name=next(
                (c["video_name"] for c in candidates if c["frame_id"] == item.frame_id),
                None,
            ),
            timestamp_seconds=item.timestamp_seconds,
            thumbnail_url=next(
                (c["thumbnail_url"] for c in candidates if c["frame_id"] == item.frame_id),
                None,
            ),
            vector_score=item.vector_score,
            graph_score=item.graph_score,
            temporal_score=item.temporal_score,
            confidence_score=item.confidence_score,
            final_score=item.final_score,
            contributions=item.contributions,
            related_nodes=[RelatedNode(**n) for n in item.related_nodes],
            relationships=[RelationshipEvidence(**r) for r in item.relationships],
            is_seed=item.is_seed,
            graph_depth_reached=item.depth,
        )
        for item in scored
    ]

    total_ms = sum(stage_ms.values())
    logger.info(
        f"Graph search '{request.query}' | candidates={len(candidates)} | "
        f"results={len(results)} | depth={depth} | "
        f"expanded_nodes={len(expansion.nodes) if expansion else 0} | "
        f"edges={len(expansion.edges) if expansion else 0} | "
        f"latency_ms={total_ms:.1f}"
    )

    return GraphSearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        latency_ms=round(total_ms, 2),
        graph_available=graph_available,
        graph_depth=depth,
        graph_expanded_nodes=len(expansion.nodes) if expansion else 0,
        graph_traversed_edges=len(expansion.edges) if expansion else 0,
        graph_truncated_depth=expansion.truncated_depth if expansion else False,
        graph_truncated_nodes=expansion.truncated_nodes if expansion else False,
        candidate_pool_size=len(candidates),
        stage_latency_ms={k: round(v, 2) for k, v in stage_ms.items()},
    )


def get_video_graph(
    db: Session, video_id: uuid.UUID
) -> VideoGraphResponse:
    """Return a video's whole graph, plus honest aggregate counts.

    ``status`` distinguishes the three cases a frontend must not conflate:

    ``pending``  the video is not ``ready`` yet, so no graph was ever built
    ``ready``    indexed; ``nodes``/``edges`` reflect the real graph, which may
                 legitimately be empty
    ``error``    the video failed to index
    """
    from app.models.video import Video

    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise ValueError("Video not found")

    if video.status != "ready":
        return VideoGraphResponse(
            video_id=str(video_id),
            status="pending" if video.status in ("queued", "processing") else "error",
            node_count=0,
            edge_count=0,
        )

    payload = serialize_graph(db, video_id)
    node_type_counts = Counter(n["node_type"] for n in payload["nodes"])
    relationship_counts = Counter(e["relationship"] for e in payload["edges"])

    return VideoGraphResponse(
        video_id=str(video_id),
        status="ready",
        node_count=len(payload["nodes"]),
        edge_count=len(payload["edges"]),
        node_type_counts=dict(sorted(node_type_counts.items())),
        relationship_counts=dict(sorted(relationship_counts.items())),
        truncated=len(payload["nodes"]) >= 5000 or len(payload["edges"]) >= 20000,
        nodes=payload["nodes"],
        edges=payload["edges"],
    )
