import time
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.video import Video
from app.models.frame import VideoFrame
from app.schemas.search import SearchRequest, SearchResult, SearchResponse
from app.services.model_service import get_model_service
from app.services.clip_service import generate_clip

logger = setup_logging()
settings = get_settings()


def _temporal_group(results: List[dict], window: float) -> List[dict]:
    """
    Group nearby timestamps into meaningful segments.
    Keeps the highest-scoring frame as representative of each group.
    """
    if not results:
        return []

    # Already sorted by score descending from the query
    # Re-sort by timestamp for grouping
    by_time = sorted(results, key=lambda r: r["timestamp_seconds"])
    groups: List[List[dict]] = []
    current: List[dict] = [by_time[0]]

    for r in by_time[1:]:
        if r["timestamp_seconds"] - current[-1]["timestamp_seconds"] <= window:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)

    # For each group keep the best score and assign group_id
    final = []
    for gid, group in enumerate(groups):
        best = max(group, key=lambda x: x["score"])
        best["group_id"] = gid
        final.append(best)

    # Re-sort by original score
    final.sort(key=lambda x: x["score"], reverse=True)
    return final


def search(db: Session, request: SearchRequest) -> SearchResponse:
    start = time.perf_counter()
    logger.info(f"Search query received: '{request.query}' | top_k={request.top_k}")

    model = get_model_service()
    query_emb = model.encode_text(request.query)[0].tolist()

    # Build SQL with optional video filter
    # Cosine distance: <=>   similarity = 1 - distance
    params = {
        "emb": str(query_emb),
        "limit": request.top_k * 3,  # over-fetch for grouping
        "min_score": request.min_score,
    }

    sql = """
        SELECT
            f.id AS frame_id,
            f.video_id,
            v.original_filename AS video_name,
            f.timestamp_seconds,
            f.thumbnail_path,
            1 - (f.embedding <=> :emb::vector) AS score
        FROM video_frames f
        JOIN videos v ON v.id = f.video_id
        WHERE v.status = 'ready'
          AND 1 - (f.embedding <=> :emb::vector) >= :min_score
    """
    if request.video_id:
        sql += " AND f.video_id = :video_id"
        params["video_id"] = str(request.video_id)

    sql += " ORDER BY f.embedding <=> :emb::vector ASC LIMIT :limit"

    rows = db.execute(text(sql), params).mappings().all()

    raw_results = []
    for row in rows:
        thumb = row["thumbnail_path"]
        # Convert absolute path to URL path
        # Assuming FRAME_ROOT is mounted and served under /media/frames
        thumb_url = f"/media/frames/{row['video_id']}/{thumb.split('/')[-1]}"
        raw_results.append({
            "frame_id": row["frame_id"],
            "video_id": row["video_id"],
            "video_name": row["video_name"],
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "score": float(row["score"]),
            "thumbnail_url": thumb_url,
            "clip_url": None,
            "group_id": None,
        })

    # Temporal grouping
    grouped = _temporal_group(raw_results, settings.TEMPORAL_GROUP_WINDOW)
    final = grouped[: request.top_k]

    latency = (time.perf_counter() - start) * 1000
    logger.info(f"Search completed | results={len(final)} | latency_ms={latency:.1f}")

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in final],
        total=len(final),
        latency_ms=round(latency, 2),
    )


def generate_result_clip(db: Session, frame_id: uuid.UUID) -> str:
    """Generate (or return existing) clip for a search result frame."""
    frame = db.query(VideoFrame).filter(VideoFrame.id == frame_id).first()
    if not frame:
        raise ValueError("Frame not found")
    video = db.query(Video).filter(Video.id == frame.video_id).first()
    if not video:
        raise ValueError("Video not found")

    clip_name = generate_clip(
        video_path=video.path,
        timestamp_seconds=frame.timestamp_seconds,
        duration=5.0,
        video_id=str(video.id),
    )
    return f"/media/clips/{clip_name}"
