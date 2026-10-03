"""HTTP layer for the graph endpoints (Phase 2).

Same shape as the existing controllers: thin, and responsible only for mapping
domain errors onto HTTP status codes. All logic lives in the services.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.schemas.graph import GraphSearchRequest, GraphSearchResponse, VideoGraphResponse
from app.services import graph_search_service

logger = logging.getLogger("videocontrolf")


def perform_graph_search(db: Session, request: GraphSearchRequest) -> GraphSearchResponse:
    """Handle ``POST /api/v1/search/graph``."""
    try:
        return graph_search_service.graph_search(db, request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Graph search failed")


def fetch_video_graph(db: Session, video_id: uuid.UUID) -> VideoGraphResponse:
    """Handle ``GET /api/v1/videos/{video_id}/graph``."""
    try:
        return graph_search_service.get_video_graph(db, video_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph fetch failed for %s: %s", video_id, exc)
        raise HTTPException(status_code=500, detail="Graph fetch failed")
