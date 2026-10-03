"""Graph routes (Phase 2).

Both endpoints are **additive**. Nothing here shadows, wraps or alters
``POST /api/v1/search``, whose contract is unchanged.

Graph mode is opt-in *by route*, not by a flag on the existing route. That was
the explicit choice: adding an optional field to ``SearchRequest`` would change
the response type for a request that does not ask for a graph, and any
difference in the old response would be a breaking change for existing clients.
A separate route makes the opt-in unambiguous and lets the new response carry
fields (``graph_score``, ``contributions``, ``related_nodes``) that have no
meaning in the vector-only response.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.controllers import graph_controller
from app.db.session import get_db
from app.schemas.graph import (
    GraphSearchRequest,
    GraphSearchResponse,
    VideoGraphResponse,
)

router = APIRouter()


@router.post(
    "/search/graph",
    response_model=GraphSearchResponse,
    summary="Graph-aware semantic video search",
    description=(
        "Vector candidate retrieval followed by a bounded graph expansion "
        "(max_depth AND max_nodes) and weighted re-ranking. Returns the full "
        "score breakdown so results are explainable without extra calls.\n\n"
        "`graph_depth=0` reproduces the vector-only ranking, which is the "
        "baseline used in the evaluation study."
    ),
)
def graph_search(request: GraphSearchRequest, db: Session = Depends(get_db)):
    return graph_controller.perform_graph_search(db, request)


@router.get(
    "/videos/{video_id}/graph",
    response_model=VideoGraphResponse,
    summary="Get the relational graph of one video",
    description=(
        "Returns every node and edge built for a video, plus per-type counts. "
        "`status` is 'pending' while indexing is still running, 'ready' once "
        "indexed (an empty node list is a valid ready graph), and 'error' if "
        "indexing failed."
    ),
)
def video_graph(video_id: UUID, db: Session = Depends(get_db)):
    return graph_controller.fetch_video_graph(db, video_id)
