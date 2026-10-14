"""Pydantic schemas for the graph layer (Phase 2).

These are **new** models. ``app/schemas/search.py`` is left byte-identical, so
``POST /api/v1/search`` and its ``SearchResponse`` contract are untouched. A
graph result is a different shape with different fields, and it is expressed
with different models rather than by widening the existing ones.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class GraphSearchRequest(BaseModel):
    """Body of ``POST /api/v1/search/graph``.

    Field names mirror :class:`~app.schemas.search.SearchRequest` where the
    meaning is the same (``query``, ``video_id``, ``top_k``) so the two
    endpoints feel identical to a client, plus ``graph_depth`` for the new
    capability.
    """

    query: str = Field(..., min_length=2, max_length=500)
    video_id: Optional[UUID] = None
    top_k: int = Field(default=10, ge=1, le=100)
    graph_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Maximum graph expansion hops. 0 disables expansion entirely, "
            "which yields the vector-only ranking. Hard-capped by "
            "GRAPH_MAX_DEPTH in the environment."
        ),
    )
    min_score: float = Field(
        default=-1.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine-similarity floor for the candidate pool. Defaults to the "
            "lowest possible cosine (no floor); see SearchRequest.min_score for "
            "why a 0.0 default is wrong for SigLIP-2 embeddings."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class RelatedNode(BaseModel):
    """One node the traversal reached from a result frame.

    Populated from real edges only. With no extractor configured these are
    SCENE and FRAME nodes reached via CONTAINS / BELONGS_TO / SIMILAR_TO /
    FOLLOWED_BY; there is no fabricated concept layer.
    """

    id: str
    node_type: str
    label: Optional[str] = None
    entity_key: Optional[str] = None
    frame_id: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    confidence: float = 0.0
    depth: Optional[int] = None


class RelationshipEvidence(BaseModel):
    """One traversed edge, including the basis that justifies it.

    ``basis`` is read from the edge's own metadata and is one of
    ``deterministic_containment``, ``timestamp_ordering``,
    ``measured_cosine_similarity`` or ``extractor:<name>``. Surfacing it lets
    the frontend state *why* an edge exists rather than merely asserting it.
    """

    relationship: str
    source_node_id: str
    target_node_id: str
    weight: float
    confidence: float
    basis: Optional[str] = None


class GraphSearchResult(BaseModel):
    """One re-ranked result with its full score breakdown.

    ``final_score`` is the weighted combination documented in
    ``graph_ranking_service``. ``contributions`` is the additive breakdown of
    that number per term, and it is what the frontend's "Why this result?"
    panel renders - no extra endpoint needed.
    """

    frame_id: UUID
    video_id: Optional[UUID] = None
    video_name: Optional[str] = None
    timestamp_seconds: float
    thumbnail_url: Optional[str] = None
    clip_url: Optional[str] = None

    vector_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    confidence_score: float = 0.0
    final_score: float = 0.0

    contributions: Dict[str, float] = Field(default_factory=dict)
    related_nodes: List[RelatedNode] = Field(default_factory=list)
    relationships: List[RelationshipEvidence] = Field(default_factory=list)

    is_seed: bool = True
    graph_depth_reached: Optional[int] = None


class GraphSearchResponse(BaseModel):
    """Body of ``POST /api/v1/search/graph``."""

    query: str
    results: List[GraphSearchResult]
    total: int
    latency_ms: float

    # --- diagnostics, so a caller can tell a thin graph from a broken one ---
    graph_available: bool = True
    graph_depth: int = 0
    graph_expanded_nodes: int = 0
    graph_traversed_edges: int = 0
    graph_truncated_depth: bool = False
    graph_truncated_nodes: bool = False
    candidate_pool_size: int = 0
    stage_latency_ms: Dict[str, float] = Field(default_factory=dict)


class GraphNodeOut(BaseModel):
    """A node as returned by ``GET /api/v1/videos/{video_id}/graph``."""

    id: str
    video_id: str
    node_type: str
    entity_key: str
    label: Optional[str] = None
    description: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    frame_id: Optional[str] = None
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class GraphEdgeOut(BaseModel):
    """An edge as returned by ``GET /api/v1/videos/{video_id}/graph``."""

    id: str
    source_node_id: str
    target_node_id: str
    relationship: str
    weight: float
    confidence: float
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class VideoGraphResponse(BaseModel):
    """Body of ``GET /api/v1/videos/{video_id}/graph``.

    An empty ``nodes``/``edges`` list with ``status='ready'`` is a valid and
    meaningful response: it says "this video is indexed and its graph is
    genuinely empty", which is distinct from "not indexed yet"
    (``status='pending'``) or "indexing failed" (``status='error'``). The
    frontend relies on that distinction to avoid showing a spurious error.
    """

    video_id: str
    status: str = "ready"
    node_count: int = 0
    edge_count: int = 0
    node_type_counts: Dict[str, int] = Field(default_factory=dict)
    relationship_counts: Dict[str, int] = Field(default_factory=dict)
    truncated: bool = False
    nodes: List[GraphNodeOut] = Field(default_factory=list)
    edges: List[GraphEdgeOut] = Field(default_factory=list)
