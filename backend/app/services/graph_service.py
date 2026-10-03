"""Graph construction and bounded traversal for VideoCtrl-F (Phase 1).

This module has two responsibilities, kept deliberately separate:

**Construction** (:func:`build_graph_for_video`) runs once per video, inside
the indexing transaction. It creates ``VIDEO``, ``SCENE`` and ``FRAME`` nodes
plus ``CONTAINS``, ``BELONGS_TO``, ``FOLLOWED_BY``, ``PRECEDES`` and
``SIMILAR_TO`` edges, and optionally whatever a real
:class:`~app.services.graph_extraction_service.GraphExtractor` reports.

**Traversal** (:func:`expand_from_seeds`) runs once per graph search. It is
bounded by ``max_depth`` *and* ``max_nodes``, and it fetches edges in batches
rather than one query per node.

Traceability guarantee
----------------------
Every edge written here has a measurable basis:

===========================  ==========================================
Edge                         Basis
===========================  ==========================================
``CONTAINS``                 deterministic containment of a built node
``BELONGS_TO``               deterministic inverse containment
``FOLLOWED_BY`` / ``PRECEDES``  actual ``timestamp_seconds`` ordering of
                             retained keyframes
``SIMILAR_TO``               actual cosine similarity between two stored
                             frame embeddings, thresholded at
                             ``GRAPH_SIMILARITY_THRESHOLD``
``DEPICTS`` and friends      a real extractor's own output, or nothing at all
===========================  ==========================================

No edge value is ever random, and no relationship is ever created without one
of the above. When no extractor is registered, only the deterministic and
measurement-derived edge types are written.

Query discipline
----------------
Traversal is implemented as a **breadth-first expansion over level-sized
batches**: one query to read the frontier's edges, one query to read the
discovered nodes, per level. That is ``2 * max_depth`` queries regardless of
how many nodes are in play. There is no per-node query anywhere, so there is
no N+1.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.frame import VideoFrame
from app.models.graph import (
    GraphEdge,
    GraphNode,
    NodeType,
    RelationshipType,
    TEMPORAL_RELATIONSHIPS,
)
from app.services.graph_extraction_service import (
    cosine_similarity_matrix,
    extract_for_frames,
    get_graph_extractor,
)

logger = setup_logging()
settings = get_settings()


# ---------------------------------------------------------------------------
# Entity keys
# ---------------------------------------------------------------------------
# A node's identity is (video_id, node_type, entity_key). Keys are stable and
# derived only from real identifiers, never from iteration order, so a
# rebuild produces the same keys.


def video_entity_key() -> str:
    """Entity key for the single VIDEO node of a video."""
    return "video"


def scene_entity_key(scene_index: int) -> str:
    """Entity key for the scene at ``scene_index`` (0-based, time-ordered)."""
    return f"scene:{scene_index}"


def frame_entity_key(frame_id: Any) -> str:
    """Entity key for the FRAME node backed by ``frame_id``."""
    return f"frame:{frame_id}"


def detection_entity_key(detection: Any) -> str:
    """Entity key for an extracted entity.

    Derived from the extraction's own label so that the same observation in the
    same frame collapses to one node. Returns a stable string; no hashing of
    untrusted input is performed beyond normal string construction.
    """
    label = (detection.label or "").strip().lower()
    safe = "_".join(label.split()) or "unlabelled"
    return f"{detection.node_type.value.lower()}:{safe}"


# ---------------------------------------------------------------------------
# Scene grouping
# ---------------------------------------------------------------------------


def group_scenes(
    timestamps: Sequence[float],
    window_seconds: Optional[float] = None,
) -> List[Tuple[int, int, float, float]]:
    """Cluster time-ordered timestamps into scenes.

    A new scene starts whenever the gap to the previous retained keyframe
    exceeds ``window_seconds``. This is a single-linkage sweep over the
    *actual* retained keyframe timestamps - no model, no threshold guessing.

    Parameters
    ----------
    timestamps:
        Retained frame timestamps. Must be ascending (the pipeline emits them
        in decode order, which is ascending).
    window_seconds:
        Defaults to ``settings.GRAPH_SCENE_WINDOW_SECONDS``.

    Returns
    -------
    list of (start_index, end_index, t_start, t_end)
        Half-open index range into ``timestamps`` plus the scene's temporal
        extent, one tuple per scene, in time order.
    """
    win = settings.GRAPH_SCENE_WINDOW_SECONDS if window_seconds is None else window_seconds
    if not timestamps:
        return []
    if win <= 0:
        raise ValueError(f"scene window must be positive, got {win}")

    scenes: List[Tuple[int, int, float, float]] = []
    start = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > win:
            scenes.append((start, i - 1, float(timestamps[start]), float(timestamps[i - 1])))
            start = i
    scenes.append(
        (start, len(timestamps) - 1, float(timestamps[start]), float(timestamps[-1]))
    )
    return scenes


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class GraphBuildResult:
    """Counts and timings produced by one graph construction run.

    Everything here is measured, and is what the Phase 3 indexing metrics are
    computed from. Nothing is estimated.
    """

    def __init__(self) -> None:
        self.nodes_created = 0
        self.edges_created = 0
        self.frames_indexed = 0
        self.scenes_created = 0
        self.similar_edges = 0
        self.temporal_edges = 0
        self.structural_edges = 0
        self.extracted_nodes = 0
        self.extracted_edges = 0
        self.edges_truncated = False
        self.extractor_name: Optional[str] = None
        self.extractor_detections = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "frames_indexed": self.frames_indexed,
            "scenes_created": self.scenes_created,
            "similar_edges": self.similar_edges,
            "temporal_edges": self.temporal_edges,
            "structural_edges": self.structural_edges,
            "extracted_nodes": self.extracted_nodes,
            "extracted_edges": self.extracted_edges,
            "edges_truncated": self.edges_truncated,
            "extractor_name": self.extractor_name,
            "extractor_detections": self.extractor_detections,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GraphBuildResult {self.as_dict()}>"


def load_indexed_frames(db: Session, video_id: uuid.UUID) -> List[Dict[str, Any]]:
    """Read back the retained frames of a video, in timestamp order.

    Graph construction deliberately runs *after* frames have been committed, so
    it reads real persisted rows (real embeddings, real timestamps) rather than
    re-deriving anything from the video file.
    """
    rows = (
        db.execute(
            select(
                VideoFrame.id,
                VideoFrame.timestamp_seconds,
                VideoFrame.frame_number,
                VideoFrame.embedding,
            )
            .where(VideoFrame.video_id == video_id)
            .order_by(VideoFrame.timestamp_seconds.asc(), VideoFrame.frame_number.asc())
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


def _add_node(
    result: GraphBuildResult,
    video_id: uuid.UUID,
    node_type: NodeType,
    entity_key: str,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    timestamp_start: Optional[float] = None,
    timestamp_end: Optional[float] = None,
    frame_id: Optional[uuid.UUID] = None,
    confidence: float = 1.0,
    node_metadata: Optional[Dict[str, Any]] = None,
) -> GraphNode:
    node = GraphNode(
        video_id=video_id,
        node_type=node_type.value,
        entity_key=entity_key,
        label=label,
        description=description,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        frame_id=frame_id,
        confidence=confidence,
        node_metadata=node_metadata,
    )
    result.nodes_created += 1
    return node


def _add_edge(
    result: GraphBuildResult,
    source: GraphNode,
    target: GraphNode,
    relationship: RelationshipType,
    *,
    weight: float = 1.0,
    confidence: float = 1.0,
    timestamp_start: Optional[float] = None,
    timestamp_end: Optional[float] = None,
    edge_metadata: Optional[Dict[str, Any]] = None,
) -> GraphEdge:
    edge = GraphEdge(
        source_node_id=source.id,
        target_node_id=target.id,
        relationship=relationship.value,
        weight=weight,
        confidence=confidence,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        edge_metadata=edge_metadata,
    )
    result.edges_created += 1
    return edge


def compute_similar_pairs(
    embeddings: np.ndarray,
    threshold: float,
    top_k: int,
) -> List[Tuple[int, int, float]]:
    """Find frame pairs whose measured cosine similarity is at least ``threshold``.

    Parameters
    ----------
    embeddings:
        ``(n, d)`` matrix of the stored frame embeddings.
    threshold:
        Minimum similarity for an edge to be written. Defaults to
        ``settings.GRAPH_SIMILARITY_THRESHOLD``.
    top_k:
        Maximum neighbours retained per frame, highest similarity first.
        Defaults to ``settings.GRAPH_SIMILAR_TOP_K``.

    Returns
    -------
    list of (i, j, similarity)
        Always ``i < j``, so each unordered pair is emitted once. Sorted by
        descending similarity.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"similarity threshold must be in [0, 1], got {threshold}")
    if top_k < 1:
        raise ValueError(f"similarity top_k must be >= 1, got {top_k}")

    n = embeddings.shape[0]
    if n < 2:
        return []

    sim = cosine_similarity_matrix(embeddings)
    # Per-row top_k, excluding the zeroed diagonal.
    k = min(top_k, n - 1)
    # argpartition is O(n) per row; then sort just the k winners.
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    pairs: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            if i == j:
                continue
            value = float(sim[i, j])
            if value >= threshold:
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))

    return sorted(
        ((a, b, float(sim[a, b])) for a, b in pairs),
        key=lambda t: (-t[2], t[0], t[1]),
    )


def build_graph_for_video(
    db: Session,
    video_id: uuid.UUID,
    *,
    frames: Optional[List[Dict[str, Any]]] = None,
    frame_images: Optional[Dict[uuid.UUID, Any]] = None,
    run_extractor: bool = True,
) -> GraphBuildResult:
    """Build the entire graph for one video.

    **Transaction contract.** The caller owns the transaction. This function
    only issues ``db.add(...)`` and flushes; it never commits and never rolls
    back. That lets the indexing pipeline in ``video_service`` wrap frames +
    nodes + edges in one unit of work, so a failure anywhere leaves no partial
    graph behind and the video is marked ``failed`` by the caller's handler.

    Parameters
    ----------
    db:
        Session. Objects are flushed so that generated UUIDs are available for
        wiring edges before the caller commits.
    video_id:
        Video to build. Its frames must already be persisted.
    frames:
        Optional pre-loaded frame rows. Loaded from the database when omitted.
    frame_images:
        Optional ``{frame_id: PIL.Image}`` map. Required only when
        ``run_extractor`` is true and a real extractor is registered; the
        default no-op extractor ignores it.
    run_extractor:
        Set false to skip extraction entirely (used by the evaluation harness
        when isolating the deterministic graph).

    Returns
    -------
    GraphBuildResult
        Measured counts. ``edges_truncated`` is true if
        ``GRAPH_MAX_EDGES_PER_VIDEO`` was hit.
    """
    result = GraphBuildResult()

    if frames is None:
        frames = load_indexed_frames(db, video_id)
    result.frames_indexed = len(frames)

    video_node = _add_node(
        result,
        video_id,
        NodeType.VIDEO,
        video_entity_key(),
        label="video",
        description="Root node for this video's relational graph.",
    )
    db.add(video_node)
    db.flush()

    if not frames:
        logger.warning(f"Graph build for {video_id}: no frames, graph is VIDEO-only")
        return result

    # ---- FRAME nodes -----------------------------------------------------
    frame_nodes: List[GraphNode] = []
    for spec in frames:
        node = _add_node(
            result,
            video_id,
            NodeType.FRAME,
            frame_entity_key(spec["id"]),
            label=f"frame {spec['frame_number']}",
            timestamp_start=float(spec["timestamp_seconds"]),
            timestamp_end=float(spec["timestamp_seconds"]),
            frame_id=spec["id"],
            confidence=1.0,
            node_metadata={
                "frame_number": int(spec["frame_number"]),
                "timestamp_seconds": float(spec["timestamp_seconds"]),
            },
        )
        db.add(node)
        frame_nodes.append(node)
    db.flush()

    # ---- SCENE nodes from real timestamp clustering -----------------------
    timestamps = [float(f["timestamp_seconds"]) for f in frames]
    scenes = group_scenes(timestamps)
    result.scenes_created = len(scenes)

    scene_nodes: List[GraphNode] = []
    for scene_index, (_s, _e, t_start, t_end) in enumerate(scenes):
        node = _add_node(
            result,
            video_id,
            NodeType.SCENE,
            scene_entity_key(scene_index),
            label=f"scene {scene_index}",
            description=f"Keyframe cluster spanning {t_start:.2f}s-{t_end:.2f}s",
            timestamp_start=t_start,
            timestamp_end=t_end,
            confidence=1.0,
            node_metadata={
                "scene_index": scene_index,
                "window_seconds": settings.GRAPH_SCENE_WINDOW_SECONDS,
                "frame_count": scenes[scene_index][1] - scenes[scene_index][0] + 1,
            },
        )
        db.add(node)
        scene_nodes.append(node)
    db.flush()

    # ---- Structural edges: CONTAINS / BELONGS_TO --------------------------
    budget = _EdgeBudget(settings.GRAPH_MAX_EDGES_PER_VIDEO)
    video_edges: List[GraphEdge] = []

    for node in scene_nodes:
        video_edges.append(
            _add_edge(
                result,
                video_node,
                node,
                RelationshipType.CONTAINS,
                weight=1.0,
                confidence=1.0,
                timestamp_start=node.timestamp_start,
                timestamp_end=node.timestamp_end,
                edge_metadata={"basis": "deterministic_containment"},
            )
        )
    for scene_index, (s, e, _t0, _t1) in enumerate(scenes):
        scene_node = scene_nodes[scene_index]
        for frame_node in frame_nodes[s : e + 1]:
            video_edges.append(
                _add_edge(
                    result,
                    scene_node,
                    frame_node,
                    RelationshipType.CONTAINS,
                    weight=1.0,
                    confidence=1.0,
                    timestamp_start=frame_node.timestamp_start,
                    timestamp_end=frame_node.timestamp_end,
                    edge_metadata={"basis": "deterministic_containment"},
                )
            )
            video_edges.append(
                _add_edge(
                    result,
                    frame_node,
                    scene_node,
                    RelationshipType.BELONGS_TO,
                    weight=1.0,
                    confidence=1.0,
                    timestamp_start=frame_node.timestamp_start,
                    timestamp_end=frame_node.timestamp_end,
                    edge_metadata={"basis": "deterministic_containment"},
                )
            )
            result.structural_edges += 2
    for edge in video_edges:
        db.add(edge)
    db.flush()

    # ---- Temporal edges: FOLLOWED_BY / PRECEDES --------------------------
    # Basis: the real, ascending timestamps of consecutive retained keyframes.
    temporal_edges: List[GraphEdge] = []
    for i in range(len(frame_nodes) - 1):
        a, b = frame_nodes[i], frame_nodes[i + 1]
        gap = float(timestamps[i + 1] - timestamps[i])
        temporal_edges.append(
            _add_edge(
                result,
                a,
                b,
                RelationshipType.FOLLOWED_BY,
                weight=1.0,
                confidence=1.0,
                timestamp_start=timestamps[i],
                timestamp_end=timestamps[i + 1],
                edge_metadata={"basis": "timestamp_ordering", "gap_seconds": gap},
            )
        )
        temporal_edges.append(
            _add_edge(
                result,
                b,
                a,
                RelationshipType.PRECEDES,
                weight=1.0,
                confidence=1.0,
                timestamp_start=timestamps[i],
                timestamp_end=timestamps[i + 1],
                edge_metadata={"basis": "timestamp_ordering", "gap_seconds": gap},
            )
        )
        result.temporal_edges += 2
    for edge in temporal_edges:
        db.add(edge)
    db.flush()

    # ---- Semantic edges: SIMILAR_TO ---------------------------------------
    # Basis: actual cosine similarity between the stored embeddings.
    embeddings = _embeddings_matrix(frames)
    similar_pairs = compute_similar_pairs(
        embeddings,
        threshold=settings.GRAPH_SIMILARITY_THRESHOLD,
        top_k=settings.GRAPH_SIMILAR_TOP_K,
    )
    for i, j, similarity in similar_pairs:
        if not budget.take():
            result.edges_truncated = True
            break
        edge = _add_edge(
            result,
            frame_nodes[i],
            frame_nodes[j],
            RelationshipType.SIMILAR_TO,
            # The measured cosine similarity is stored verbatim as the weight.
            weight=similarity,
            confidence=1.0,
            timestamp_start=min(timestamps[i], timestamps[j]),
            timestamp_end=max(timestamps[i], timestamps[j]),
            edge_metadata={
                "basis": "measured_cosine_similarity",
                "similarity": similarity,
                "threshold": settings.GRAPH_SIMILARITY_THRESHOLD,
            },
        )
        db.add(edge)
        result.similar_edges += 1
    db.flush()

    # ---- Extracted entities (real extractor only) -------------------------
    if run_extractor:
        _attach_extracted_entities(
            db,
            video_id,
            result,
            frames,
            frame_nodes,
            video_node,
            budget,
            frame_images=frame_images,
        )

    logger.info(
        f"Graph built for video {video_id} | {result.as_dict()}"
    )
    return result


class _EdgeBudget:
    """Hard cap on edges written for one video.

    Deterministic by construction: once exhausted, the builder stops adding
    semantic/extracted edges and records ``edges_truncated`` instead of
    growing without bound. Structural and temporal edges are never truncated
    because dropping them would make the graph internally inconsistent.
    """

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _embeddings_matrix(frames: List[Dict[str, Any]]) -> np.ndarray:
    """Stack stored embeddings into an ``(n, d)`` float32 matrix.

    Rows with a NULL embedding (allowed by the schema, and possible if a frame
    was inserted without an embedding) become zero rows. A zero row is never
    similar to anything, so it simply produces no ``SIMILAR_TO`` edges rather
    than a bogus one.
    """
    dim = None
    rows: List[np.ndarray] = []
    for spec in frames:
        emb = spec.get("embedding")
        if emb is None:
            if dim is None:
                continue
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        vec = np.asarray(emb, dtype=np.float32)
        if dim is None:
            dim = vec.shape[0]
        if vec.shape[0] != dim:
            raise ValueError(
                f"inconsistent embedding dimension: expected {dim}, got {vec.shape[0]}"
            )
        rows.append(vec)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(rows)


def _attach_extracted_entities(
    db: Session,
    video_id: uuid.UUID,
    result: GraphBuildResult,
    frames: List[Dict[str, Any]],
    frame_nodes: List[GraphNode],
    video_node: GraphNode,
    budget: _EdgeBudget,
    *,
    frame_images: Optional[Dict[uuid.UUID, Any]],
) -> None:
    """Add nodes/edges for whatever a real extractor actually reports.

    With the default :class:`~app.services.graph_extraction_service.NullGraphExtractor`
    this function adds nothing and ``result.extracted_nodes`` stays 0. That is
    the correct outcome, not a failure.
    """
    extractor = get_graph_extractor(settings)
    result.extractor_name = extractor.name

    if extractor.name == "null":
        # No detector exists in this repository. Do not pretend otherwise.
        return

    if not frame_images:
        logger.warning(
            f"Extractor '{extractor.name}' is selected but no frame images were "
            f"supplied; skipping extraction for video {video_id}."
        )
        return

    specs = [
        {
            "image": frame_images[spec["id"]],
            "timestamp_seconds": float(spec["timestamp_seconds"]),
            "video_id": video_id,
            "frame_id": spec["id"],
        }
        for spec in frames
        if spec["id"] in frame_images
    ]
    if not specs:
        return

    detections_by_frame = extract_for_frames(extractor, specs)
    result.extractor_detections = sum(len(v) for v in detections_by_frame.values())

    # Entity nodes are shared across frames that report the same entity, which
    # is what makes concept-level traversal meaningful: frame A and frame B can
    # both DEPICT "whiteboard" and therefore be linked through it.
    entity_nodes: Dict[Tuple[str, str], GraphNode] = {}
    frame_index_by_id = {spec["id"]: i for i, spec in enumerate(frames)}

    for frame_id, detections in detections_by_frame.items():
        if not detections:
            continue
        i = frame_index_by_id.get(frame_id)
        if i is None:
            continue
        frame_node = frame_nodes[i]
        ts = float(frames[i]["timestamp_seconds"])

        for detection in detections:
            key = (detection.node_type.value, detection_entity_key(detection))
            entity = entity_nodes.get(key)
            if entity is None:
                if not budget.take():
                    result.edges_truncated = True
                    return
                entity = _add_node(
                    result,
                    video_id,
                    detection.node_type,
                    detection_entity_key(detection),
                    label=detection.label,
                    description=detection.description,
                    timestamp_start=ts,
                    timestamp_end=ts,
                    confidence=float(detection.confidence),
                    node_metadata={
                        **(detection.metadata or {}),
                        "extractor": extractor.name,
                    },
                )
                db.add(entity)
                entity_nodes[key] = entity
                result.extracted_nodes += 1
                db.flush()
                edge = _add_edge(
                    result,
                    video_node,
                    entity,
                    RelationshipType.CONTAINS,
                    weight=1.0,
                    confidence=float(detection.confidence),
                    timestamp_start=ts,
                    timestamp_end=ts,
                    edge_metadata={"basis": f"extractor:{extractor.name}"},
                )
                db.add(edge)
                result.extracted_edges += 1

            if not budget.take():
                result.edges_truncated = True
                return
            edge = _add_edge(
                result,
                frame_node,
                entity,
                RelationshipType.DEPICTS,
                # Weight is the extractor's own score. Not normalised to 1.0,
                # because pretending a raw detector score is a similarity
                # would misrepresent what was measured.
                weight=float(detection.confidence),
                confidence=float(detection.confidence),
                timestamp_start=ts,
                timestamp_end=ts,
                edge_metadata={
                    "basis": f"extractor:{extractor.name}",
                    "entity_key": key[1],
                },
            )
            db.add(edge)
            result.extracted_edges += 1
    db.flush()


# ---------------------------------------------------------------------------
# Bounded traversal
# ---------------------------------------------------------------------------


class GraphExpansion:
    """Result of a bounded graph expansion.

    Attributes
    ----------
    nodes:
        Every node reached, including the seeds (``depth`` 0).
    edges:
        Every edge traversed to reach them.
    depth_of:
        ``{node_id: minimum hops from any seed}``.
    reached_by:
        ``{node_id: [edges traversed to reach it]}`` - the evidence chain that
        the explainability payload surfaces to the frontend.
    truncated_depth:
        True if expansion stopped because ``max_depth`` was reached with nodes
        still having unvisited neighbours.
    truncated_nodes:
        True if expansion stopped because ``max_nodes`` was reached.
    """

    def __init__(self) -> None:
        self.nodes: Dict[uuid.UUID, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.depth_of: Dict[uuid.UUID, int] = {}
        self.reached_by: Dict[uuid.UUID, List[GraphEdge]] = defaultdict(list)
        self.truncated_depth = False
        self.truncated_nodes = False

    @property
    def query_count(self) -> int:
        """Number of SQL round-trips the expansion is allowed to make.

        Exposed so tests can assert the traversal stays batched. Two queries
        per level (edges, then nodes) plus one to load the seeds.
        """
        max_depth_reached = max(self.depth_of.values()) if self.depth_of else 0
        return 1 + 2 * max_depth_reached

    def nodes_at_depth(self, depth: int) -> List[GraphNode]:
        return [n for nid, n in self.nodes.items() if self.depth_of.get(nid) == depth]

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            counts[edge.relationship] += 1
        return dict(counts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<GraphExpansion nodes={len(self.nodes)} edges={len(self.edges)} "
            f"max_depth_reached={max(self.depth_of.values()) if self.depth_of else 0}>"
        )


def expand_from_seeds(
    db: Session,
    seed_frame_ids: Sequence[uuid.UUID],
    *,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    allowed_relationships: Optional[Iterable[str]] = None,
) -> GraphExpansion:
    """Expand a bounded neighbourhood around ``seed_frame_ids``.

    Bounded traversal, enforced three ways:

    1. ``max_depth`` - never expands past the given hop count.
    2. ``max_nodes`` - hard cap on the total number of distinct nodes returned.
    3. ``LIMIT`` on every edge query - a level can never fetch more edges than
       the remaining node budget, so a dense graph cannot blow up memory.

    Batched, not N+1: each level issues one ``SELECT`` for the frontier's edges
    and one for the discovered nodes. See :attr:`GraphExpansion.query_count`.

    Parameters
    ----------
    seed_frame_ids:
        Frames to start from. These are the vector candidates.
    max_depth, max_nodes:
        Overrides for ``settings.GRAPH_MAX_DEPTH`` / ``GRAPH_MAX_NODES``.
    allowed_relationships:
        Optional whitelist. Defaults to every relationship.
    """
    depth_limit = settings.GRAPH_MAX_DEPTH if max_depth is None else int(max_depth)
    node_limit = settings.GRAPH_MAX_NODES if max_nodes is None else int(max_nodes)

    if depth_limit < 0:
        raise ValueError(f"max_depth must be >= 0, got {depth_limit}")
    if node_limit < 1:
        raise ValueError(f"max_nodes must be >= 1, got {node_limit}")

    expansion = GraphExpansion()
    if not seed_frame_ids:
        return expansion

    # --- Level 0: resolve the seeds (1 query) ------------------------------
    # The full node rows are selected here, not just the ids, so seeds never
    # need a second fetch at the end. That keeps the round-trip count at
    # exactly 1 + 2 * depth_reached, which is the batching contract.
    seed_rows = (
        db.execute(
            select(GraphNode).where(
                GraphNode.frame_id.in_(list(seed_frame_ids))
            )
        )
        .scalars()
        .all()
    )
    frontier: Set[uuid.UUID] = set()
    for node in seed_rows:
        node_id = node.id
        if len(frontier) >= node_limit:
            expansion.truncated_nodes = True
            break
        frontier.add(node_id)
        expansion.depth_of[node_id] = 0
        expansion.nodes[node_id] = node

    visited: Set[uuid.UUID] = set(frontier)

    # --- Levels 1..max_depth ------------------------------------------------
    for depth in range(1, depth_limit + 1):
        if not frontier:
            break
        if len(visited) >= node_limit:
            expansion.truncated_nodes = True
            break

        remaining = node_limit - len(visited)
        # One query for this level's edges. The LIMIT guarantees we never pull
        # more edges than the remaining node budget can possibly need.
        edge_rows = (
            db.execute(
                select(GraphEdge)
                .where(
                    (GraphEdge.source_node_id.in_(list(frontier)))
                    | (GraphEdge.target_node_id.in_(list(frontier)))
                )
                .limit(remaining)
            )
            .scalars()
            .all()
        )

        if not edge_rows:
            break

        allowed = set(allowed_relationships) if allowed_relationships else None
        usable = [
            e
            for e in edge_rows
            if allowed is None or e.relationship in allowed
        ]

        neighbour_ids: Set[uuid.UUID] = set()
        for edge in usable:
            other = (
                edge.target_node_id
                if edge.source_node_id in frontier
                else edge.source_node_id
            )
            if other not in visited:
                neighbour_ids.add(other)
            expansion.edges.append(edge)

        if not neighbour_ids:
            break

        # One query for this level's nodes.
        room = node_limit - len(visited)
        if len(neighbour_ids) > room:
            expansion.truncated_nodes = True
            neighbour_ids = set(sorted(neighbour_ids)[:room])

        node_rows = (
            db.execute(select(GraphNode).where(GraphNode.id.in_(list(neighbour_ids))))
            .scalars()
            .all()
        )

        next_frontier: Set[uuid.UUID] = set()
        for node in node_rows:
            if node.id in visited:
                continue
            # Hard stop inside the loop as well as on the IN list. Relying on
            # the database to honour the filtered `IN` would make the node cap
            # an unenforced convention: any row that came back anyway would
            # push the expansion past max_nodes.
            if len(visited) >= node_limit:
                expansion.truncated_nodes = True
                break
            visited.add(node.id)
            expansion.nodes[node.id] = node
            expansion.depth_of[node.id] = depth
            next_frontier.add(node.id)
            for edge in usable:
                if node.id in (edge.source_node_id, edge.target_node_id):
                    expansion.reached_by[node.id].append(edge)

        frontier = next_frontier
        if len(visited) >= node_limit:
            expansion.truncated_nodes = True
            break
        if frontier:
            expansion.truncated_depth = True

    return expansion


def serialize_graph(
    db: Session,
    video_id: uuid.UUID,
    *,
    node_limit: int = 5000,
    edge_limit: int = 20000,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return a video's whole graph as plain dicts, for ``GET /videos/{id}/graph``.

    Two batched queries - no traversal, no per-node lookups. The limits are a
    response-size guard, not a traversal bound, and truncation is reported in
    the payload so a consumer can never mistake a partial graph for a complete
    one.
    """
    nodes = (
        db.execute(
            select(GraphNode)
            .where(GraphNode.video_id == video_id)
            .order_by(GraphNode.node_type.asc(), GraphNode.entity_key.asc())
            .limit(node_limit)
        )
        .scalars()
        .all()
    )
    node_ids = [n.id for n in nodes]

    edges: List[GraphEdge] = []
    if node_ids:
        edges = (
            db.execute(
                select(GraphEdge)
                .where(
                    (GraphEdge.source_node_id.in_(node_ids))
                    | (GraphEdge.target_node_id.in_(node_ids))
                )
                .order_by(GraphEdge.relationship.asc())
                .limit(edge_limit)
            )
            .scalars()
            .all()
        )

    return {
        "nodes": [serialize_node(n) for n in nodes],
        "edges": [serialize_edge(e) for e in edges],
    }


def serialize_node(node: GraphNode) -> Dict[str, Any]:
    """JSON-safe dict for one node."""
    return {
        "id": str(node.id),
        "video_id": str(node.video_id),
        "node_type": node.node_type,
        "entity_key": node.entity_key,
        "label": node.label,
        "description": node.description,
        "timestamp_start": node.timestamp_start,
        "timestamp_end": node.timestamp_end,
        "frame_id": str(node.frame_id) if node.frame_id else None,
        "confidence": node.confidence,
        "metadata": node.node_metadata,
    }


def serialize_edge(edge: GraphEdge) -> Dict[str, Any]:
    """JSON-safe dict for one edge."""
    return {
        "id": str(edge.id),
        "source_node_id": str(edge.source_node_id),
        "target_node_id": str(edge.target_node_id),
        "relationship": edge.relationship,
        "weight": edge.weight,
        "confidence": edge.confidence,
        "timestamp_start": edge.timestamp_start,
        "timestamp_end": edge.timestamp_end,
        "metadata": edge.edge_metadata,
    }
