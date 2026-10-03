"""Relational knowledge-graph layer for VideoCtrl-F (Phase 1).

Two tables, both living in the *same* PostgreSQL database that already holds
pgvector. No second database, no Neo4j, no graph engine: the graph is a plain
relational projection that participates in the existing transaction and
deletion semantics.

Design decisions and why
------------------------
1. ``node_type`` / ``relationship`` are ``String`` columns guarded by CHECK
   constraints, *not* native PostgreSQL ENUM types. This deliberately mirrors
   the existing ``videos.status VARCHAR(32)`` + ``VideoStatus`` Pydantic enum
   style, and it keeps the vocabulary extensible: adding a relationship later
   is an ordinary constraint migration rather than an ``ALTER TYPE ... ADD
   VALUE`` dance. The authoritative Python-side vocabularies are
   :class:`NodeType` and :class:`RelationshipType` below.

2. The Python attribute for the JSONB column is ``node_metadata`` / 
   ``edge_metadata``, mapped onto a database column literally named
   ``metadata``. ``metadata`` is a reserved attribute name on SQLAlchemy
   declarative classes (it would shadow ``Base.metadata``), so the attribute
   and the column name intentionally differ. On-disk schema still matches the
   specification: ``metadata JSONB``.

3. ``graph_nodes.video_id`` and ``graph_nodes.frame_id`` are real foreign keys
   with ``ON DELETE CASCADE``. Deleting a video therefore removes its frames,
   its nodes, and (transitively) every edge touching those nodes. Orphaned
   edges are structurally impossible, not merely unlikely.

4. ``UNIQUE (video_id, node_type, entity_key)`` makes node creation
   idempotent. Re-running graph construction for a video updates rather than
   duplicates, which is what makes the transactional rebuild in
   ``graph_service`` safe to retry.

5. Every edge must have a traceable basis. ``weight`` and ``confidence`` are
   populated from measured data only (actual cosine similarity, actual
   timestamp ordering) or from a real extractor's own output. No production
   code path invents an edge value. See ``graph_extraction_service`` for the
   documented no-op extractor used when no detector exists.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.db.session import Base


class NodeType(str, Enum):
    """Node vocabulary.

    Only ``VIDEO``, ``SCENE`` and ``FRAME`` are written by the shipped
    extraction pipeline, because those are the only node kinds derivable from
    data this repository actually produces (timestamps, retained keyframes and
    their measured embeddings).

    ``OBJECT``, ``TEXT`` and ``CONCEPT`` are declared because the schema must
    be able to hold them, but they are written *only* by a real extractor
    plugged into ``graph_extraction_service.GraphExtractor``. With the default
    ``NullGraphExtractor`` in place they are never written and the tables
    contain zero rows of those types.

    ``PERSON``, ``EVENT``, ``ACTION`` and ``LOCATION`` are deliberately absent.
    Per the specification they may only be added once a real extractor emits
    them; adding unused enum members now would misrepresent the system's
    capability.
    """

    VIDEO = "VIDEO"
    SCENE = "SCENE"
    FRAME = "FRAME"
    OBJECT = "OBJECT"
    TEXT = "TEXT"
    CONCEPT = "CONCEPT"


class RelationshipType(str, Enum):
    """Edge vocabulary.

    ``CONTAINS``   video -> scene / scene -> frame   (containment, deterministic)
    ``BELONGS_TO`` frame -> scene                    (inverse of CONTAINS)
    ``DEPICTS``    frame -> object/text/concept      (requires a real extractor)
    ``FOLLOWED_BY``frame -> frame, next in time      (deterministic ordering)
    ``PRECEDES``   frame -> frame, previous in time  (inverse of FOLLOWED_BY)
    ``SIMILAR_TO`` frame <-> frame                   (measured cosine similarity)
    ``RELATED_TO`` generic soft link                (requires a real extractor)
    ``SUPPORTS``   evidence -> claim                 (requires a real extractor)
    ``ASSOCIATED_WITH`` concept <-> concept          (requires a real extractor)
    ``OCCURS_NEAR``any -> any, within a time window  (deterministic timestamps)
    """

    CONTAINS = "CONTAINS"
    DEPICTS = "DEPICTS"
    BELONGS_TO = "BELONGS_TO"
    FOLLOWED_BY = "FOLLOWED_BY"
    PRECEDES = "PRECEDES"
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    OCCURS_NEAR = "OCCURS_NEAR"


# Allowed values, derived from the enums so the CHECK constraints and the
# Python vocabularies can never drift apart.
_NODE_TYPES = ", ".join(f"'{m.value}'" for m in NodeType)
_RELATIONSHIPS = ", ".join(f"'{m.value}'" for m in RelationshipType)


class GraphNode(Base):
    """A node in the per-video relational graph.

    One row is one addressable entity. ``entity_key`` is the stable identity of
    the node *within* its video, e.g. ``"frame:<frame_uuid>"`` or
    ``"scene:7"``. Callers reach nodes by ``(video_id, node_type,
    entity_key)``; nothing depends on the surrogate ``id`` except edges.
    """

    __tablename__ = "graph_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Owning video. ON DELETE CASCADE is the root of the "no orphans, ever"
    # guarantee: deleting a video cascades to its nodes, and each deleted node
    # cascades to every edge that references it.
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    node_type = Column(String(32), nullable=False, index=True)
    entity_key = Column(String(512), nullable=False)
    label = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)

    # Temporal extent. For a single frame, start == end == its timestamp.
    timestamp_start = Column(Float, nullable=True)
    timestamp_end = Column(Float, nullable=True)

    # Set when this node corresponds to a concrete retained keyframe. Real FK
    # with cascade, so frames removed from video_frames cannot leave nodes
    # pointing at nothing.
    frame_id = Column(
        UUID(as_uuid=True),
        ForeignKey("video_frames.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # How much trust the producing stage placed in this node, in [0, 1].
    # Nodes built deterministically from timestamps/embeddings use 1.0.
    # Nodes produced by a real extractor carry that extractor's own score.
    confidence = Column(Float, nullable=False, default=1.0)

    # Column is literally named `metadata`; the Python attribute is
    # `node_metadata` because `metadata` is reserved by SQLAlchemy declarative.
    node_metadata = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        # Makes node creation idempotent and gives graph construction a stable
        # natural key to address nodes by.
        UniqueConstraint(
            "video_id", "node_type", "entity_key", name="uq_graph_nodes_identity"
        ),
        CheckConstraint(
            f"node_type IN ({_NODE_TYPES})", name="ck_graph_nodes_node_type"
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_graph_nodes_confidence",
        ),
        CheckConstraint(
            "timestamp_start IS NULL OR timestamp_end IS NULL "
            "OR timestamp_end >= timestamp_start",
            name="ck_graph_nodes_time_order",
        ),
        Index("idx_graph_nodes_video_type", "video_id", "node_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<GraphNode {self.node_type}:{self.entity_key} "
            f"video={self.video_id}>"
        )


class GraphEdge(Base):
    """A directed, weighted, timestamped relationship between two nodes.

    ``weight`` is the measured strength of the relationship (cosine similarity
    for ``SIMILAR_TO``); ``confidence`` is how much the producing stage trusts
    the edge at all. Keeping them separate lets a strong-but-uncertain edge
    (``weight=0.95, confidence=0.5``) rank below a certain-but-moderate one.
    """

    __tablename__ = "graph_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey("graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    relationship = Column(String(32), nullable=False, index=True)
    weight = Column(Float, nullable=False, default=1.0)
    confidence = Column(Float, nullable=False, default=1.0)

    timestamp_start = Column(Float, nullable=True)
    timestamp_end = Column(Float, nullable=True)

    edge_metadata = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            f"relationship IN ({_RELATIONSHIPS})",
            name="ck_graph_edges_relationship",
        ),
        # An edge must connect two distinct nodes; self-loops carry no
        # information and make traversal termination harder to reason about.
        CheckConstraint(
            "source_node_id <> target_node_id", name="ck_graph_edges_no_self_loop"
        ),
        CheckConstraint(
            "weight IS NULL OR (weight >= 0.0 AND weight <= 1.0)",
            name="ck_graph_edges_weight",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_graph_edges_confidence",
        ),
        CheckConstraint(
            "timestamp_start IS NULL OR timestamp_end IS NULL "
            "OR timestamp_end >= timestamp_start",
            name="ck_graph_edges_time_order",
        ),
        # Prevents the same relationship being written twice between the same
        # ordered pair. Deterministic rebuilds then converge instead of
        # accumulating duplicates.
        UniqueConstraint(
            "source_node_id",
            "target_node_id",
            "relationship",
            name="uq_graph_edges_relationship_once",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<GraphEdge {self.relationship} "
            f"{self.source_node_id}->{self.target_node_id} w={self.weight}>"
        )


# Relationships that express *semantic* similarity rather than structure or
# time. Used by the ranking service to separate "this frame looks like the
# query" from "this frame is adjacent in time to the query".
SEMANTIC_RELATIONSHIPS: frozenset = frozenset(
    {RelationshipType.SIMILAR_TO.value, RelationshipType.RELATED_TO.value}
)

# Relationships that express *ordering in time*.
TEMPORAL_RELATIONSHIPS: frozenset = frozenset(
    {
        RelationshipType.FOLLOWED_BY.value,
        RelationshipType.PRECEDES.value,
        RelationshipType.OCCURS_NEAR.value,
    }
)

# Relationships that express *structural containment*.
STRUCTURAL_RELATIONSHIPS: frozenset = frozenset(
    {RelationshipType.CONTAINS.value, RelationshipType.BELONGS_TO.value}
)
