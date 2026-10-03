"""graph layer: graph_nodes + graph_edges

Revision ID: 0001_graph_layer
Revises:
Create Date: 2026-09-27

Scope and safety
----------------
Purely additive. This revision creates two new tables and touches nothing that
already exists:

* ``videos``                - untouched
* ``video_frames``          - untouched
* ``init.sql``              - untouched, still bootstraps a fresh Docker volume

``downgrade()`` drops only the two graph tables, and drops ``graph_edges``
first so the foreign keys never block. Because ``ON DELETE CASCADE`` links
``graph_edges -> graph_nodes -> videos``, dropping ``graph_nodes`` alone would
also work, but doing it in the right order keeps the intent obvious.

Deleting a video cascades to its nodes and from there to every edge touching
those nodes, so orphaned edges are structurally impossible rather than merely
avoided by application code.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0001_graph_layer"
down_revision = None
branch_labels = None
depends_on = None


# Kept as module constants so the CHECK constraints and the Python enums in
# app/models/graph.py are generated from the same vocabulary.
NODE_TYPES = ("VIDEO", "SCENE", "FRAME", "OBJECT", "TEXT", "CONCEPT")
RELATIONSHIPS = (
    "CONTAINS",
    "DEPICTS",
    "BELONGS_TO",
    "FOLLOWED_BY",
    "PRECEDES",
    "SIMILAR_TO",
    "RELATED_TO",
    "SUPPORTS",
    "ASSOCIATED_WITH",
    "OCCURS_NEAR",
)


def _in_list(values) -> str:
    return ", ".join("'%s'" % v for v in values)


def upgrade() -> None:
    op.create_table(
        "graph_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("video_id", UUID(as_uuid=True), nullable=False),
        sa.Column("node_type", sa.String(length=32), nullable=False),
        sa.Column("entity_key", sa.String(length=512), nullable=False),
        sa.Column("label", sa.String(length=512), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("timestamp_start", sa.Float(), nullable=True),
        sa.Column("timestamp_end", sa.Float(), nullable=True),
        sa.Column("frame_id", UUID(as_uuid=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        # Column is literally `metadata`; the ORM attribute is `node_metadata`
        # because `metadata` is reserved on SQLAlchemy declarative classes.
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.id"],
            name="fk_graph_nodes_video_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["frame_id"],
            ["video_frames.id"],
            name="fk_graph_nodes_frame_id",
            ondelete="CASCADE",
        ),
        # Makes node creation idempotent: rebuilding a video's graph converges
        # on the same rows instead of duplicating them.
        sa.UniqueConstraint(
            "video_id", "node_type", "entity_key", name="uq_graph_nodes_identity"
        ),
        sa.CheckConstraint(
            "node_type IN (%s)" % _in_list(NODE_TYPES),
            name="ck_graph_nodes_node_type",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_graph_nodes_confidence",
        ),
        sa.CheckConstraint(
            "timestamp_start IS NULL OR timestamp_end IS NULL "
            "OR timestamp_end >= timestamp_start",
            name="ck_graph_nodes_time_order",
        ),
    )

    op.create_table(
        "graph_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("source_node_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_node_id", UUID(as_uuid=True), nullable=False),
        sa.Column("relationship", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("timestamp_start", sa.Float(), nullable=True),
        sa.Column("timestamp_end", sa.Float(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["source_node_id"],
            ["graph_nodes.id"],
            name="fk_graph_edges_source_node_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id"],
            ["graph_nodes.id"],
            name="fk_graph_edges_target_node_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "relationship IN (%s)" % _in_list(RELATIONSHIPS),
            name="ck_graph_edges_relationship",
        ),
        sa.CheckConstraint(
            "source_node_id <> target_node_id", name="ck_graph_edges_no_self_loop"
        ),
        sa.CheckConstraint(
            "weight IS NULL OR (weight >= 0.0 AND weight <= 1.0)",
            name="ck_graph_edges_weight",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_graph_edges_confidence",
        ),
        sa.CheckConstraint(
            "timestamp_start IS NULL OR timestamp_end IS NULL "
            "OR timestamp_end >= timestamp_start",
            name="ck_graph_edges_time_order",
        ),
        # One edge of each relationship type per ordered node pair, so repeated
        # deterministic rebuilds converge instead of accumulating duplicates.
        sa.UniqueConstraint(
            "source_node_id",
            "target_node_id",
            "relationship",
            name="uq_graph_edges_relationship_once",
        ),
    )

    # --- indexes ---------------------------------------------------------
    op.create_index("ix_graph_nodes_video_id", "graph_nodes", ["video_id"])
    op.create_index("ix_graph_nodes_node_type", "graph_nodes", ["node_type"])
    op.create_index("ix_graph_nodes_frame_id", "graph_nodes", ["frame_id"])
    # Composite covering the dominant access pattern: "all nodes of this type
    # for this video", which is both graph serialisation and the video-graph
    # endpoint's filter.
    op.create_index(
        "idx_graph_nodes_video_type", "graph_nodes", ["video_id", "node_type"]
    )

    op.create_index(
        "ix_graph_edges_source_node_id", "graph_edges", ["source_node_id"]
    )
    op.create_index(
        "ix_graph_edges_target_node_id", "graph_edges", ["target_node_id"]
    )
    op.create_index("ix_graph_edges_relationship", "graph_edges", ["relationship"])


def downgrade() -> None:
    # Edges first: explicit, and it keeps the dependency order obvious even
    # though the CASCADE would handle it.
    op.drop_index("ix_graph_edges_relationship", table_name="graph_edges")
    op.drop_index("ix_graph_edges_target_node_id", table_name="graph_edges")
    op.drop_index("ix_graph_edges_source_node_id", table_name="graph_edges")
    op.drop_table("graph_edges")

    op.drop_index("idx_graph_nodes_video_type", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_frame_id", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_node_type", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_video_id", table_name="graph_nodes")
    op.drop_table("graph_nodes")
