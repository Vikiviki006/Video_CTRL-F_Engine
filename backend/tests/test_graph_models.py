"""Unit tests for the graph schema.

These assert the *structural* guarantees the specification requires: foreign
keys with cascades, the six required indexes, and the CHECK constraints that
keep node types and relationships inside their vocabularies. They read SQLAlchemy
metadata, so they run without a database.

All data referenced here is synthetic and in-memory; nothing is written to a
database.
"""

import uuid

import pytest

from app.models.graph import (
    GraphEdge,
    GraphNode,
    NodeType,
    RelationshipType,
    SEMANTIC_RELATIONSHIPS,
    STRUCTURAL_RELATIONSHIPS,
    TEMPORAL_RELATIONSHIPS,
)


# ---------------------------------------------------------------------------
# Table and column presence
# ---------------------------------------------------------------------------


def test_graph_tables_exist_with_expected_names():
    assert GraphNode.__tablename__ == "graph_nodes"
    assert GraphEdge.__tablename__ == "graph_edges"


def test_node_columns_match_the_specification():
    table = GraphNode.__table__
    # The Table's ColumnCollection is keyed by the *column* name, which is
    # literally `metadata`; the ORM exposes it as `node_metadata`, because
    # `metadata` is reserved on declarative classes. Both names are correct -
    # this asserts each in its own place.
    assert "metadata" in table.c.keys()
    assert "node_metadata" not in table.c.keys()
    assert table.c["metadata"].type.__class__.__name__ == "JSONB"
    assert "node_metadata" in [c.key for c in GraphNode.__mapper__.column_attrs]

    expected = {
        "id",
        "video_id",
        "node_type",
        "entity_key",
        "label",
        "description",
        "timestamp_start",
        "timestamp_end",
        "frame_id",
        "confidence",
        "metadata",
        "created_at",
    }
    assert expected.issubset(set(table.c.keys()))


def test_edge_columns_match_the_specification():
    table = GraphEdge.__table__
    assert "metadata" in table.c.keys()
    assert "edge_metadata" in [c.key for c in GraphEdge.__mapper__.column_attrs]

    expected = {
        "id",
        "source_node_id",
        "target_node_id",
        "relationship",
        "weight",
        "confidence",
        "timestamp_start",
        "timestamp_end",
        "metadata",
        "created_at",
    }
    assert expected.issubset(set(table.c.keys()))


# ---------------------------------------------------------------------------
# Foreign keys and cascades
# ---------------------------------------------------------------------------


def _fk_map(table):
    return {
        tuple(element.parent.name for element in constraint.elements): constraint
        for constraint in table.foreign_key_constraints
    }


def test_node_video_fk_cascades_on_delete():
    # Deleting a video must remove its nodes, which is the root of the
    # "no orphaned edges, ever" guarantee.
    constraints = _fk_map(GraphNode.__table__)
    key = ("video_id",)
    assert key in constraints
    assert constraints[key].ondelete == "CASCADE"
    assert {e.target_fullname for e in constraints[key].elements} == {"videos.id"}


def test_node_frame_fk_cascades_on_delete():
    constraints = _fk_map(GraphNode.__table__)
    key = ("frame_id",)
    assert key in constraints
    assert constraints[key].ondelete == "CASCADE"
    assert {e.target_fullname for e in constraints[key].elements} == {"video_frames.id"}


@pytest.mark.parametrize("column", ["source_node_id", "target_node_id"])
def test_edge_node_fks_cascade_on_delete(column):
    constraints = _fk_map(GraphEdge.__table__)
    key = (column,)
    assert key in constraints
    assert constraints[key].ondelete == "CASCADE"
    assert {e.target_fullname for e in constraints[key].elements} == {"graph_nodes.id"}


# ---------------------------------------------------------------------------
# Required indexes
# ---------------------------------------------------------------------------


def _index_names(table):
    names = {idx.name for idx in table.indexes}
    names |= {
        c.name
        for c in table.columns
        if c.index is True or (c.index is not None and c.index is not False)
    }
    return names


def test_required_node_indexes_present():
    names = _index_names(GraphNode.__table__)
    assert "ix_graph_nodes_video_id" in names
    assert "ix_graph_nodes_node_type" in names
    assert "ix_graph_nodes_frame_id" in names
    # The composite that serves the dominant access pattern.
    assert "idx_graph_nodes_video_type" in names


def test_required_edge_indexes_present():
    names = _index_names(GraphEdge.__table__)
    assert "ix_graph_edges_source_node_id" in names
    assert "ix_graph_edges_target_node_id" in names
    assert "ix_graph_edges_relationship" in names


# ---------------------------------------------------------------------------
# Check constraints and vocabularies
# ---------------------------------------------------------------------------


def _check_names(table):
    return {c.name for c in table.constraints if c.__class__.__name__ == "CheckConstraint"}


def test_node_type_check_constraint_exists():
    assert "ck_graph_nodes_node_type" in _check_names(GraphNode.__table__)


def test_relationship_check_constraint_exists():
    assert "ck_graph_edges_relationship" in _check_names(GraphEdge.__table__)


def test_no_self_loop_constraint_exists():
    assert "ck_graph_edges_no_self_loop" in _check_names(GraphEdge.__table__)


def test_confidence_and_weight_are_bounded():
    node_checks = _check_names(GraphNode.__table__)
    edge_checks = _check_names(GraphEdge.__table__)
    assert "ck_graph_nodes_confidence" in node_checks
    assert "ck_graph_edges_confidence" in edge_checks
    assert "ck_graph_edges_weight" in edge_checks


def test_unique_constraints_make_construction_idempotent():
    node_uniques = {
        tuple(c.name for c in u.columns) for u in GraphNode.__table__.constraints
        if u.__class__.__name__ == "UniqueConstraint"
    }
    edge_uniques = {
        tuple(c.name for c in u.columns) for u in GraphEdge.__table__.constraints
        if u.__class__.__name__ == "UniqueConstraint"
    }
    assert ("video_id", "node_type", "entity_key") in node_uniques
    assert ("source_node_id", "target_node_id", "relationship") in edge_uniques


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


def test_node_type_vocabulary_matches_the_specified_minimal_set():
    values = {m.value for m in NodeType}
    assert values == {"VIDEO", "SCENE", "FRAME", "OBJECT", "TEXT", "CONCEPT"}


def test_person_event_action_location_are_absent_until_a_real_extractor_exists():
    # The specification forbids adding these until a real extractor emits them.
    values = {m.value for m in NodeType}
    for forbidden in ("PERSON", "EVENT", "ACTION", "LOCATION"):
        assert forbidden not in values


def test_relationship_vocabulary_matches_the_specification():
    values = {m.value for m in RelationshipType}
    assert values == {
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
    }


def test_relationship_families_are_disjoint_and_cover_the_enums():
    every = {m.value for m in RelationshipType}
    combined = SEMANTIC_RELATIONSHIPS | TEMPORAL_RELATIONSHIPS | STRUCTURAL_RELATIONSHIPS
    # DEPICTS/SUPPORTS/ASSOCIATED_WITH are extractor-only and deliberately not
    # classified into the three scoring families, so `combined` is a subset.
    assert combined.issubset(every)
    assert not (SEMANTIC_RELATIONSHIPS & TEMPORAL_RELATIONSHIPS)
    assert not (SEMANTIC_RELATIONSHIPS & STRUCTURAL_RELATIONSHIPS)
    assert not (TEMPORAL_RELATIONSHIPS & STRUCTURAL_RELATIONSHIPS)


# ---------------------------------------------------------------------------
# Instantiation (synthetic, never persisted)
# ---------------------------------------------------------------------------


def test_node_accepts_valid_type_and_bounds_confidence():
    node = GraphNode(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        node_type=NodeType.FRAME.value,
        entity_key="frame:synthetic",
        confidence=0.5,
    )
    assert node.node_type == "FRAME"
    assert node.node_metadata is None  # JSONB column, unset


def test_node_metadata_attribute_maps_to_the_metadata_column():
    node = GraphNode(node_type="FRAME", entity_key="k", node_metadata={"a": 1})
    # The ORM attribute is `node_metadata`...
    assert node.node_metadata == {"a": 1}
    # ...while the column it writes to is literally named `metadata` on disk.
    assert GraphNode.__table__.c["metadata"] is GraphNode.__mapper__.column_attrs[
        "node_metadata"
    ].columns[0]
