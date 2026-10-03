"""Unit tests for graph construction and bounded traversal.

Two properties matter most and are tested directly:

1. **Boundedness.** Traversal must hard-stop at both ``max_depth`` and
   ``max_nodes``. A test that only checks a small graph would pass even for an
   unbounded implementation, so these tests use graphs deliberately larger than
   the caps.

2. **Batching, not N+1.** The number of SQL round-trips must be
   ``1 + 2 * depth_reached`` regardless of graph size. This is asserted by
   counting queries against an instrumented session.

All graph data below is **synthetic**, built in memory by the helpers in
``conftest.py``, and is never written to a database. It exists solely to
exercise the algorithms.
"""

import uuid

import numpy as np
import pytest

from app.models.graph import NodeType, RelationshipType
from app.services.graph_service import (
    GraphExpansion,
    compute_similar_pairs,
    detection_entity_key,
    expand_from_seeds,
    frame_entity_key,
    group_scenes,
    scene_entity_key,
    video_entity_key,
)
from tests.conftest import ScriptedSession, make_edge, make_node


# ---------------------------------------------------------------------------
# Entity keys
# ---------------------------------------------------------------------------


def test_entity_keys_are_stable_and_prefixed():
    frame_id = uuid.uuid4()
    assert video_entity_key() == "video"
    assert scene_entity_key(3) == "scene:3"
    assert frame_entity_key(frame_id) == f"frame:{frame_id}"
    # Stability: the same input always yields the same key.
    assert frame_entity_key(frame_id) == frame_entity_key(frame_id)


def test_detection_entity_key_normalises_label():
    from app.services.graph_extraction_service import Detection

    det = Detection(node_type=NodeType.CONCEPT, label="  White   Board ")
    assert detection_entity_key(det) == "concept:white_board"
    det2 = Detection(node_type=NodeType.CONCEPT, label="White   Board")
    assert detection_entity_key(det) == detection_entity_key(det2)


# ---------------------------------------------------------------------------
# Scene grouping
# ---------------------------------------------------------------------------


def test_group_scenes_splits_on_the_configured_window():
    # Gaps: 1.0 (keep), 5.0 (exceeds 3.0 window -> new scene), 0.5 (keep)
    timestamps = [0.0, 1.0, 6.0, 6.5]
    scenes = group_scenes(timestamps, window_seconds=3.0)
    assert scenes == [(0, 1, 0.0, 1.0), (2, 3, 6.0, 6.5)]


def test_group_scenes_keeps_everything_in_one_scene_when_contiguous():
    timestamps = [0.0, 1.0, 2.0, 3.0]
    scenes = group_scenes(timestamps, window_seconds=3.0)
    assert len(scenes) == 1
    assert scenes[0] == (0, 3, 0.0, 3.0)


def test_group_scenes_handles_empty_and_single_inputs():
    assert group_scenes([], window_seconds=3.0) == []
    assert group_scenes([4.2], window_seconds=3.0) == [(0, 0, 4.2, 4.2)]


def test_group_scenes_rejects_non_positive_window():
    with pytest.raises(ValueError, match="must be positive"):
        group_scenes([0.0, 1.0], window_seconds=0.0)


# ---------------------------------------------------------------------------
# Similar-pair computation (real cosine similarity, thresholded)
# ---------------------------------------------------------------------------


def test_similarity_pairs_only_include_pairs_above_the_threshold():
    # Three unit-ish vectors; e1 is nearly identical to e2, e3 is orthogonal.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],  # cos ~= 0.99 with e1
            [0.0, 0.0, 1.0],  # orthogonal to both
        ],
        dtype=np.float32,
    )
    pairs = compute_similar_pairs(embeddings, threshold=0.80, top_k=5)
    assert len(pairs) == 1
    i, j, similarity = pairs[0]
    assert (i, j) == (0, 1)
    assert similarity == pytest.approx(0.990, abs=1e-3)


def test_similarity_pairs_emit_each_unordered_pair_once():
    # Symmetric similarity must not produce both (0,1) and (1,0).
    embeddings = np.array([[1.0, 0.1], [1.0, 0.1], [0.0, 1.0]], dtype=np.float32)
    pairs = compute_similar_pairs(embeddings, threshold=0.5, top_k=5)
    unordered = {frozenset((i, j)) for i, j, _ in pairs}
    for i, j, _ in pairs:
        assert i < j, "pairs must be emitted with i < j"
    assert len(unordered) == len(pairs)


def test_similarity_pairs_respect_top_k():
    # Four mutually-similar vectors, top_k=1 -> at most 3 edges.
    embeddings = np.eye(4, dtype=np.float32) + 1.0
    pairs = compute_similar_pairs(embeddings, threshold=0.0, top_k=1)
    assert len(pairs) <= 3


def test_similarity_pairs_returns_nothing_for_a_single_frame():
    assert compute_similar_pairs(np.zeros((1, 3), dtype=np.float32), 0.8, 5) == []


def test_similarity_pairs_reject_invalid_arguments():
    emb = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        compute_similar_pairs(emb, threshold=1.5, top_k=5)
    with pytest.raises(ValueError, match="top_k"):
        compute_similar_pairs(emb, threshold=0.5, top_k=0)


# ---------------------------------------------------------------------------
# Bounded traversal
# ---------------------------------------------------------------------------


def _seed_rows(seed_nodes):
    """Level-0 query now selects full node rows, so seeds need no second fetch."""
    return list(seed_nodes)


def test_expand_returns_empty_without_seeds_and_makes_no_query():
    session = ScriptedSession([])
    expansion = expand_from_seeds(session, [], max_depth=2, max_nodes=10)
    assert expansion.nodes == {}
    assert session.execute_calls == 0


def test_expand_resolves_seeds_in_a_single_query():
    """Seeds come back as full rows, so no second fetch is needed for them.

    With depth 1 the traversal issues exactly two queries: resolve seeds, then
    find the frontier's edges. A third "reload the seed nodes" query would mean
    the batching contract is broken.
    """
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    session = ScriptedSession([_seed_rows([seed]), []])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=10)

    assert session.execute_calls == 2
    assert session.unused_responses == 0
    assert expansion.depth_of[seed.id] == 0
    assert expansion.nodes[seed.id] is seed


def test_expand_stops_at_max_depth():
    """A 3-level chain with max_depth=1 must not reach level 2."""
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    level1 = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    level2 = make_node(node_type="FRAME", frame_id=uuid.uuid4())

    # Level 1 edges: seed -> level1
    l1_edges = [make_edge(seed, level1, RelationshipType.SIMILAR_TO.value, weight=0.9)]
    # Level 2 edges would be returned only if the traversal kept going
    l2_edges = [make_edge(level1, level2, RelationshipType.SIMILAR_TO.value, weight=0.9)]

    session = ScriptedSession(
        [
            _seed_rows([seed]),  # 1. resolve seeds
            l1_edges,  # 2. level-1 edges
            [level1],  # 3. level-1 nodes
        ]
    )
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=100)

    assert level1.id in expansion.nodes
    assert level2.id not in expansion.nodes
    assert expansion.depth_of[level1.id] == 1
    # 3 queries: seeds, edges, nodes. Nothing beyond depth 1 was fetched.
    assert session.execute_calls == 3


def test_expand_is_batched_not_n_plus_one():
    """Query count must depend on depth, never on how many nodes are reached."""
    # 20 nodes all connected to the single seed, but the traversal is allowed
    # at most 8 nodes, so it must stop early rather than fetch all 20.
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    neighbours = [make_node(node_type="FRAME", frame_id=uuid.uuid4()) for _ in range(20)]
    edges = [make_edge(seed, n, RelationshipType.SIMILAR_TO.value, weight=0.85) for n in neighbours]

    session = ScriptedSession([_seed_rows([seed]), edges[:8], neighbours[:8]])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=9)

    # seed + at most 8 neighbours = 9 nodes, which is the cap.
    assert len(expansion.nodes) <= 9
    assert expansion.truncated_nodes is True
    # 3 queries total regardless of the 20 available neighbours.
    assert session.execute_calls == 3


def test_expand_hard_caps_max_nodes():
    """The node cap is enforced in code, not delegated to the database.

    The scripted session deliberately returns *more* rows than the cap allows,
    simulating a backend that does not honour the filtered ``IN`` list. The
    traversal must still stop at ``max_nodes``.
    """
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    neighbours = [make_node(node_type="FRAME", frame_id=uuid.uuid4()) for _ in range(5)]
    edges = [make_edge(seed, n, RelationshipType.SIMILAR_TO.value, weight=0.9) for n in neighbours]

    session = ScriptedSession([_seed_rows([seed]), edges, neighbours])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=3)

    # 1 seed + at most 2 neighbours = the cap of 3, even though 5 came back.
    assert len(expansion.nodes) == 3
    assert expansion.truncated_nodes is True


def test_expand_respects_max_depth_zero():
    """Depth 0 must not expand at all - it is the vector-only control."""
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    neighbour = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    session = ScriptedSession([_seed_rows([seed])])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=0, max_nodes=50)

    assert list(expansion.nodes) == [seed.id]
    assert session.execute_calls == 1
    assert not expansion.edges


def test_expand_records_reached_by_evidence_chain():
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    neighbour = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    edge = make_edge(seed, neighbour, RelationshipType.SIMILAR_TO.value, weight=0.91)

    session = ScriptedSession([_seed_rows([seed]), [edge], [neighbour]])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=50)

    # The evidence chain is what the explainability payload surfaces.
    assert expansion.reached_by[neighbour.id] == [edge]
    assert edge.relationship == "SIMILAR_TO"


def test_expand_filters_by_allowed_relationship():
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    wanted = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    unwanted = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    edge_wanted = make_edge(seed, wanted, RelationshipType.SIMILAR_TO.value, weight=0.95)
    edge_unwanted = make_edge(seed, unwanted, RelationshipType.FOLLOWED_BY.value, weight=1.0)

    session = ScriptedSession(
        [_seed_rows([seed]), [edge_wanted, edge_unwanted], [wanted]]
    )
    expansion = expand_from_seeds(
        session,
        [seed.frame_id],
        max_depth=1,
        max_nodes=50,
        allowed_relationships={RelationshipType.SIMILAR_TO.value},
    )
    assert wanted.id in expansion.nodes
    assert unwanted.id not in expansion.nodes
    # The filtered-out edge was never traversed.
    assert [e.relationship for e in expansion.edges] == ["SIMILAR_TO"]


def test_expand_edge_type_counts_are_measured_from_traversed_edges():
    seed = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    a = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    b = make_node(node_type="FRAME", frame_id=uuid.uuid4())
    edges = [
        make_edge(seed, a, RelationshipType.SIMILAR_TO.value, weight=0.9),
        make_edge(seed, b, RelationshipType.FOLLOWED_BY.value, weight=1.0),
    ]
    session = ScriptedSession([_seed_rows([seed]), edges, [a, b]])
    expansion = expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=50)
    counts = expansion.edge_type_counts()
    assert counts.get("SIMILAR_TO") == 1
    assert counts.get("FOLLOWED_BY") == 1


def test_expand_rejects_invalid_bounds():
    seed = make_node(frame_id=uuid.uuid4())
    session = ScriptedSession([])
    with pytest.raises(ValueError, match="max_depth"):
        expand_from_seeds(session, [seed.frame_id], max_depth=-1)
    with pytest.raises(ValueError, match="max_nodes"):
        expand_from_seeds(session, [seed.frame_id], max_depth=1, max_nodes=0)


def test_expansion_query_count_predicts_round_trips():
    """``query_count`` is the documented bound: 1 seed query + 2 per level."""
    expansion = GraphExpansion()
    node = make_node()
    expansion.depth_of[node.id] = 0
    assert expansion.query_count == 1
    expansion.depth_of[make_node().id] = 2
    assert expansion.query_count == 1 + 2 * 2


def test_scripted_session_detects_extra_queries():
    """A traversal that asks for more than the batching contract allows fails."""
    seed = make_node(frame_id=uuid.uuid4())
    session = ScriptedSession([_seed_rows([seed])])
    with pytest.raises(AssertionError, match="ran out of responses"):
        expand_from_seeds(session, [seed.frame_id], max_depth=2, max_nodes=50)
