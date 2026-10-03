"""Unit tests for graph scoring, ranking and tie-breaking.

The scoring formula under test is the one documented at the top of
``app/services/graph_ranking_service.py``:

.. code-block:: text

    final_score = (a*vector + b*graph + c*temporal + d*confidence) / (a+b+c+d)

All graph data here is **synthetic**, assembled in memory, and never written
to a database. Expected values are computed by hand in the comments.
"""

import uuid

import pytest

from app.services.graph_ranking_service import (
    RankingConfig,
    ablation_configs,
    clip01,
    combine_scores,
    confidence_score,
    direct_similarity,
    edge_confidence_score,
    graph_score,
    rank_candidates,
    scene_overlap,
    temporal_adjacency,
    temporal_score,
    vector_score,
)
from app.services.graph_service import GraphExpansion
from tests.conftest import make_edge, make_node


# ---------------------------------------------------------------------------
# clip01 and the individual score terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.5, 0.5),
        (-0.2, 0.0),  # a negative cosine contributes nothing, never a penalty
        (1.5, 1.0),
        (0.0, 0.0),
        (float("nan"), 0.0),
        (None, 0.0),
    ],
)
def test_clip01_bounds_and_nan_handling(raw, expected):
    assert clip01(raw) == expected


def test_vector_score_is_clipped_cosine_similarity():
    # cosine similarity from pgvector is 1 - distance, already in [-1, 1]
    assert vector_score(0.87) == pytest.approx(0.87)
    assert vector_score(-0.1) == 0.0


# ---------------------------------------------------------------------------
# GraphScore sub-terms
# ---------------------------------------------------------------------------


def test_direct_similarity_uses_the_stored_edge_weight():
    seed = make_node()
    other = make_node()
    edge = make_edge(seed, other, "SIMILAR_TO", weight=0.93)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    # The weight is the measured cosine similarity, used verbatim.
    assert direct_similarity(other.id, {seed.id}, pairs) == pytest.approx(0.93)


def test_direct_similarity_is_direction_agnostic():
    seed = make_node()
    other = make_node()
    edge = make_edge(other, seed, "SIMILAR_TO", weight=0.81)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    assert direct_similarity(other.id, {seed.id}, pairs) == pytest.approx(0.81)


def test_direct_similarity_is_zero_without_a_link():
    seed, other = make_node(), make_node()
    assert direct_similarity(other.id, {seed.id}, {}) == 0.0


def test_scene_overlap_is_the_fraction_of_seed_scenes_shared():
    seed = make_node()
    in_scene = make_node()
    out_scene = make_node()
    scene_a, scene_b = make_node(node_type="SCENE"), make_node(node_type="SCENE")
    scenes = {seed.id: {scene_a.id}, in_scene.id: {scene_a.id}, out_scene.id: {scene_b.id}}
    # in_scene shares the seed's only scene -> 1/1 = 1.0
    assert scene_overlap(scenes, in_scene.id, {seed.id}) == pytest.approx(1.0)
    # out_scene shares none -> 0/1 = 0.0
    assert scene_overlap(scenes, out_scene.id, {seed.id}) == 0.0


def test_scene_overlap_is_zero_when_no_seed_has_a_scene():
    seed, other = make_node(), make_node()
    assert scene_overlap({}, other.id, {seed.id}) == 0.0


def test_edge_confidence_is_the_mean_of_traversed_edges():
    a = make_edge(make_node(), make_node(), weight=1.0, confidence=0.6)
    b = make_edge(make_node(), make_node(), weight=1.0, confidence=0.8)
    # mean(0.6, 0.8) = 0.7
    assert edge_confidence_score([a, b]) == pytest.approx(0.7)


def test_edge_confidence_is_zero_for_a_seed_reached_at_depth_zero():
    # Nothing was traversed to reach a seed, so there is no evidence to score.
    assert edge_confidence_score([]) == 0.0


def test_temporal_adjacency_uses_the_reciprocal_gap_function():
    seed = make_node()
    other = make_node()
    edge = make_edge(seed, other, "FOLLOWED_BY", weight=1.0)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    stamps = {seed.id: 10.0, other.id: 12.0}
    # gap 2.0, window 5.0 -> 1 / (1 + 2/5) = 1/1.4 = 0.714285...
    assert temporal_adjacency(other.id, stamps, {seed.id}, pairs, 5.0) == pytest.approx(
        1 / 1.4
    )


def test_temporal_adjacency_is_half_at_exactly_one_window():
    seed, other = make_node(), make_node()
    edge = make_edge(seed, other, "FOLLOWED_BY", weight=1.0)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    stamps = {seed.id: 0.0, other.id: 5.0}
    # gap == window -> 1 / (1 + 1) = 0.5
    assert temporal_adjacency(other.id, stamps, {seed.id}, pairs, 5.0) == pytest.approx(0.5)


def test_temporal_adjacency_ignores_non_temporal_edges():
    """A CONTAINS edge is not evidence of temporal proximity."""
    seed, other = make_node(), make_node()
    edge = make_edge(seed, other, "CONTAINS", weight=1.0)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    stamps = {seed.id: 0.0, other.id: 0.1}
    assert temporal_adjacency(other.id, stamps, {seed.id}, pairs, 5.0) == 0.0


# ---------------------------------------------------------------------------
# TemporalScore
# ---------------------------------------------------------------------------


def test_temporal_score_is_one_for_adjacent_candidates():
    # nearest neighbour 1.0 s away, window 5.0 -> 1 - 1/5 = 0.8
    assert temporal_score(10.0, [11.0, 40.0], 5.0) == pytest.approx(0.8)


def test_temporal_score_is_zero_beyond_the_window():
    assert temporal_score(10.0, [20.0], 5.0) == 0.0


def test_temporal_score_is_zero_for_a_lone_candidate():
    # No other candidate exists, so there is no temporal context to reward.
    assert temporal_score(10.0, [], 5.0) == 0.0


def test_temporal_score_uses_only_the_other_candidates():
    """The caller must exclude self; a self-gap of 0 would score 1.0 always."""
    # neighbours at +20 s only -> 1 - 20/5 is negative -> clipped to 0
    assert temporal_score(10.0, [30.0], 5.0) == 0.0
    # a neighbour at +1 s -> 1 - 1/5 = 0.8
    assert temporal_score(10.0, [11.0, 30.0], 5.0) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------


def test_confidence_score_geometric_mean():
    # geometric mean of 0.5 and 0.5 = 0.5
    assert confidence_score([0.5, 0.5]) == pytest.approx(0.5)
    # geometric mean of 0.25 and 1.0 = sqrt(0.25) = 0.5
    assert confidence_score([0.25, 1.0]) == pytest.approx(0.5)


def test_confidence_score_is_one_for_the_deterministic_graph():
    """With no extractor, every node is built with confidence 1.0.

    That makes the term ranking-neutral in the shipped configuration - a true
    property, asserted here so a future change cannot silently alter it.
    """
    assert confidence_score([1.0, 1.0, 1.0]) == 1.0


def test_confidence_score_is_zero_for_an_unconfident_node():
    assert confidence_score([0.0, 1.0]) == 0.0


def test_confidence_score_is_zero_for_no_input():
    assert confidence_score([]) == 0.0


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


def test_combine_scores_matches_the_documented_formula():
    cfg = RankingConfig(
        vector_weight=0.65, graph_weight=0.20, temporal_weight=0.10, confidence_weight=0.05
    )
    result = combine_scores(vector=1.0, graph=0.5, temporal=0.0, confidence=1.0, config=cfg)
    # (0.65*1.0 + 0.20*0.5 + 0.10*0.0 + 0.05*1.0) / 1.0 = 0.80
    assert result == pytest.approx(0.80)


def test_combine_scores_normalises_by_the_active_weight_sum():
    """Halving every weight must not change the score."""
    cfg = RankingConfig(
        vector_weight=0.325, graph_weight=0.10, temporal_weight=0.05, confidence_weight=0.025
    )
    result = combine_scores(vector=0.8, graph=0.4, temporal=0.2, confidence=1.0, config=cfg)
    # (0.325*0.8 + 0.10*0.4 + 0.05*0.2 + 0.025*1.0) / 0.5
    # = (0.26 + 0.04 + 0.01 + 0.025) / 0.5 = 0.335 / 0.5 = 0.67
    assert result == pytest.approx(0.67)


def test_graph_score_is_the_documented_four_term_convex_combination():
    """Hand-computed: three of four terms are 1.0, the fourth is 0.0.

    The only edge here is SIMILAR_TO, so:
        direct_similarity = 1.00 (edge weight)
        scene_overlap     = 1.00 (shares the seed's scene)
        edge_confidence   = 1.00 (traversed edge confidence)
        temporal_adjacency= 0.00 (SIMILAR_TO is not a *temporal* relationship,
                                   so a zero timestamp gap earns nothing)

    graph_score = (0.40*1.0 + 0.25*1.0 + 0.20*1.0 + 0.15*0.0) / 1.00 = 0.85
    """
    seed = make_node()
    node = make_node()
    scene = make_node(node_type="SCENE")
    edge = make_edge(seed, node, "SIMILAR_TO", weight=1.0, confidence=1.0)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    scenes = {seed.id: {scene.id}, node.id: {scene.id}}
    stamps = {seed.id: 0.0, node.id: 0.0}

    cfg = RankingConfig(
        graph_direct_coeff=0.40,
        graph_scene_coeff=0.25,
        graph_edge_conf_coeff=0.20,
        graph_temporal_coeff=0.15,
    ).validate()

    result = graph_score(node.id, {seed.id}, pairs, [edge], scenes, stamps, cfg, 5.0)
    assert result == pytest.approx(0.85)


def test_graph_score_reaches_one_when_all_four_terms_are_saturated():
    """Same as above but with a real FOLLOWED_BY edge at zero gap."""
    seed = make_node()
    node = make_node()
    scene = make_node(node_type="SCENE")
    sim = make_edge(seed, node, "SIMILAR_TO", weight=1.0, confidence=1.0)
    temp = make_edge(seed, node, "FOLLOWED_BY", weight=1.0, confidence=1.0)
    pairs = {
        (sim.source_node_id, sim.target_node_id): sim,
        (temp.source_node_id, temp.target_node_id): temp,
    }
    scenes = {seed.id: {scene.id}, node.id: {scene.id}}
    stamps = {seed.id: 0.0, node.id: 0.0}

    cfg = RankingConfig(
        graph_direct_coeff=0.40,
        graph_scene_coeff=0.25,
        graph_edge_conf_coeff=0.20,
        graph_temporal_coeff=0.15,
    ).validate()

    # All four terms 1.0 -> (0.40 + 0.25 + 0.20 + 0.15) / 1.00 = 1.00
    result = graph_score(
        node.id, {seed.id}, pairs, [sim, temp], scenes, stamps, cfg, 5.0
    )
    assert result == pytest.approx(1.0)


def test_graph_score_is_zero_for_an_unreached_node():
    cfg = RankingConfig().validate()
    node, seed = make_node(), make_node()
    assert graph_score(node.id, {seed.id}, {}, [], {}, {}, cfg, 5.0) == 0.0


def test_graph_score_is_ignored_when_graph_weight_is_zero():
    """Condition A must not be able to benefit from graph structure at all."""
    cfg = RankingConfig(graph_weight=0.0).validate()
    node, seed = make_node(), make_node()
    edge = make_edge(seed, node, "SIMILAR_TO", weight=1.0)
    pairs = {(edge.source_node_id, edge.target_node_id): edge}
    assert graph_score(node.id, {seed.id}, pairs, [edge], {}, {}, cfg, 5.0) == 0.0


# ---------------------------------------------------------------------------
# RankingConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_negative_weights():
    with pytest.raises(ValueError, match=">= 0"):
        RankingConfig(vector_weight=-1.0).validate()


def test_config_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="at least one"):
        RankingConfig(
            vector_weight=0, graph_weight=0, temporal_weight=0, confidence_weight=0
        ).validate()


def test_config_rejects_graph_weight_without_coefficients():
    with pytest.raises(ValueError, match="coefficient"):
        RankingConfig(
            vector_weight=0.5,
            graph_weight=0.5,
            graph_direct_coeff=0.0,
            graph_scene_coeff=0.0,
            graph_edge_conf_coeff=0.0,
            graph_temporal_coeff=0.0,
        ).validate()


def test_ablation_configs_are_exactly_the_four_specified_conditions():
    configs = ablation_configs()
    assert set(configs) == {"A", "B", "C", "D"}
    # A is pure vector
    assert configs["A"].vector_weight == 1.0
    assert configs["A"].graph_weight == 0.0
    assert configs["A"].temporal_weight == 0.0
    # B has temporal but no graph
    assert configs["B"].graph_weight == 0.0
    assert configs["B"].temporal_weight > 0
    # C has graph but no temporal
    assert configs["C"].graph_weight > 0
    assert configs["C"].temporal_weight == 0.0
    # D has both
    assert configs["D"].graph_weight > 0
    assert configs["D"].temporal_weight > 0
    # D leaves confidence out, because with no extractor it is a constant.
    assert configs["D"].confidence_weight == 0.0


def test_every_ablation_condition_has_active_weights_summing_above_zero():
    for key, cfg in ablation_configs().items():
        assert cfg.active_weight_sum > 0.0, key


# ---------------------------------------------------------------------------
# End-to-end ranking and tie-breaking
# ---------------------------------------------------------------------------


def _candidate(frame_id, ts, score, video_id=None):
    return {
        "frame_id": frame_id,
        "video_id": video_id or uuid.uuid4(),
        "timestamp_seconds": ts,
        "vector_score": score,
    }


def test_ranking_without_an_expansion_is_pure_vector_ordering():
    """The vector-only control must be exactly what the vector score says."""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ranked = rank_candidates(
        [_candidate(a, 1.0, 0.9), _candidate(b, 2.0, 0.5), _candidate(c, 3.0, 0.7)],
        expansion=None,
        config=RankingConfig(vector_weight=1.0, graph_weight=0.0,
                             temporal_weight=0.0, confidence_weight=0.0).validate(),
    )
    assert [r.frame_id for r in ranked] == [a, c, b]


def test_tie_breaking_is_deterministic_and_documented():
    """Equal final scores fall back to vector score, then timestamp, then id."""
    a, b = uuid.uuid4(), uuid.uuid4()
    # Identical vector scores, and graph/temporal disabled so finals tie.
    cfg = RankingConfig(vector_weight=1.0, graph_weight=0.0,
                        temporal_weight=0.0, confidence_weight=0.0).validate()
    ranked = rank_candidates(
        [_candidate(a, 9.0, 0.5), _candidate(b, 2.0, 0.5)], expansion=None, config=cfg
    )
    # Earlier timestamp wins when everything else is equal.
    assert [r.frame_id for r in ranked] == [b, a]

    # Same timestamp too -> the frame UUID breaks the tie, so the order is
    # stable across runs rather than dependent on input order.
    c, d = uuid.uuid4(), uuid.uuid4()
    ordered_ids = sorted([c, d], key=str)
    ranked2 = rank_candidates(
        [_candidate(c, 5.0, 0.5), _candidate(d, 5.0, 0.5)], expansion=None, config=cfg
    )
    assert [str(r.frame_id) for r in ranked2] == [str(x) for x in ordered_ids]


def test_graph_score_lifts_a_connected_candidate():
    """A candidate reachable via a measured SIMILAR_TO edge gains graph score."""
    seed_frame, linked_frame = uuid.uuid4(), uuid.uuid4()
    seed_node = make_node(node_type="FRAME", frame_id=seed_frame, timestamp=1.0)
    linked_node = make_node(node_type="FRAME", frame_id=linked_frame, timestamp=1.1)
    edge = make_edge(seed_node, linked_node, "SIMILAR_TO", weight=0.95, confidence=1.0)

    expansion = GraphExpansion()
    expansion.nodes[seed_node.id] = seed_node
    expansion.nodes[linked_node.id] = linked_node
    expansion.edges.append(edge)
    expansion.depth_of[seed_node.id] = 0
    expansion.depth_of[linked_node.id] = 1
    expansion.reached_by[linked_node.id].append(edge)

    ranked = rank_candidates(
        [_candidate(seed_frame, 1.0, 0.8), _candidate(linked_frame, 1.1, 0.8)],
        expansion=expansion,
        config=RankingConfig().validate(),
    )
    by_frame = {r.frame_id: r for r in ranked}
    # The linked candidate has a positive graph score; the seed does not,
    # because nothing was traversed to reach a depth-0 node.
    assert by_frame[linked_frame].graph_score > 0.0
    assert by_frame[seed_frame].graph_score == 0.0
    assert by_frame[linked_frame].is_seed is False
    assert by_frame[seed_frame].is_seed is True


def test_ranking_exposes_the_explainability_breakdown():
    seed_frame = uuid.uuid4()
    seed_node = make_node(node_type="FRAME", frame_id=seed_frame, timestamp=1.0)
    expansion = GraphExpansion()
    expansion.nodes[seed_node.id] = seed_node
    expansion.depth_of[seed_node.id] = 0

    ranked = rank_candidates(
        [_candidate(seed_frame, 1.0, 0.9)], expansion=expansion
    )
    item = ranked[0]
    # The four contributions must sum to the final score - that is what makes
    # the "Why this result?" panel trustworthy.
    assert set(item.contributions) == {"vector", "graph", "temporal", "confidence"}
    assert sum(item.contributions.values()) == pytest.approx(item.final_score)


def test_ranking_truncates_to_top_k():
    ids = [uuid.uuid4() for _ in range(5)]
    candidates = [_candidate(f, float(i), 0.9 - i * 0.1) for i, f in enumerate(ids)]
    ranked = rank_candidates(candidates, expansion=None, top_k=3)
    assert len(ranked) == 3


def test_ranking_with_empty_candidates_returns_empty():
    assert rank_candidates([], expansion=None) == []
