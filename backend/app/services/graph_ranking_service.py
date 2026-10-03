"""Graph-aware re-ranking for VideoCtrl-F (Phase 2).

The scoring formula, in full
===========================

Every score below is clamped to ``[0, 1]`` before use, and the weighted sum is
divided by the sum of the active weights, so ``final_score`` is always a
convex combination and always lands in ``[0, 1]``.

.. code-block:: text

    final_score = ( alpha * vector_score
                  + beta  * graph_score
                  + gamma * temporal_score
                  + delta * confidence_score )
                  / ( alpha + beta + gamma + delta )

    with the defaults shipped in .env:

        alpha = VECTOR_WEIGHT     = 0.65
        beta  = GRAPH_WEIGHT      = 0.20
        gamma = TEMPORAL_WEIGHT   = 0.10
        delta = CONFIDENCE_WEIGHT = 0.05
        ( 0.65 + 0.20 + 0.10 + 0.05 = 1.00 )

VectorScore
-----------
``vector_score = clip01( 1 - cosine_distance(query_embedding, frame_embedding) )``

This is the *identical* quantity the existing ``/api/v1/search`` ranks on, read
straight from pgvector's ``<=>`` operator. It is not a new signal and it is not
renormalised, so the vector baseline and the graph baseline remain directly
comparable. Because SigLIP-2 cosine similarity is only meaningful above chance,
it is clipped to ``[0, 1]``: a negative cosine similarity contributes 0 rather
than a negative score that would fight the other terms.

GraphScore
----------
A documented convex combination of four sub-scores, each traceable to a real
edge that the bounded traversal actually traversed:

.. code-block:: text

    graph_score = ( a_d * direct_similarity
                  + a_s * scene_overlap
                  + a_c * edge_confidence
                  + a_t * temporal_adjacency )
                  / ( a_d + a_s + a_c + a_t )

    a_d = GRAPH_DIRECT_COEFF      = 0.40
    a_s = GRAPH_SCENE_COEFF       = 0.25
    a_c = GRAPH_EDGE_CONF_COEFF   = 0.20
    a_t = GRAPH_TEMPORAL_COEFF    = 0.15

**direct_similarity** - "this frame looks like something the query already
matched". For candidate ``F`` and the seed set ``S`` (the vector candidates):

.. code-block:: text

    direct_similarity(F, S) = max over SIMILAR_TO edges (F <-> G), G in S  of edge.weight
                            = 0  if no such edge exists

``edge.weight`` is the *measured* cosine similarity stored when the edge was
built, so this term can never exceed the evidence that produced it.

**scene_overlap** - "this frame shares a scene with a strong hit", i.e. it is
part of the same contiguous keyframe cluster:

.. code-block:: text

    scene_overlap(F, S) = min( 1, |SCENE(F)  intersection  SCENE(S)| / max(1, |SCENE(S)|) )

where ``SCENE(x)`` is the set of SCENE nodes containing ``x``'s FRAME node and
``SCENE(S)`` is the union over all seeds. With the default scene window this
term rewards a frame that sits inside an already-relevant shot.

**edge_confidence** - how much the producing stage trusted the evidence that
reached ``F``:

.. code-block:: text

    edge_confidence(F) = mean( edge.confidence for each edge traversed to reach F )
                       = 0  for a seed reached at depth 0 (nothing was traversed)

**temporal_adjacency** - "``F`` sits right next to a strong hit in time", and
only counted when a real temporal edge connects them:

.. code-block:: text

    temporal_adjacency(F, S) = max over temporal edges (F <-> G), G in S  of
                                 1 / (1 + |t(F) - t(G)| / TEMPORAL_WINDOW_SECONDS)
                             = 0  if no temporal edge connects F to any seed

The ``1 / (1 + d/W)`` form is monotone decreasing in the timestamp gap ``d`` and
equals 1 at zero gap, reaching 0.5 at exactly one window. It is a chosen
convention, not a fitted or optimal value.

TemporalScore
-------------
Independent of the graph. It measures whether a result sits in a *dense region
of the timeline* rather than being an isolated spike, using the other retained
candidates as evidence:

.. code-block:: text

    temporal_score(r) = max( 0, 1 - min_{r' != r} |t(r) - t(r')| / TEMPORAL_WINDOW_SECONDS )
                     = 0  when r is the only candidate

Rationale: a contiguous run of relevant frames is more likely to be a real
coherent scene than one isolated frame that happens to score well. This is a
design choice, stated as such, not a validated claim - the Phase 3 ablation
exists precisely to measure whether it actually helps.

ConfidenceScore
---------------
.. code-block:: text

    confidence_score(r) = ( geometric mean of node.confidence over r and its related nodes )

Kept separate from ``graph_score``'s ``edge_confidence`` because that term
scores the *edges*, while this scores the *nodes* an extractor produced.

Honest limitation: with the default ``NullGraphExtractor`` there is no
extractor, every node is built deterministically with ``confidence = 1.0``, and
so this term is identically 1.0 for every candidate. It is therefore
**ranking-neutral in the shipped configuration**, and the evaluation harness
records that fact rather than implying the term is doing work. It becomes
meaningful only once a real extractor is plugged in.

Tie-breaking
------------
Sorting is total and deterministic:

.. code-block:: text

    key = ( -final_score, -vector_score, timestamp_seconds, frame_id )

so ties fall back first to the raw vector score, then to the earlier
timestamp, then to the frame UUID. Two runs over the same data always produce
byte-identical orderings, which is what makes the ablation numbers
reproducible.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.graph import TEMPORAL_RELATIONSHIPS, GraphEdge, GraphNode
from app.services.graph_service import GraphExpansion

logger = setup_logging()
settings = get_settings()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingConfig:
    """Weight set for one ranking configuration.

    The four ablation conditions of the evaluation study are expressed purely
    as different :class:`RankingConfig` values, so the ablation exercises the
    exact same code path as production and cannot drift from it.

    ==========================  ======  ======  ======  ======  =========
    Condition                   vector  graph   temporal  conf   name
    ==========================  ======  ======  ======  ======  =========
    A - vector only             1.00    0.00    0.00      0.00   ``A``
    B - vector + temporal       0.65    0.00    0.35      0.00   ``B``
    C - vector + graph          0.80    0.20    0.00      0.00   ``C``
    D - vector + graph + temp   0.65    0.20    0.15      0.00   ``D``
    ==========================  ======  ======  ======  ======  =========
    """

    name: str = "production"
    vector_weight: float = 0.65
    graph_weight: float = 0.20
    temporal_weight: float = 0.10
    confidence_weight: float = 0.05
    graph_direct_coeff: float = 0.40
    graph_scene_coeff: float = 0.25
    graph_edge_conf_coeff: float = 0.20
    graph_temporal_coeff: float = 0.15

    @property
    def active_weight_sum(self) -> float:
        return (
            self.vector_weight
            + self.graph_weight
            + self.temporal_weight
            + self.confidence_weight
        )

    @property
    def graph_coeff_sum(self) -> float:
        return (
            self.graph_direct_coeff
            + self.graph_scene_coeff
            + self.graph_edge_conf_coeff
            + self.graph_temporal_coeff
        )

    def validate(self) -> "RankingConfig":
        """Reject a configuration that could produce an unbounded or NaN score."""
        for name, value in (
            ("vector_weight", self.vector_weight),
            ("graph_weight", self.graph_weight),
            ("temporal_weight", self.temporal_weight),
            ("confidence_weight", self.confidence_weight),
            ("graph_direct_coeff", self.graph_direct_coeff),
            ("graph_scene_coeff", self.graph_scene_coeff),
            ("graph_edge_conf_coeff", self.graph_edge_conf_coeff),
            ("graph_temporal_coeff", self.graph_temporal_coeff),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.active_weight_sum <= 0.0:
            raise ValueError("at least one ranking weight must be > 0")
        if self.graph_weight > 0.0 and self.graph_coeff_sum <= 0.0:
            raise ValueError(
                "graph_weight > 0 requires at least one GraphScore coefficient > 0"
            )
        return self

    @classmethod
    def from_settings(cls, name: str = "production") -> "RankingConfig":
        """Build from the environment-driven settings."""
        return cls(
            name=name,
            vector_weight=settings.VECTOR_WEIGHT,
            graph_weight=settings.GRAPH_WEIGHT,
            temporal_weight=settings.TEMPORAL_WEIGHT,
            confidence_weight=settings.CONFIDENCE_WEIGHT,
            graph_direct_coeff=settings.GRAPH_DIRECT_COEFF,
            graph_scene_coeff=settings.GRAPH_SCENE_COEFF,
            graph_edge_conf_coeff=settings.GRAPH_EDGE_CONF_COEFF,
            graph_temporal_coeff=settings.GRAPH_TEMPORAL_COEFF,
        ).validate()


def ablation_configs() -> Dict[str, RankingConfig]:
    """The four ablation conditions A/B/C/D from the evaluation study.

    Weights are renormalised so that, within each condition, the active terms
    sum to 1.0. That keeps every condition on the same scale and makes the
    comparison a statement about *which signals are used*, not about their raw
    magnitudes.
    """
    return {
        "A": RankingConfig(
            name="A: vector only",
            vector_weight=1.0,
            graph_weight=0.0,
            temporal_weight=0.0,
            confidence_weight=0.0,
        ).validate(),
        "B": RankingConfig(
            name="B: vector + temporal",
            vector_weight=0.65,
            graph_weight=0.0,
            temporal_weight=0.35,
            confidence_weight=0.0,
        ).validate(),
        "C": RankingConfig(
            name="C: vector + graph",
            vector_weight=0.80,
            graph_weight=0.20,
            temporal_weight=0.0,
            confidence_weight=0.0,
        ).validate(),
        "D": RankingConfig(
            name="D: vector + graph + temporal",
            vector_weight=0.65,
            graph_weight=0.20,
            temporal_weight=0.15,
            confidence_weight=0.0,
        ).validate(),
    }


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


def clip01(value: float) -> float:
    """Clamp to ``[0, 1]``. NaN maps to 0.0 rather than propagating."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def vector_score(cosine_similarity: float) -> float:
    """VectorScore = clip01(1 - cosine_distance) = clip01(cosine_similarity)."""
    return clip01(cosine_similarity)


def direct_similarity(
    node_id: uuid.UUID,
    seed_ids: Set[uuid.UUID],
    edges_by_pair: Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge],
) -> float:
    """GraphScore.direct_similarity - strongest measured SIMILAR_TO link to a seed."""
    best = 0.0
    for seed in seed_ids:
        for key in ((node_id, seed), (seed, node_id)):
            edge = edges_by_pair.get(key)
            if edge is not None:
                best = max(best, clip01(edge.weight))
    return best


def scene_overlap(
    scenes_of_node: Dict[uuid.UUID, Set[uuid.UUID]],
    node_id: uuid.UUID,
    seed_ids: Set[uuid.UUID],
) -> float:
    """GraphScore.scene_overlap - fraction of seed scenes this node also sits in."""
    seed_scenes: Set[uuid.UUID] = set()
    for seed in seed_ids:
        seed_scenes |= scenes_of_node.get(seed, set())
    if not seed_scenes:
        return 0.0
    own = scenes_of_node.get(node_id, set())
    return clip01(len(own & seed_scenes) / len(seed_scenes))


def edge_confidence_score(reached_edges: Sequence[GraphEdge]) -> float:
    """GraphScore.edge_confidence - mean confidence of the traversed edges."""
    if not reached_edges:
        return 0.0
    return clip01(sum(clip01(e.confidence) for e in reached_edges) / len(reached_edges))


def temporal_adjacency(
    node_id: uuid.UUID,
    node_timestamps: Dict[uuid.UUID, float],
    seed_ids: Set[uuid.UUID],
    edges_by_pair: Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge],
    window_seconds: float,
) -> float:
    """GraphScore.temporal_adjacency - 1/(1 + dt/W) over real temporal edges only."""
    if window_seconds <= 0.0:
        return 0.0
    own_ts = node_timestamps.get(node_id)
    if own_ts is None:
        return 0.0
    best = 0.0
    for seed in seed_ids:
        for key in ((node_id, seed), (seed, node_id)):
            edge = edges_by_pair.get(key)
            if edge is None or edge.relationship not in TEMPORAL_RELATIONSHIPS:
                continue
            seed_ts = node_timestamps.get(seed)
            if seed_ts is None:
                continue
            gap = abs(own_ts - seed_ts)
            best = max(best, 1.0 / (1.0 + gap / window_seconds))
    return clip01(best)


def graph_score(
    node_id: uuid.UUID,
    seed_ids: Set[uuid.UUID],
    edges_by_pair: Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge],
    reached_edges: Sequence[GraphEdge],
    scenes_of_node: Dict[uuid.UUID, Set[uuid.UUID]],
    node_timestamps: Dict[uuid.UUID, float],
    config: RankingConfig,
    window_seconds: float,
) -> float:
    """GraphScore - the documented four-term convex combination.

    Each term is in ``[0, 1]``; the result is divided by the coefficient sum so
    it is also in ``[0, 1]`` regardless of how the coefficients are set.
    """
    if config.graph_weight <= 0.0:
        return 0.0

    terms = (
        config.graph_direct_coeff
        * direct_similarity(node_id, seed_ids, edges_by_pair),
        config.graph_scene_coeff
        * scene_overlap(scenes_of_node, node_id, seed_ids),
        config.graph_edge_conf_coeff * edge_confidence_score(reached_edges),
        config.graph_temporal_coeff
        * temporal_adjacency(
            node_id, node_timestamps, seed_ids, edges_by_pair, window_seconds
        ),
    )
    total = sum(terms)
    coeff_sum = config.graph_coeff_sum
    if coeff_sum <= 0.0:
        return 0.0
    return clip01(total / coeff_sum)


def temporal_score(
    timestamp: float,
    other_timestamps: Sequence[float],
    window_seconds: float,
) -> float:
    """TemporalScore - density of *other* candidates near this one in time.

    .. code-block:: text

        temporal_score(r) = max( 0, 1 - min_{r' != r} |t(r) - t(r')| / TEMPORAL_WINDOW_SECONDS )
                         = 0  when r is the only candidate

    ``other_timestamps`` must exclude ``r`` itself. Including it would make
    every candidate's nearest gap exactly 0.0 and therefore score 1.0 for all
    of them, which would make the term a constant and silently useless - the
    caller is responsible for passing the other candidates only.

    Rationale: a contiguous run of relevant frames is more likely to be a real
    coherent scene than one isolated frame that happens to score well. This is a
    design choice, stated as such, not a validated claim - the Phase 3 ablation
    exists precisely to measure whether it actually helps.
    """
    if window_seconds <= 0.0:
        return 0.0
    nearest: Optional[float] = None
    for other in other_timestamps:
        gap = abs(other - timestamp)
        if nearest is None or gap < nearest:
            nearest = gap
    if nearest is None:
        return 0.0
    return clip01(max(0.0, 1.0 - nearest / window_seconds))


def confidence_score(confidences: Sequence[float]) -> float:
    """ConfidenceScore - geometric mean of node confidences.

    Geometric rather than arithmetic so a single unconfident node drags the
    whole score down, which is the intended behaviour for extractor output.

    With no extractor configured every node has ``confidence = 1.0``, so this
    returns exactly 1.0 for every candidate and contributes a constant. That is
    a true property of the current system, not an assumption.
    """
    values = [clip01(c) for c in confidences]
    if not values:
        return 0.0
    if any(c <= 0.0 for c in values):
        return 0.0
    return clip01(math.exp(sum(math.log(c) for c in values) / len(values)))


def combine_scores(
    *,
    vector: float,
    graph: float,
    temporal: float,
    confidence: float,
    config: RankingConfig,
) -> float:
    """The final weighted combination, normalised by the active weight sum."""
    total = (
        config.vector_weight * clip01(vector)
        + config.graph_weight * clip01(graph)
        + config.temporal_weight * clip01(temporal)
        + config.confidence_weight * clip01(confidence)
    )
    weight_sum = config.active_weight_sum
    if weight_sum <= 0.0:
        return 0.0
    return clip01(total / weight_sum)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """One re-ranked result, including its full score breakdown.

    The breakdown is what powers the "Why this result?" panel: the frontend
    renders ``contributions`` directly and needs no extra endpoint.
    """

    frame_id: uuid.UUID
    video_id: Optional[uuid.UUID]
    timestamp_seconds: float
    vector_score: float
    graph_score: float
    temporal_score: float
    confidence_score: float
    final_score: float
    contributions: Dict[str, float] = field(default_factory=dict)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    depth: Optional[int] = None
    is_seed: bool = False

    def sort_key(self) -> Tuple[float, float, float, str]:
        """Total, deterministic ordering. See module docstring, Tie-breaking."""
        return (
            -self.final_score,
            -self.vector_score,
            self.timestamp_seconds,
            str(self.frame_id),
        )


def _index_expansion(
    expansion: GraphExpansion,
) -> Tuple[
    Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge],
    Dict[uuid.UUID, Set[uuid.UUID]],
    Dict[uuid.UUID, float],
]:
    """Flatten an expansion into lookup tables. Pure in-memory, no queries."""
    edges_by_pair: Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge] = {}
    for edge in expansion.edges:
        edges_by_pair.setdefault((edge.source_node_id, edge.target_node_id), edge)

    scenes_of_node: Dict[uuid.UUID, Set[uuid.UUID]] = {}
    node_timestamps: Dict[uuid.UUID, float] = {}
    for node in expansion.nodes.values():
        if node.node_type == "FRAME" and node.timestamp_start is not None:
            node_timestamps[node.id] = float(node.timestamp_start)
        for edge in expansion.edges:
            if edge.source_node_id == node.id and edge.relationship == "CONTAINS":
                target = expansion.nodes.get(edge.target_node_id)
                if target is not None and target.node_type == "SCENE":
                    scenes_of_node.setdefault(node.id, set()).add(target.id)
            if edge.target_node_id == node.id and edge.relationship == "CONTAINS":
                source = expansion.nodes.get(edge.source_node_id)
                if source is not None and source.node_type == "SCENE":
                    scenes_of_node.setdefault(node.id, set()).add(source.id)
    return edges_by_pair, scenes_of_node, node_timestamps


def rank_candidates(
    candidates: Sequence[Dict[str, Any]],
    expansion: Optional[GraphExpansion],
    *,
    config: Optional[RankingConfig] = None,
    window_seconds: Optional[float] = None,
    top_k: Optional[int] = None,
    include_explanations: bool = True,
) -> List[ScoredCandidate]:
    """Re-rank vector candidates using graph context.

    Parameters
    ----------
    candidates:
        Dicts with ``frame_id``, ``video_id``, ``timestamp_seconds`` and
        ``vector_score``. This is exactly what the pgvector query yields, so
        the vector-only baseline and the graph pipeline consume the same input.
    expansion:
        Result of :func:`~app.services.graph_service.expand_from_seeds`, or
        ``None`` for the graph-free condition. When ``None`` every graph term
        is 0 and only the vector and temporal terms are active, which is what
        makes condition A/B a true ablation rather than a different code path.
    config:
        Weight set. Defaults to :meth:`RankingConfig.from_settings`.
    window_seconds:
        Temporal scale. Defaults to ``settings.TEMPORAL_WINDOW_SECONDS``.
    top_k:
        Truncate after sorting. ``None`` returns everything.
    include_explanations:
        Populate ``related_nodes`` / ``relationships``. Turning this off is a
        pure performance option and does not change any score.
    """
    cfg = (config or RankingConfig.from_settings()).validate()
    win = settings.TEMPORAL_WINDOW_SECONDS if window_seconds is None else window_seconds

    if expansion is None:
        edges_by_pair: Dict[Tuple[uuid.UUID, uuid.UUID], GraphEdge] = {}
        scenes_of_node: Dict[uuid.UUID, Set[uuid.UUID]] = {}
        node_timestamps: Dict[uuid.UUID, float] = {}
        seed_ids: Set[uuid.UUID] = set()
        depth_of: Dict[uuid.UUID, int] = {}
        reached_by: Dict[uuid.UUID, List[GraphEdge]] = {}
    else:
        edges_by_pair, scenes_of_node, node_timestamps = _index_expansion(expansion)
        depth_of = expansion.depth_of
        reached_by = expansion.reached_by
        seed_ids = {nid for nid, d in depth_of.items() if d == 0}

    all_ts = [float(c["timestamp_seconds"]) for c in candidates]

    # Map each candidate frame to its graph node id so the graph terms apply to
    # exactly the frames being ranked.
    frame_id_to_node: Dict[uuid.UUID, uuid.UUID] = {}
    if expansion is not None:
        for node in expansion.nodes.values():
            if node.node_type == "FRAME" and node.frame_id is not None:
                frame_id_to_node.setdefault(node.frame_id, node.id)

    scored: List[ScoredCandidate] = []
    for index, cand in enumerate(candidates):
        frame_id = cand["frame_id"]
        ts = float(cand["timestamp_seconds"])
        v = vector_score(cand["vector_score"])

        # Every candidate except this one. Excluding self is essential: a zero
        # gap to itself would otherwise make temporal_score 1.0 for all.
        other_ts = all_ts[:index] + all_ts[index + 1 :]
        node_id = frame_id_to_node.get(frame_id)
        if node_id is None:
            g = 0.0
            reached: List[GraphEdge] = []
            depth: Optional[int] = None
        else:
            reached = reached_by.get(node_id, [])
            depth = depth_of.get(node_id)
            g = graph_score(
                node_id,
                seed_ids,
                edges_by_pair,
                reached,
                scenes_of_node,
                node_timestamps,
                cfg,
                win,
            )

        t = temporal_score(ts, other_ts, win)

        confidences: List[float] = []
        if node_id is not None and expansion is not None:
            node = expansion.nodes.get(node_id)
            if node is not None:
                confidences.append(node.confidence)
                if include_explanations:
                    for edge in reached:
                        other_id = (
                            edge.target_node_id
                            if edge.source_node_id == node_id
                            else edge.source_node_id
                        )
                        other = expansion.nodes.get(other_id)
                        if other is not None:
                            confidences.append(other.confidence)
        c = confidence_score(confidences) if confidences else 0.0

        final = combine_scores(vector=v, graph=g, temporal=t, confidence=c, config=cfg)

        item = ScoredCandidate(
            frame_id=frame_id,
            video_id=cand.get("video_id"),
            timestamp_seconds=ts,
            vector_score=v,
            graph_score=g,
            temporal_score=t,
            confidence_score=c,
            final_score=final,
            depth=depth,
            is_seed=depth == 0,
        )

        weight_sum = cfg.active_weight_sum
        item.contributions = {
            "vector": cfg.vector_weight * v / weight_sum if weight_sum else 0.0,
            "graph": cfg.graph_weight * g / weight_sum if weight_sum else 0.0,
            "temporal": cfg.temporal_weight * t / weight_sum if weight_sum else 0.0,
            "confidence": cfg.confidence_weight * c / weight_sum if weight_sum else 0.0,
        }

        if include_explanations and node_id is not None and expansion is not None:
            item.related_nodes = _describe_related_nodes(node_id, expansion, include_explanations)
            item.relationships = _describe_relationships(node_id, reached)

        scored.append(item)

    scored.sort(key=lambda s: s.sort_key())
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def _describe_related_nodes(
    node_id: uuid.UUID,
    expansion: GraphExpansion,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Compact, JSON-safe description of the nodes reached from ``node_id``.

    Ordered by traversal depth then label, so the frontend's "related concepts"
    list is stable across runs.
    """
    out: List[Dict[str, Any]] = []
    seen: Set[uuid.UUID] = set()
    for other_id, edges in sorted(
        expansion.reached_by.items(), key=lambda kv: (expansion.depth_of.get(kv[0], 0), str(kv[0]))
    ):
        if other_id == node_id or other_id in seen:
            continue
        node = expansion.nodes.get(other_id)
        if node is None:
            continue
        seen.add(other_id)
        out.append(
            {
                "id": str(other_id),
                "node_type": node.node_type,
                "label": node.label,
                "entity_key": node.entity_key,
                "frame_id": str(node.frame_id) if node.frame_id else None,
                "timestamp_seconds": node.timestamp_start,
                "confidence": node.confidence,
                "depth": expansion.depth_of.get(other_id),
            }
        )
        if len(out) >= limit:
            break
    return out


def _describe_relationships(
    node_id: uuid.UUID, reached_edges: Sequence[GraphEdge]
) -> List[Dict[str, Any]]:
    """The concrete edge chain that justified this result's graph score."""
    out: List[Dict[str, Any]] = []
    for edge in reached_edges[:20]:
        out.append(
            {
                "relationship": edge.relationship,
                "source_node_id": str(edge.source_node_id),
                "target_node_id": str(edge.target_node_id),
                "weight": edge.weight,
                "confidence": edge.confidence,
                "basis": (edge.edge_metadata or {}).get("basis"),
            }
        )
    return out
