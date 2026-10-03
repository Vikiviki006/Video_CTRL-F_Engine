"""Unit tests for the information-retrieval metrics.

Every expected value in this file is **computed by hand** and written out in
the comment next to the assertion, so a reader can verify the implementation
without running anything. No expected value was obtained by running the code
and copying the output, which would make the test circular.

The ranked lists below are clearly-labelled synthetic fixtures used only for
metric verification. They are never written to disk and never touch a
production code path.
"""

import math

import pytest

from experiments.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    evaluate_run,
    frame_reduction_percent,
    hit_rate_at_k,
    is_relevant,
    latency_stats,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    storage_overhead_ratio,
)


# ---------------------------------------------------------------------------
# Synthetic ranked lists used across these tests (labelled, not ground truth)
# ---------------------------------------------------------------------------
# A "relevant" marker is expressed as an id in RELEVANT so matching is by id.
RELEVANT = ["f1"]
RANKED_HIT_FIRST = [
    {"id": "f1", "timestamp_seconds": 1.0},  # rank 1  -> relevant
    {"id": "f2", "timestamp_seconds": 2.0},  # rank 2
    {"id": "f3", "timestamp_seconds": 3.0},  # rank 3
    {"id": "f4", "timestamp_seconds": 4.0},  # rank 4
    {"id": "f5", "timestamp_seconds": 5.0},  # rank 5
]
RANKED_HIT_THIRD = [
    {"id": "f2", "timestamp_seconds": 2.0},  # rank 1
    {"id": "f3", "timestamp_seconds": 3.0},  # rank 2
    {"id": "f1", "timestamp_seconds": 1.0},  # rank 3  -> relevant
    {"id": "f4", "timestamp_seconds": 4.0},  # rank 4
]
RANKED_NO_HIT = [
    {"id": "f2", "timestamp_seconds": 2.0},
    {"id": "f3", "timestamp_seconds": 3.0},
    {"id": "f4", "timestamp_seconds": 4.0},
]


# ---------------------------------------------------------------------------
# Relevance matching
# ---------------------------------------------------------------------------


def test_relevance_matches_by_id():
    assert is_relevant({"id": "f1"}, RELEVANT, [], 2.0) is True
    assert is_relevant({"id": "f9"}, RELEVANT, [], 2.0) is False


def test_relevance_matches_by_timestamp_within_tolerance():
    # |1.5 - 1.0| = 0.5 <= 2.0 tolerance -> relevant
    assert is_relevant({"timestamp_seconds": 1.5}, [], [1.0], 2.0) is True
    # |5.0 - 1.0| = 4.0 > 2.0 tolerance -> not relevant
    assert is_relevant({"timestamp_seconds": 5.0}, [], [1.0], 2.0) is False


def test_relevance_tolerance_boundary_is_inclusive():
    # |3.0 - 1.0| = 2.0, exactly the tolerance -> relevant ("<= tolerance")
    assert is_relevant({"timestamp_seconds": 3.0}, [], [1.0], 2.0) is True


def test_item_with_neither_id_nor_timestamp_is_never_relevant():
    assert is_relevant({}, RELEVANT, [1.0], 2.0) is False


# ---------------------------------------------------------------------------
# Precision@K - denominator is K, not min(K, len(ranked))
# ---------------------------------------------------------------------------


def test_precision_at_1_when_relevant_item_is_first():
    # hits = 1, K = 1 -> 1/1 = 1.0
    assert precision_at_k(RANKED_HIT_FIRST, 1, RELEVANT, [], 2.0) == 1.0


def test_precision_at_1_when_relevant_item_is_third():
    # top-1 is {"id": "f2"} -> 0 hits -> 0/1 = 0.0
    assert precision_at_k(RANKED_HIT_THIRD, 1, RELEVANT, [], 2.0) == 0.0


def test_precision_at_3_with_one_hit_in_top_3():
    # RANKED_HIT_THIRD top-3 = [f2, f3, f1]; hits = 1 -> 1/3
    assert precision_at_k(RANKED_HIT_THIRD, 3, RELEVANT, [], 2.0) == pytest.approx(1 / 3)


def test_precision_denominator_is_k_not_retrieved_count():
    # 4 results returned, K = 10, 1 hit -> 1/10, penalising the short list.
    short = [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}, {"id": "f4"}]
    assert precision_at_k(short, 10, RELEVANT, [], 2.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def test_recall_at_1_with_single_relevant_item_found_first():
    # 1 hit / 1 relevant -> 1.0
    assert recall_at_k(RANKED_HIT_FIRST, 1, RELEVANT, [], 2.0) == 1.0


def test_recall_at_2_misses_relevant_item_at_rank_3():
    # 0 hits / 1 relevant -> 0.0
    assert recall_at_k(RANKED_HIT_THIRD, 2, RELEVANT, [], 2.0) == 0.0


def test_recall_at_5_finds_it():
    # 1 hit / 1 relevant -> 1.0
    assert recall_at_k(RANKED_HIT_THIRD, 5, RELEVANT, [], 2.0) == 1.0


def test_recall_with_multiple_relevant_items():
    relevant = ["f1", "f3"]
    # top-2 = [f1, f2] -> 1 hit / 2 relevant -> 0.5
    assert recall_at_k(RANKED_HIT_FIRST, 2, relevant, [], 2.0) == pytest.approx(0.5)


def test_recall_is_zero_when_query_has_no_ground_truth():
    # Undefined ratio; must be 0.0, never 1.0 and never an exception.
    assert recall_at_k(RANKED_HIT_FIRST, 5, [], [], 2.0) == 0.0


# ---------------------------------------------------------------------------
# HitRate@K
# ---------------------------------------------------------------------------


def test_hit_rate_at_1_true_when_first_is_relevant():
    assert hit_rate_at_k(RANKED_HIT_FIRST, 1, RELEVANT, [], 2.0) == 1.0


def test_hit_rate_at_1_false_when_first_is_not_relevant():
    assert hit_rate_at_k(RANKED_HIT_THIRD, 1, RELEVANT, [], 2.0) == 0.0


def test_hit_rate_at_5_true_for_third_position_hit():
    assert hit_rate_at_k(RANKED_HIT_THIRD, 5, RELEVANT, [], 2.0) == 1.0


def test_hit_rate_zero_when_never_relevant():
    assert hit_rate_at_k(RANKED_NO_HIT, 10, RELEVANT, [], 2.0) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_reciprocal_rank_first_position():
    # 1/1 = 1.0
    assert reciprocal_rank(RANKED_HIT_FIRST, RELEVANT, [], 2.0) == 1.0


def test_reciprocal_rank_third_position():
    # 1/3
    assert reciprocal_rank(RANKED_HIT_THIRD, RELEVANT, [], 2.0) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_no_hit():
    assert reciprocal_rank(RANKED_NO_HIT, RELEVANT, [], 2.0) == 0.0


# ---------------------------------------------------------------------------
# Average precision
# ---------------------------------------------------------------------------


def test_average_precision_relevant_first_of_two():
    # ranked = [f1, f2, f3, f4, f5], relevant = {f1}
    # i=1: hit -> P@1 = 1/1 = 1 ; i=2..5: no hits
    # AP = 1 / R = 1/1 = 1.0
    assert average_precision(RANKED_HIT_FIRST, RELEVANT, [], 2.0) == 1.0


def test_average_precision_relevant_at_rank_3():
    # i=3: hit -> P@3 = 1/3 ; AP = (1/3)/1 = 1/3
    assert average_precision(RANKED_HIT_THIRD, RELEVANT, [], 2.0) == pytest.approx(1 / 3)


def test_average_precision_two_relevant_items():
    # ranked = [f1, f2, f3], relevant = {f1, f3}, R = 2
    # i=1: hit -> P@1 = 1   -> contributes 1
    # i=3: hit -> P@3 = 2/3 -> contributes 2/3
    # AP = (1 + 2/3) / 2 = 5/6
    ranked = [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}]
    assert average_precision(ranked, ["f1", "f3"], [], 2.0) == pytest.approx(5 / 6)


def test_average_precision_perfect_ranking_with_two_relevant():
    # ranked = [f1, f3, f2], relevant = {f1, f3}
    # i=1: P=1 ; i=2: P=2/2=1 -> sum 2 ; AP = 2/2 = 1.0
    ranked = [{"id": "f1"}, {"id": "f3"}, {"id": "f2"}]
    assert average_precision(ranked, ["f1", "f3"], [], 2.0) == 1.0


def test_average_precision_zero_without_ground_truth():
    assert average_precision(RANKED_HIT_FIRST, [], [], 2.0) == 0.0


# ---------------------------------------------------------------------------
# NDCG@K  (binary relevance)
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one():
    # relevant first, R = 1, K = 3
    # DCG@3 = 1/log2(2) + 0 + 0 = 1/1 = 1
    # IDCG@3 = 1/log2(2) = 1
    # NDCG = 1.0
    assert ndcg_at_k(RANKED_HIT_FIRST, 3, RELEVANT, [], 2.0) == pytest.approx(1.0)


def test_ndcg_relevant_at_rank_3_with_k_3():
    # DCG@3 = 0 + 0 + 1/log2(4) = 0.5
    # IDCG@3 = 1/log2(2) = 1
    # NDCG@3 = 0.5
    assert ndcg_at_k(RANKED_HIT_THIRD, 3, RELEVANT, [], 2.0) == pytest.approx(0.5)


def test_ndcg_relevant_at_rank_2_with_k_4():
    # DCG@4 = 0 + 1/log2(3) = 0.6309297535714574
    # IDCG@4 = 1/log2(2) = 1
    # NDCG@4 = 0.63093
    ranked = [{"id": "f2"}, {"id": "f1"}, {"id": "f3"}, {"id": "f4"}]
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, 4, RELEVANT, [], 2.0) == pytest.approx(expected)


def test_ndcg_truncates_ideal_at_k():
    # 2 relevant, K = 1: only one position can be ideal.
    # DCG@1 = 1 if f1 first, IDCG@1 = 1 -> 1.0
    ranked = [{"id": "f1"}, {"id": "f3"}]
    assert ndcg_at_k(ranked, 1, ["f1", "f3"], [], 2.0) == 1.0


def test_ndcg_is_zero_when_nothing_relevant_exists():
    assert ndcg_at_k(RANKED_NO_HIT, 10, RELEVANT, [], 2.0) == 0.0


def test_dcg_uses_log2_of_rank_plus_one():
    # rank 1 -> log2(2) = 1 -> gain 1
    assert dcg_at_k([{"id": "f1"}], 1, RELEVANT, [], 2.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------


def test_evaluate_run_averages_across_queries():
    runs = [
        {"ranked": RANKED_HIT_FIRST, "relevant_ids": RELEVANT, "tolerance_seconds": 2.0},
        {"ranked": RANKED_HIT_THIRD, "relevant_ids": RELEVANT, "tolerance_seconds": 2.0},
    ]
    out = evaluate_run(runs, ks=(1, 5), hitrate_ks=(1, 5))
    # recall@1: (1.0 + 0.0) / 2 = 0.5
    assert out["recall@1"] == pytest.approx(0.5)
    # recall@5: (1.0 + 1.0) / 2 = 1.0
    assert out["recall@5"] == pytest.approx(1.0)
    # mrr: (1.0 + 1/3) / 2 = 2/3
    assert out["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    # map: (1.0 + 1/3) / 2
    assert out["map"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert out["num_queries"] == 2


def test_evaluate_run_rejects_empty_input_rather_than_returning_zeros():
    # An empty run has no defined metrics. Reporting zeros here is exactly how
    # a fabricated "Recall 0.0" would enter a results file.
    with pytest.raises(ValueError, match="at least one query run"):
        evaluate_run([])


# ---------------------------------------------------------------------------
# Percentiles and latency
# ---------------------------------------------------------------------------


def test_percentile_median_of_even_sample():
    # [1..10] -> position (10-1)*50/100 = 4.5 -> between 5 and 6 -> 5.5
    assert percentile(list(range(1, 11)), 50) == pytest.approx(5.5)


def test_percentile_p95_of_known_sample():
    values = [float(v) for v in range(1, 21)]  # 1..20
    # position = 19 * 0.95 = 18.05 -> data[18]=19, data[19]=20
    # 19 + (20-19)*0.05 = 19.05
    assert percentile(values, 95) == pytest.approx(19.05)


def test_percentile_p99_of_known_sample():
    values = [float(v) for v in range(1, 21)]
    # position = 19 * 0.99 = 18.81 -> 19 + 1*0.81 = 19.81
    assert percentile(values, 99) == pytest.approx(19.81)


def test_percentile_handles_single_sample():
    assert percentile([7.0], 95) == 7.0


def test_latency_stats_for_known_sample():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = latency_stats(values)
    assert stats["count"] == 5
    assert stats["avg_ms"] == pytest.approx(3.0)
    assert stats["median_ms"] == pytest.approx(3.0)
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 5.0
    assert stats["p95_ms"] == pytest.approx(4.8)  # pos = 4*0.95 = 3.8 -> 4 + 1*0.8


def test_latency_stats_empty_sample_is_none_not_zero():
    stats = latency_stats([])
    assert stats["count"] == 0
    # None renders as N/A in the reports; 0.0 would read as "measured, zero ms".
    assert stats["avg_ms"] is None
    assert stats["p95_ms"] is None


# ---------------------------------------------------------------------------
# Indexing / storage arithmetic
# ---------------------------------------------------------------------------


def test_frame_reduction_percent():
    # (1 - 250/1000) * 100 = 75.0
    assert frame_reduction_percent(1000, 250) == pytest.approx(75.0)


def test_frame_reduction_percent_zero_original_is_none():
    assert frame_reduction_percent(0, 10) is None


def test_frame_reduction_percent_no_reduction_is_zero():
    assert frame_reduction_percent(100, 100) == pytest.approx(0.0)


def test_storage_overhead_ratio():
    # 500 derived / 1000 raw = 0.5
    assert storage_overhead_ratio(1000, 500) == pytest.approx(0.5)


def test_storage_overhead_ratio_zero_raw_is_none():
    assert storage_overhead_ratio(0, 500) is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_k(k):
    for fn in (precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            fn(RANKED_HIT_FIRST, k, RELEVANT, [], 2.0)


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)
