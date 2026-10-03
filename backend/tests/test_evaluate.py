import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).parents[2] / "experiments" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_compare_systems_reports_retrieval_and_operational_metrics():
    examples = [{
        "id": "q1",
        "query": "find the red car",
        "relevant_ids": ["frame-1"],
        "relevant_timestamps": [],
        "tolerance_seconds": 2,
        "relevant_relationships": ["before"],
        "expected_contradiction": False,
    }]
    prediction = [{
        "id": "q1",
        "retrieved": [{"id": "frame-1", "timestamp_seconds": 10, "score": .9}],
        "evidence_ids": ["frame-1"],
        "answer_faithfulness": 1,
        "contradiction_detected": False,
        "relationship_labels": ["before"],
        "latency_ms": 12,
        "input_tokens": 10,
        "output_tokens": 5,
    }]
    report = evaluate.compare_systems(examples, prediction, prediction, condition={"llm": "fixed"})
    metrics = report["systems"]["baseline"]["metrics"]
    assert metrics["retrieval_precision"] == 1
    assert metrics["retrieval_recall"] == 1
    assert metrics["mrr"] == 1
    assert metrics["ndcg"] == 1
    assert metrics["token_consumption_mean"] == 15
    assert metrics["latency_ms_mean"] == 12


def test_missing_prediction_is_explicit():
    with pytest.raises(ValueError, match="missing example ids"):
        evaluate.evaluate_system("baseline", [{"id": "q1"}], [], condition={})
