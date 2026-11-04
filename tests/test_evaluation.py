"""Unit tests for evaluation metrics."""
import json
from pathlib import Path

from app.evaluation.metrics import (
    intent_accuracy,
    escalation_accuracy,
    response_groundedness,
    resolution_rate,
    check_thresholds,
)


def test_intent_accuracy_perfect():
    preds = ["billing", "technical", "general", "refund_request"]
    truths = ["billing", "technical", "general", "refund_request"]
    result = intent_accuracy(preds, truths)
    assert result.score == 1.0
    assert result.passed == 4
    assert result.total == 4


def test_intent_accuracy_partial():
    preds = ["billing", "technical", "general", "billing"]
    truths = ["billing", "technical", "general", "refund_request"]
    result = intent_accuracy(preds, truths)
    assert result.score == 0.75
    assert result.passed == 3


def test_escalation_accuracy_perfect():
    preds = [True, False, True, False]
    truths = [True, False, True, False]
    result = escalation_accuracy(preds, truths)
    assert result.score == 1.0


def test_resolution_rate():
    statuses = ["resolved", "resolved", "in_progress", "resolved"]
    result = resolution_rate(statuses)
    assert result.score == 0.75


def test_response_groundedness():
    responses = ["You can refund within 30 days."]
    contexts = [["Refund policy: Items can be refunded within 30 days of purchase."]]
    result = response_groundedness(responses, contexts)
    assert result.score == 1.0


def test_check_thresholds_pass():
    from app.evaluation.metrics import EvalResult

    results = [
        EvalResult(name="Intent Accuracy", score=0.9, total=10, passed=9, details=[]),
        EvalResult(name="Escalation Accuracy", score=0.9, total=10, passed=9, details=[]),
    ]
    thresholds = {"Intent Accuracy": 0.8, "Escalation Accuracy": 0.8}
    assert check_thresholds(results, thresholds) is True


def test_check_thresholds_fail():
    from app.evaluation.metrics import EvalResult

    results = [
        EvalResult(name="Intent Accuracy", score=0.5, total=10, passed=5, details=[]),
    ]
    thresholds = {"Intent Accuracy": 0.8}
    assert check_thresholds(results, thresholds) is False


def test_golden_set_loads():
    path = Path("app/evaluation/golden_set.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 150
    assert all("expected_intent" in item for item in data)
    assert all("expected_escalate" in item for item in data)
    assert all("expected_status" in item for item in data)
    assert all("category" in item for item in data)
    # Verify 30 per category
    from collections import Counter
    dist = Counter(item["category"] for item in data)
    for cat in ["billing", "technical", "general", "refund_request", "human_handoff"]:
        assert dist[cat] == 30, f"Expected 30 {cat} questions, got {dist[cat]}"