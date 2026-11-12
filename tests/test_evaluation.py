"""Unit tests for evaluation metrics and dataset integrity."""
import json
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.metrics import (
    check_thresholds,
    escalation_accuracy,
    intent_accuracy,
    resolution_rate,
    response_groundedness,
    status_accuracy,
)
from app.evaluation.metrics_v2 import mrr, precision_at_k, recall_at_k


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


def test_status_accuracy():
    result = status_accuracy(
        ["resolved", "in_progress", "open"], ["resolved", "in_progress", "resolved"]
    )
    assert result.score == 2 / 3


def test_response_groundedness():
    responses = ["You can refund within 30 days."]
    contexts = [["Refund policy: Items can be refunded within 30 days of purchase."]]
    result = response_groundedness(responses, contexts)
    assert result.score == 1.0


def test_response_groundedness_rejects_unrelated_answer():
    responses = ["Bananas are yellow."]
    contexts = [["Refund policy: Items can be refunded within 30 days."]]
    result = response_groundedness(responses, contexts)
    assert result.score == 0.0


def test_retrieval_metrics_use_document_ids():
    retrieved = [["refund_policy", "billing_invoices", "shipping"]]
    relevant = [["refund_policy", "billing_invoices"]]
    assert precision_at_k(retrieved, relevant, k=3).score == pytest.approx(2 / 3, abs=1e-4)
    assert recall_at_k(retrieved, relevant, k=3).score == 1.0
    assert mrr(retrieved, relevant).score == 1.0


def test_retrieval_metrics_skip_queries_without_relevant_docs():
    retrieved = [["shipping"], ["refund_policy"]]
    relevant = [[], ["refund_policy"]]
    precision = precision_at_k(retrieved, relevant, k=3)
    assert precision.total == 1
    assert precision.score == 1.0
    assert mrr(retrieved, relevant).total == 1
    assert recall_at_k(retrieved, relevant, k=3).total == 1


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
    assert all("relevant_doc_ids" in item for item in data)
    # Every labelled document must exist in the knowledge base.
    from ingestion.knowledge_base import KNOWLEDGE_BASE

    known_ids = {document["id"] for document in KNOWLEDGE_BASE}
    for item in data:
        assert set(item["relevant_doc_ids"]).issubset(known_ids)

    dist = Counter(item["category"] for item in data)
    for cat in ["billing", "technical", "general", "refund_request", "human_handoff"]:
        assert dist[cat] == 30, f"Expected 30 {cat} questions, got {dist[cat]}"


def _judge_with_raw(raw: str):
    from unittest.mock import AsyncMock, patch

    from app.evaluation.llm_judge import LLMJudge

    judge = LLMJudge()
    patcher = patch.object(judge.llm, "generate", AsyncMock(return_value=raw))
    return judge, patcher


def test_judge_parses_supported():
    import asyncio

    judge, patcher = _judge_with_raw("SUPPORTED")
    with patcher:
        assert asyncio.run(judge.classify("answer", "context")) == "SUPPORTED"
        assert asyncio.run(judge.judge_groundedness("answer", "context")) is True


def test_judge_parses_unsupported_not_supported():
    import asyncio

    judge, patcher = _judge_with_raw("UNSUPPORTED")
    with patcher:
        assert asyncio.run(judge.classify("answer", "context")) == "UNSUPPORTED"


def test_judge_parses_refusal():
    import asyncio

    judge, patcher = _judge_with_raw("REFUSAL")
    with patcher:
        assert asyncio.run(judge.classify("answer", "context")) == "REFUSAL"
        assert asyncio.run(judge.judge_groundedness("answer", "context")) is False


def test_judge_treats_unknown_output_as_unsupported():
    import asyncio

    judge, patcher = _judge_with_raw("I am not sure what you mean")
    with patcher:
        assert asyncio.run(judge.classify("answer", "context")) == "UNSUPPORTED"
