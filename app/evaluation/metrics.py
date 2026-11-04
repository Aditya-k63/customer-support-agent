"""Core evaluation metrics for the support agent."""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    score: float
    total: int
    passed: int
    details: List[Dict[str, Any]]


def intent_accuracy(predictions: List[str], ground_truth: List[str]) -> EvalResult:
    """% of queries classified to the correct intent category."""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(ground_truth)
    details = [
        {
            "query": q,
            "predicted": p,
            "expected": g,
            "pass": p == g,
        }
        for q, p, g in zip(ground_truth, predictions, ground_truth)
    ]
    return EvalResult(
        name="Intent Accuracy",
        score=correct / total if total else 0.0,
        total=total,
        passed=correct,
        details=details,
    )


def escalation_accuracy(predictions: List[bool], ground_truth: List[bool]) -> EvalResult:
    """% of times escalation decision matches ground truth."""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total = len(ground_truth)
    return EvalResult(
        name="Escalation Accuracy",
        score=correct / total if total else 0.0,
        total=total,
        passed=correct,
        details=[],
    )


def response_groundedness(responses: List[str], contexts: List[List[str]]) -> EvalResult:
    """% of responses where key claims are supported by retrieved context."""
    passed = 0
    total = len(responses)
    details = []
    for response, context in zip(responses, contexts):
        context_text = " ".join(context).lower()
        response_words = set(response.lower().split())
        context_words = set(context_text.split())
        overlap = len(response_words & context_words)
        supported = overlap > 0
        if supported:
            passed += 1
        details.append(
            {
                "response": response[:100],
                "context_overlap": overlap,
                "pass": supported,
            }
        )
    return EvalResult(
        name="Response Groundedness",
        score=passed / total if total else 0.0,
        total=total,
        passed=passed,
        details=details,
    )


def resolution_rate(statuses: List[str]) -> EvalResult:
    """% of tickets that reached 'resolved' without human escalation."""
    resolved = sum(1 for s in statuses if s == "resolved")
    total = len(statuses)
    return EvalResult(
        name="Ticket Resolution Rate",
        score=resolved / total if total else 0.0,
        total=total,
        passed=resolved,
        details=[],
    )


def check_thresholds(results: List[EvalResult], thresholds: Dict[str, float]) -> bool:
    """Return True if all metrics meet their minimum thresholds."""
    for result in results:
        threshold = thresholds.get(result.name, 0.0)
        if result.score < threshold:
            return False
    return True