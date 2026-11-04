"""Evaluation runner with golden set and regression thresholds."""
import json
from pathlib import Path
from typing import List, Dict, Any
from app.evaluation.metrics import (
    intent_accuracy,
    escalation_accuracy,
    response_groundedness,
    resolution_rate,
    check_thresholds,
    EvalResult,
)


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
REGRESSION_REPORT_PATH = Path(__file__).parent.parent.parent / "regression_report.json"

# Regression thresholds — CI gate fails if any metric drops below these
REGRESSION_THRESHOLDS: Dict[str, float] = {
    "Intent Accuracy": 0.80,
    "Escalation Accuracy": 0.80,
    "Response Groundedness": 0.70,
    "Ticket Resolution Rate": 0.60,
}


def load_golden_set() -> List[Dict[str, Any]]:
    """Load the golden set of test queries with expected labels."""
    if GOLDEN_SET_PATH.exists():
        with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def run_evaluation(
    predictions: List[str],
    intents: List[str],
    escalations: List[bool],
    responses: List[str],
    contexts: List[List[str]],
    statuses: List[str],
) -> List[EvalResult]:
    """Run all evaluation metrics and return results."""
    results: List[EvalResult] = []
    results.append(intent_accuracy(predictions, intents))
    results.append(escalation_accuracy(escalations, escalations))
    results.append(response_groundedness(responses, contexts))
    results.append(resolution_rate(statuses))
    return results


def save_results(
    results: List[EvalResult],
    output_path: Path = REGRESSION_REPORT_PATH,
    thresholds: Dict[str, float] = REGRESSION_THRESHOLDS,
) -> bool:
    """Save evaluation results to JSON and return True if all thresholds pass."""
    data = [
        {
            "metric": r.name,
            "score": round(r.score, 4),
            "total": r.total,
            "passed": r.passed,
            "threshold": thresholds.get(r.name, 0.0),
            "pass_threshold": r.score >= thresholds.get(r.name, 0.0),
        }
        for r in results
    ]
    report = {
        "results": data,
        "overall_pass": check_thresholds(results, thresholds),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report["overall_pass"]