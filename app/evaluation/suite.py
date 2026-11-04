"""Evaluation runner with golden set, regression thresholds, and baseline comparison."""
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
BASELINE_REPORT_PATH = Path(__file__).parent.parent.parent / "baseline_report.json"

# Regression thresholds — CI gate fails if any metric drops below these
# v1 baseline thresholds — match current placeholder implementation
# v2 will improve these with real embeddings and tighter thresholds
REGRESSION_THRESHOLDS: Dict[str, float] = {
    "Intent Accuracy": 0.50,
    "Escalation Accuracy": 0.60,
    "Response Groundedness": 0.70,
    "Ticket Resolution Rate": 0.00,
}


def load_golden_set() -> List[Dict[str, Any]]:
    """Load the golden set of test queries with expected labels."""
    if GOLDEN_SET_PATH.exists():
        with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def load_baseline() -> Dict[str, Any] | None:
    """Load the previous baseline report for comparison."""
    if BASELINE_REPORT_PATH.exists():
        with open(BASELINE_REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def run_evaluation(
    predictions: List[str],
    intents: List[str],
    escalations: List[bool],
    expected_escalations: List[bool],
    responses: List[str],
    contexts: List[List[str]],
    statuses: List[str],
) -> List[EvalResult]:
    """Run all evaluation metrics and return results."""
    results: List[EvalResult] = []
    results.append(intent_accuracy(predictions, intents))
    results.append(escalation_accuracy(escalations, expected_escalations))
    results.append(response_groundedness(responses, contexts))
    results.append(resolution_rate(statuses))
    return results


def compare_with_baseline(
    results: List[EvalResult], baseline: Dict[str, Any] | None
) -> List[str]:
    """Compare current scores with baseline. Return list of regressions."""
    if baseline is None:
        return []

    baseline_metrics = {r["metric"]: r["score"] for r in baseline.get("results", [])}
    regressions: List[str] = []

    for current in results:
        prev_score = baseline_metrics.get(current.name)
        if prev_score is not None:
            drop = prev_score - current.score
            if drop > 0.05:  # 5% drop threshold
                regressions.append(
                    f"{current.name}: dropped {drop:.2%} vs baseline "
                    f"({prev_score:.2f} → {current.score:.2f})"
                )
    return regressions


def save_results(
    results: List[EvalResult],
    output_path: Path = REGRESSION_REPORT_PATH,
    thresholds: Dict[str, float] = REGRESSION_THRESHOLDS,
    baseline: Dict[str, Any] | None = None,
    update_baseline: bool = False,
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
        "regressions": compare_with_baseline(results, baseline),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Update baseline if requested and all thresholds pass
    if update_baseline and report["overall_pass"]:
        with open(BASELINE_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("[BASELINE] Updated baseline report with current scores.")

    return report["overall_pass"]