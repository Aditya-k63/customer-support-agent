"""Run full evaluation + regression check against the golden set.

Usage:
    python scripts/run_evaluation.py

Exits with code 1 if any regression threshold is not met (CI gate).
"""
import asyncio
import sys

from app.evaluation.suite import run_evaluation, save_results, load_golden_set
from app.agents.graph import build_graph


async def main() -> int:
    golden = load_golden_set()
    if not golden:
        print("ERROR: Golden set is empty.")
        return 1

    graph = build_graph()

    predictions: list[str] = []
    intents: list[str] = []
    escalations: list[bool] = []
    responses: list[str] = []
    contexts: list[list[str]] = []
    statuses: list[str] = []

    for item in golden:
        initial_state = {
            "messages": [],
            "user_query": item["query"],
            "intent": None,
            "context": [],
            "confidence_score": 0.0,
            "ticket_id": None,
            "status": "open",
            "escalation_flag": False,
            "session_id": "eval-session",
        }

        try:
            result = await graph.ainvoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR running graph for query '{item['query']}': {exc}")
            return 1

        predictions.append(result.get("intent", "general"))
        intents.append(item["expected_intent"])
        escalations.append(result.get("escalation_flag", False))
        responses.append(result.get("response", ""))
        contexts.append(result.get("context", []))
        statuses.append(result.get("status", "open"))

    results = run_evaluation(predictions, intents, escalations, responses, contexts, statuses)
    passed = save_results(results)

    if passed:
        print("\n✅ REGRESSION CHECK PASSED — all metrics meet thresholds.")
        return 0
    print("\n❌ REGRESSION CHECK FAILED — one or more metrics below threshold.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))