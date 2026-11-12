"""Deterministic regression gate for CI.

This suite uses a mocked LLM and a deterministic retrieval stub. It validates
graph routing, escalation decisions, ticket status persistence and the
evaluation pipeline without external services. It is NOT a measurement of real
LLM or real retrieval quality; use ``scripts/run_real_eval.py`` for that.

Exits non-zero when a gated metric falls below its threshold.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.graph import build_graph  # noqa: E402
from app.agents.tools.ticket_db import TicketDB  # noqa: E402
from app.agents.tools.vector_store import RetrievedChunk, VectorStore  # noqa: E402
from app.evaluation.metrics import EvalResult, status_accuracy  # noqa: E402
from app.evaluation.metrics_v2 import (  # noqa: E402
    escalation_precision_recall,
    mrr,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.mock_llm import mock_llm_generate  # noqa: E402
from app.evaluation.suite import (  # noqa: E402
    compare_with_baseline,
    load_baseline,
    load_golden_set,
    run_evaluation,
    save_results,
)
from ingestion.knowledge_base import KNOWLEDGE_BASE  # noqa: E402

DOC_BY_ID = {document["id"]: document for document in KNOWLEDGE_BASE}
# Label lookup so the deterministic stub can return exactly the documents that
# answer a question, mirroring what the golden set expects.
LABELS_BY_QUERY = {
    item["query"]: item.get("relevant_doc_ids", [])
    for item in load_golden_set()
}

DETERMINISTIC_THRESHOLDS = {
    "Intent Accuracy": 0.50,
    "Escalation Precision": 0.55,
    "Escalation Recall": 0.90,
    "Status Accuracy (correctly routed)": 0.95,
}


def make_state(item: dict) -> dict:
    return {
        "messages": [],
        "user_query": item["query"],
        "intent": None,
        "context": [],
        "retrieved_doc_ids": [],
        "retrieval_scores": [],
        "retrieval_similarity": 0.0,
        "answer_confidence": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": "deterministic-eval",
        "response": "",
    }


async def stub_search_documents(
    self, query: str, top_k: int = 5, threshold: float = 0.0
) -> list[RetrievedChunk]:
    """Return the labelled documents for a query.

    Deterministic only. This proves that document IDs flow through the graph;
    it does not measure semantic retrieval quality.
    """
    chunks = []
    for doc_id in LABELS_BY_QUERY.get(query, [])[:top_k]:
        document = DOC_BY_ID[doc_id]
        chunks.append(
            RetrievedChunk(
                doc_id=f"{doc_id}#chunk-0",
                content=document["content"],
                similarity=0.9,
                metadata=document["metadata"],
            )
        )
    return chunks


def source_id(doc_id: str) -> str:
    return doc_id.split("#", 1)[0]


async def main() -> int:
    golden = load_golden_set()
    if not golden:
        print("ERROR: Golden set is empty.")
        return 1

    graph = build_graph()

    predictions: list[str] = []
    expected_intents: list[str] = []
    escalations: list[bool] = []
    expected_escalations: list[bool] = []
    responses: list[str] = []
    contexts: list[list[str]] = []
    statuses: list[str] = []
    retrieved_ids: list[list[str]] = []
    relevant_ids: list[list[str]] = []
    correctly_routed_statuses: list[str] = []
    correctly_routed_expected: list[str] = []

    with patch(
        "app.services.llm.LLMService.generate",
        new=AsyncMock(side_effect=mock_llm_generate),
    ), patch.object(
        VectorStore, "search_documents", new=stub_search_documents
    ), patch.object(
        TicketDB, "get_or_create_ticket", new=AsyncMock(return_value=999)
    ), patch.object(
        TicketDB, "add_conversation", new=AsyncMock(return_value=1)
    ):
        for item in golden:
            result = await graph.ainvoke(make_state(item))

            predicted = result.get("intent", "general")
            expected = item["expected_intent"]
            predictions.append(predicted)
            expected_intents.append(expected)
            escalations.append(bool(result.get("escalation_flag", False)))
            expected_escalations.append(bool(item["expected_escalate"]))
            responses.append(result.get("response", ""))
            contexts.append(result.get("context", []))
            statuses.append(result.get("status", "open"))
            retrieved_ids.append(
                [source_id(doc_id) for doc_id in result.get("retrieved_doc_ids", [])]
            )
            relevant_ids.append(item.get("relevant_doc_ids", []))

            if predicted == expected:
                correctly_routed_statuses.append(result.get("status", "open"))
                correctly_routed_expected.append(item["expected_status"])

    results = run_evaluation(
        predictions,
        expected_intents,
        escalations,
        expected_escalations,
        responses,
        contexts,
        statuses,
        expected_statuses=[item["expected_status"] for item in golden],
    )

    v2_results = [
        precision_at_k(retrieved_ids, relevant_ids, k=3),
        recall_at_k(retrieved_ids, relevant_ids, k=3),
        mrr(retrieved_ids, relevant_ids),
    ]
    v2_results.extend(
        escalation_precision_recall(escalations, expected_escalations).values()
    )
    v2_results.append(
        status_accuracy(correctly_routed_statuses, correctly_routed_expected)
    )
    # Rename the conditional status metric so it cannot be mistaken for the
    # unconditional status metric.
    v2_results[-1].name = "Status Accuracy (correctly routed)"

    all_results: list[EvalResult] = results + v2_results
    baseline = load_baseline()
    regressions = compare_with_baseline(all_results, baseline)

    gated = list(DETERMINISTIC_THRESHOLDS)
    gated_results = [r for r in all_results if r.name in gated]
    passed = save_results(
        gated_results,
        thresholds=DETERMINISTIC_THRESHOLDS,
        baseline=baseline,
    )

    report_path = ROOT / "regression_report.json"
    report = {
        "suite": "deterministic",
        "retrieval_backend": "stub",
        "note": (
            "Deterministic gate. Retrieval and generation are mocked; retrieval "
            "metrics validate identifier plumbing, not semantic quality."
        ),
        "gated_metrics": [r.name for r in gated_results],
        "all_metrics": [
            {"metric": r.name, "score": round(r.score, 4)} for r in all_results
        ],
        "regressions": regressions,
        "overall_pass": passed and not regressions,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if regressions:
        print("\n[WARN] Regressions detected vs baseline:")
        for regression in regressions:
            print(f"  - {regression}")

    if passed and not regressions:
        print("\n[PASS] Deterministic regression check passed.")
        return 0
    print("\n[FAIL] Deterministic regression check failed.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
