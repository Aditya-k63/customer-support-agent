"""Deterministic regression tests for graph behaviour.

The LLM and vector store are mocked so results are stable and do not depend on
external APIs, PostgreSQL or the embedding model.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.graph import build_graph
from app.evaluation.mock_llm import mock_llm_generate

GOLDEN_SET_PATH = Path("app/evaluation/golden_set.json")


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_state(query: str, intent: str | None = None) -> dict:
    return {
        "messages": [],
        "user_query": query,
        "intent": intent,
        "context": [],
        "retrieved_doc_ids": [],
        "retrieval_scores": [],
        "retrieval_similarity": 0.0,
        "answer_confidence": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": "regression-session",
        "response": "",
    }


async def mock_search_documents(*args, **kwargs) -> list:
    return []


def graph_patches(llm):
    return (
        patch("app.services.llm.LLMService.generate", new=llm),
        patch("app.agents.tools.vector_store.VectorStore.search_documents", new=mock_search_documents),
        patch("app.agents.tools.ticket_db.TicketDB.get_or_create_ticket", new=AsyncMock(return_value=999)),
        patch("app.agents.tools.ticket_db.TicketDB.add_conversation", new=AsyncMock(return_value=1)),
    )


@pytest.mark.asyncio
async def test_regression_intent_classification():
    """Intent classification accuracy must stay at or above the v1 baseline."""
    golden = load_golden_set()
    graph = build_graph()
    correct = 0
    failures = []

    llm, search, ticket, conversation = graph_patches(
        AsyncMock(side_effect=mock_llm_generate)
    )
    with llm, search, ticket, conversation:
        for item in golden:
            result = await graph.ainvoke(make_state(item["query"]))
            predicted = result.get("intent", "general")
            expected = item["expected_intent"]
            if predicted == expected:
                correct += 1
            else:
                failures.append(
                    f"  '{item['query']}': expected {expected}, got {predicted}"
                )

    accuracy = correct / len(golden) if golden else 0.0
    threshold = 0.50
    assert accuracy >= threshold, (
        f"Intent accuracy regression: {accuracy:.2%} ({correct}/{len(golden)}) "
        f"is below the v1 baseline threshold of {threshold:.0%}.\n"
        + "\n".join(failures[:10])
    )


@pytest.mark.asyncio
async def test_regression_escalation_logic():
    """Queries labelled as escalation must escalate."""
    golden = load_golden_set()
    graph = build_graph()

    llm, search, ticket, conversation = graph_patches(
        AsyncMock(side_effect=mock_llm_generate)
    )
    with llm, search, ticket, conversation:
        for item in golden:
            if not item["expected_escalate"]:
                continue
            result = await graph.ainvoke(make_state(item["query"]))
            assert result.get("escalation_flag") is True, (
                f"Escalation regression for query '{item['query']}'"
            )


@pytest.mark.asyncio
async def test_regression_ticket_status_non_escalated():
    """Non-escalated tickets must be persisted as 'resolved'."""
    golden = load_golden_set()
    graph = build_graph()

    for item in golden:
        if item["expected_escalate"]:
            continue
        llm, search, ticket, conversation = graph_patches(
            AsyncMock(return_value=item["expected_intent"])
        )
        with llm, search, ticket, conversation:
            result = await graph.ainvoke(make_state(item["query"]))
        assert result.get("status") == "resolved", (
            f"Status regression for query '{item['query']}': "
            f"expected resolved, got {result.get('status')}"
        )
