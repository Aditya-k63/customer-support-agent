"""Regression tests: ensures previously-verified behavior does not degrade.

These tests run the agent against the golden set with a mocked LLM so results
are deterministic and independent of external API variability.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.graph import build_graph


GOLDEN_SET_PATH = Path("app/evaluation/golden_set.json")


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_state(query: str, intent: str | None = None) -> dict:
    """Build a valid SupportState dict for graph invocation."""
    return {
        "messages": [],
        "user_query": query,
        "intent": intent,
        "context": [],
        "confidence_score": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": "regression-session",
        "response": "",
    }


# Query -> expected intent mapping for mocked LLM responses
MOCK_INTENT_MAP: dict[str, str] = {
    "I want to refund my order": "refund_request",
    "How do I reset my password?": "technical",
    "When will my invoice be generated?": "billing",
    "What are your business hours?": "general",
    "I was charged twice for the same item": "billing",
    "Can I speak to a human agent?": "human_handoff",
    "How do I track my shipment?": "general",
    "I need to update my billing address": "billing",
    "The website is not loading": "technical",
    "I want to cancel my subscription": "refund_request",
}


def mock_llm_generate(*args, **kwargs) -> str:
    """Deterministic mock: classify intent from the user query in the prompt."""
    # LLMService.generate(self, prompt, ...) -> args[1] is the prompt string
    prompt = args[1] if len(args) > 1 else (args[0] if args else "")
    for query, intent in MOCK_INTENT_MAP.items():
        if query in prompt:
            return intent
    return "general"


@pytest.mark.asyncio
async def test_regression_intent_classification():
    """Refund/billing/technical/general queries must keep their intent labels."""
    golden = load_golden_set()
    graph = build_graph()

    for item in golden:
        state = make_state(item["query"])
        with patch(
            "app.services.llm.LLMService.generate",
            new=AsyncMock(side_effect=mock_llm_generate),
        ):
            result = await graph.ainvoke(state)
        predicted = result.get("intent", "general")
        expected = item["expected_intent"]

        # Regression: intent must match golden set exactly
        assert predicted == expected, (
            f"Intent regression for query '{item['query']}': "
            f"expected {expected}, got {predicted}"
        )


@pytest.mark.asyncio
async def test_regression_escalation_logic():
    """Refund and human_handoff intents must always escalate."""
    golden = load_golden_set()
    graph = build_graph()

    for item in golden:
        if item["expected_escalate"]:
            state = make_state(item["query"])
            with patch(
                "app.services.llm.LLMService.generate",
                new=AsyncMock(side_effect=mock_llm_generate),
            ):
                result = await graph.ainvoke(state)
            assert result.get("escalation_flag") is True, (
                f"Escalation regression for query '{item['query']}'"
            )


@pytest.mark.asyncio
async def test_regression_ticket_status_non_escalated():
    """Non-escalated tickets must reach 'resolved' status."""
    golden = load_golden_set()
    graph = build_graph()

    for item in golden:
        if not item["expected_escalate"]:
            state = make_state(item["query"], intent=item["expected_intent"])
            with patch(
                "app.services.llm.LLMService.generate",
                new=AsyncMock(return_value="Here is your answer."),
            ):
                result = await graph.ainvoke(state)
            assert result.get("status") == "resolved", (
                f"Status regression for query '{item['query']}': "
                f"expected resolved, got {result.get('status')}"
            )