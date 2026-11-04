"""Regression tests: ensures previously-verified behavior does not degrade.

These tests run the agent against the golden set with a mocked LLM and
mocked TicketDB so results are deterministic and independent of external
API variability or database availability.
"""
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


# Keyword-based intent classification for mocked LLM (works with 150-question golden set)
# Synced with scripts/run_evaluation.py MOCK_INTENT_KEYWORDS
MOCK_INTENT_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "refund", "charge", "invoice", "payment", "bill", "subscription",
        "credit card", "promo", "tax", "currency", "deposit", "overcharged",
        "loyalty", "discount", "receipt", "payment method", "billing address",
        "payment due", "billing plan", "accidental charge", "billing history",
        "declined", "credit card details", "itemized invoice", "promo code",
        "miss a payment", "pause", "billing adjustment", "charged after",
        "tax receipt", "pending charge", "split payment", "tax information",
        "failed service", "overcharged", "billing support", "loyalty discount",
    ],
    "technical": [
        "password", "login", "crash", "app", "website", "404", "cache",
        "video", "upload", "search", "notification", "dashboard", "freezing",
        "export", "SSL", "domain", "checkout", "dark mode", "email", "API",
        "integration", "timeout", "cookie", "MFA", "download", "form",
        "not loading", "log into", "two-factor", "browser", "won't play",
        "reset my password", "can't log", "keeps crashing", "enable two-factor",
        "404 error", "clear my browser", "won't play", "update the app",
        "uploading files", "search function", "notification settings",
        "access my dashboard", "page keeps freezing", "export my data",
        "SSL error", "connect my domain", "checkout button", "enable dark mode",
        "verify my email", "API is returning", "set up integrations",
        "timeout error", "clear my cookies", "mobile app won't open",
        "reset my MFA", "download my files", "form won't submit",
    ],
    "general": [
        "hours", "shipment", "office", "contact", "return policy",
        "shipping", "address", "privacy", "account", "mobile app",
        "delete my account", "countries", "verify my identity", "student",
        "refer a friend", "terms of service", "unsubscribe", "live chat",
        "update my profile", "refund window", "enterprise", "get started",
        "free tier", "upgrade", "training", "feedback", "mission",
        "help center", "status page", "business hours", "track my shipment",
        "office located", "customer support", "ship internationally",
        "shipping address", "privacy policy", "create an account",
        "mobile app", "delete my account", "countries do you serve",
        "verify my identity", "student discounts", "refer a friend",
        "terms of service", "unsubscribe from emails", "live chat",
        "update my profile", "refund window", "enterprise plans",
        "get started", "features are included", "upgrade my account",
        "offer training", "provide feedback", "company mission",
        "find your help center", "status page",
    ],
    "refund_request": [
        "refund", "cancel", "money back", "defective", "dispute",
        "return", "reverse", "transaction", "purchase", "opt out",
        "chargeback", "double billed", "store credit", "undo",
        "want to refund", "want to cancel", "defective product",
        "money back", "request a refund", "didn't use",
        "reverse this", "return my purchase", "long does a refund",
        "dispute this charge", "partial refund", "not satisfied",
        "i want to refund", "i want to cancel", "can i get a refund",
        "how do i request a refund", "i was charged for",
        "can i reverse", "i want to return", "how long does",
        "i want to dispute", "can i get a partial", "i'm not satisfied",
        "can i reverse a completed", "i want to opt out",
        "how do i initiate", "i was double billed",
        "can i get store credit", "i want to undo",
        "the product never arrived", "can i escalate",
        "i want to revoke", "how do i file",
        "i was charged without", "can i get my deposit",
        "i want to cancel and refund", "the service was not",
        "can i speak to someone", "i need an emergency",
        "how do i check my refund", "i want to reverse a recurring",
    ],
    "human_handoff": [
        "human agent", "real person", "transfer", "supervisor",
        "urgent", "manager", "review", "live agent", "bot",
        "call back", "customer service", "person", "now",
        "support staff", "human", "talk to someone",
        "phone number", "chat", "contact me directly",
        "human intervention", "routed", "in charge",
        "live help", "fix this", "transferred", "customer care",
        "speak to a human", "real person", "transfer me",
        "supervisor", "urgent", "manager", "review my case",
        "can i speak to a human", "i need to talk",
        "can you transfer", "i want to speak",
        "this is urgent", "can i escalate",
        "i need a human", "please connect me",
        "i have a complex", "can i get a live",
        "i don't want to talk to a bot",
        "can a human call", "i need to speak",
        "can you transfer this", "i need help from",
        "can i get a live agent", "i don't want",
        "can a human call me", "i need to speak to",
        "can you transfer", "i want to speak",
        "can i get a phone", "i want to chat",
        "can someone contact", "i need human",
        "can you route", "i want to speak",
        "can i get live", "i need a human",
        "can i be transferred", "i want to talk",
    ],
}


def mock_llm_generate(*args, **kwargs) -> str:
    """Deterministic mock: keyword-based intent classification."""
    prompt = args[1] if len(args) > 1 else (args[0] if args else "")
    prompt_lower = prompt.lower()

    best_intent = "general"
    best_score = 0
    for intent, keywords in MOCK_INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


async def mock_create_ticket(*args, **kwargs) -> int:
    """Mock ticket creation that returns a fake ID without touching the DB."""
    return 999


async def mock_add_conversation(*args, **kwargs) -> int:
    """Mock conversation creation that returns a fake ID without touching the DB."""
    return 999


@pytest.mark.asyncio
async def test_regression_intent_classification():
    """Intent classification accuracy must stay at or above v1 baseline threshold."""
    golden = load_golden_set()
    graph = build_graph()

    correct = 0
    total = len(golden)
    failures = []

    for item in golden:
        state = make_state(item["query"])
        with patch(
            "app.services.llm.LLMService.generate",
            new=AsyncMock(side_effect=mock_llm_generate),
        ), patch(
            "app.agents.tools.ticket_db.TicketDB.create_ticket",
            new=mock_create_ticket,
        ), patch(
            "app.agents.tools.ticket_db.TicketDB.add_conversation",
            new=mock_add_conversation,
        ):
            result = await graph.ainvoke(state)
        predicted = result.get("intent", "general")
        expected = item["expected_intent"]

        if predicted == expected:
            correct += 1
        else:
            failures.append(
                f"  '{item['query']}': expected {expected}, got {predicted}"
            )

    accuracy = correct / total if total else 0.0
    threshold = 0.50  # v1 baseline threshold

    assert accuracy >= threshold, (
        f"Intent accuracy regression: {accuracy:.2%} ({correct}/{total}) "
        f"is below v1 baseline threshold of {threshold:.0%}.\n"
        + "\n".join(failures[:10])
        + (f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else "")
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
            ), patch(
                "app.agents.tools.ticket_db.TicketDB.create_ticket",
                new=mock_create_ticket,
            ), patch(
                "app.agents.tools.ticket_db.TicketDB.add_conversation",
                new=mock_add_conversation,
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
            ), patch(
                "app.agents.tools.ticket_db.TicketDB.create_ticket",
                new=mock_create_ticket,
            ), patch(
                "app.agents.tools.ticket_db.TicketDB.add_conversation",
                new=mock_add_conversation,
            ):
                result = await graph.ainvoke(state)
            assert result.get("status") == "resolved", (
                f"Status regression for query '{item['query']}': "
                f"expected resolved, got {result.get('status')}"
            )