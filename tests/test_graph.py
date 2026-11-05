import pytest
from unittest.mock import AsyncMock, patch
from app.agents.nodes.intent_classifier import IntentClassifierNode
from app.agents.nodes.escalation_handler import EscalationHandlerNode
from app.agents.state import SupportState


@pytest.mark.asyncio
async def test_intent_classifier_refund():
    node = IntentClassifierNode()
    state = SupportState(user_query="I want a refund for my order", session_id="s1")
    with patch.object(node.llm, "generate", AsyncMock(return_value="refund_request")):
        result = await node.run(state)
    assert result.intent == "refund_request"


@pytest.mark.asyncio
async def test_intent_classifier_billing():
    node = IntentClassifierNode()
    state = SupportState(user_query="I was charged twice", session_id="s1")
    with patch.object(node.llm, "generate", AsyncMock(return_value="billing")):
        result = await node.run(state)
    assert result.intent == "billing"


@pytest.mark.asyncio
async def test_escalation_handler_refund():
    node = EscalationHandlerNode()
    state = SupportState(
        user_query="refund",
        intent="refund_request",
        confidence_score=0.9,
        session_id="s1",
    )
    assert node.should_escalate(state) is True


@pytest.mark.asyncio
async def test_escalation_handler_low_confidence():
    """Confidence alone does not trigger escalation in v1 (intent-based only)."""
    node = EscalationHandlerNode()
    state = SupportState(
        user_query="unknown question",
        intent="general",
        confidence_score=0.3,
        session_id="s1",
    )
    assert node.should_escalate(state) is False


@pytest.mark.asyncio
async def test_escalation_handler_no_escalate():
    node = EscalationHandlerNode()
    state = SupportState(
        user_query="hello",
        intent="general",
        confidence_score=0.9,
        session_id="s1",
    )
    assert node.should_escalate(state) is False