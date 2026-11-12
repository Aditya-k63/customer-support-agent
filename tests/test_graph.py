import sys
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.nodes.escalation_handler import EscalationHandlerNode
from app.agents.nodes.intent_classifier import IntentClassifierNode
from app.agents.nodes.rag_retriever import RAGRetrieverNode
from app.agents.nodes.response_generator import (
    ResponseGeneratorNode,
    estimate_answer_confidence,
)
from app.agents.nodes.ticket_manager import TicketManagerNode
from app.agents.state import SupportState
from app.agents.tools.vector_store import RetrievedChunk
from app.services.embeddings import (
    EmbeddingModelUnavailableError,
    EmbeddingService,
)


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
        user_query="refund", intent="refund_request", session_id="s1"
    )
    assert node.should_escalate(state) is True


@pytest.mark.asyncio
async def test_escalation_handler_no_escalate():
    node = EscalationHandlerNode()
    state = SupportState(user_query="hello", intent="general", session_id="s1")
    assert node.should_escalate(state) is False


@pytest.mark.asyncio
async def test_rag_retriever_retrieves_once_and_passes_ids():
    node = RAGRetrieverNode()
    chunks = [
        RetrievedChunk(doc_id="refund_policy#chunk-0", content="Refunds", similarity=0.8)
    ]
    with patch.object(
        node.vector_store, "search_documents", AsyncMock(return_value=chunks)
    ) as search:
        state = await node.run(SupportState(user_query="refund", session_id="s1"))

    search.assert_awaited_once()
    assert state.context == ["Refunds"]
    assert state.retrieved_doc_ids == ["refund_policy#chunk-0"]
    assert state.retrieval_similarity == 0.8


@pytest.mark.asyncio
async def test_response_generator_does_not_retrieve_again():
    node = ResponseGeneratorNode()
    state = SupportState(
        user_query="How do I get a refund?",
        context=["Refund policy: Items can be refunded within 30 days."],
        retrieval_similarity=0.81,
        session_id="s1",
    )
    with patch.object(
        node.llm,
        "generate",
        AsyncMock(return_value="Items can be refunded within 30 days."),
    ):
        result = await node.run(state)

    assert result.response == "Items can be refunded within 30 days."
    assert result.retrieval_similarity == 0.81
    assert result.answer_confidence > 0.5
    # Retrieval similarity and answer confidence are distinct signals.
    assert result.answer_confidence != result.retrieval_similarity


def test_estimate_answer_confidence_unrelated_is_low():
    score = estimate_answer_confidence(
        "Bananas are yellow and grow in tropical climates.",
        ["Refund policy: items can be refunded within 30 days."],
    )
    assert score < 0.2


@pytest.mark.asyncio
async def test_ticket_manager_writes_final_status_before_persisting():
    captured = {}

    class FakeTicketDB:
        async def get_or_create_ticket(self, **kwargs):
            captured.update(kwargs)
            return 7

        async def add_conversation(self, ticket_id, role, content):
            captured.setdefault("messages", []).append((role, content))
            return len(captured["messages"])

    with patch(
        "app.agents.nodes.ticket_manager.TicketDB", return_value=FakeTicketDB()
    ):
        node = TicketManagerNode()
        state = SupportState(
            user_query="I was charged twice",
            intent="billing",
            escalation_flag=True,
            response="Escalating to a human.",
            session_id="session-x",
        )
        result = await node.run(state)

    assert captured["status"] == "in_progress"
    assert result.status == "in_progress"
    assert result.ticket_id == 7
    assert captured["messages"] == [
        ("user", "I was charged twice"),
        ("assistant", "Escalating to a human."),
    ]


@pytest.mark.asyncio
async def test_ticket_manager_resolves_non_escalated_ticket():
    captured = {}

    class FakeTicketDB:
        async def get_or_create_ticket(self, **kwargs):
            captured.update(kwargs)
            return 8

        async def add_conversation(self, ticket_id, role, content):
            return 1

    with patch(
        "app.agents.nodes.ticket_manager.TicketDB", return_value=FakeTicketDB()
    ):
        node = TicketManagerNode()
        state = SupportState(
            user_query="reset password",
            intent="technical",
            escalation_flag=False,
            response="Use the forgot password link.",
            session_id="session-y",
        )
        result = await node.run(state)

    assert captured["status"] == "resolved"
    assert result.status == "resolved"


def test_embedding_service_fails_loudly_without_model():
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(EmbeddingModelUnavailableError):
            EmbeddingService()._load_model()
